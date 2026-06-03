from __future__ import annotations

import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.training.policy import MaskedActorCritic, to_cpu_state_dict
from src.training.ppo_trainer import (
    action_label,
    append_jsonl,
    resolve_device,
    summarize_action_distribution,
    summarize_training_episodes,
    update_policy,
)
from src.training.spaces import ACTION_DIM, team_of
from src.training.team_env import DEFAULT_CONTROLLED_PLAYERS, TeamTrainingEnv, TeamTrainingEnvConfig


@dataclass
class TeamPPOConfig:
    seed: int = 20260603
    controlled_players: tuple[str, ...] = DEFAULT_CONTROLLED_PLAYERS
    opponent_bot: str = "scripted"
    ally_bot: str = "scripted"
    max_ticks: int = 300
    workers: int = field(default_factory=lambda: max(1, os.cpu_count() or 1))
    rollout_steps: int = 300
    updates: int = 10
    ppo_epochs: int = 4
    minibatch_size: int = 1024
    hidden_size: int = 128
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    eval_episodes: int = 50
    eval_interval: int = 1
    checkpoint_interval: int = 1
    torch_threads: int = 1
    device: str = "auto"

    def __post_init__(self) -> None:
        self.controlled_players = tuple(self.controlled_players)
        if not self.controlled_players:
            raise ValueError("TeamPPOConfig requires at least one controlled player")


def _env_config(config: TeamPPOConfig, seed: int) -> TeamTrainingEnvConfig:
    return TeamTrainingEnvConfig(
        controlled_players=config.controlled_players,
        seed=seed,
        opponent_bot=config.opponent_bot,
        ally_bot=config.ally_bot,
        max_ticks=config.max_ticks,
        record_replay=False,
        record_trace=False,
    )


def make_team_policies(config: TeamPPOConfig, device: torch.device | str = "cpu") -> dict[str, MaskedActorCritic]:
    return {
        player_id: MaskedActorCritic(hidden_size=config.hidden_size).to(device)
        for player_id in config.controlled_players
    }


def make_team_optimizers(
    policies: dict[str, MaskedActorCritic],
    config: TeamPPOConfig,
) -> dict[str, torch.optim.Optimizer]:
    return {
        player_id: torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        for player_id, policy in policies.items()
    }


def load_single_policy_into_team(
    checkpoint_path: Path,
    policies: dict[str, MaskedActorCritic],
    map_location: str | torch.device = "cpu",
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    policy_state = checkpoint["policy_state"]
    for policy in policies.values():
        policy.load_state_dict(policy_state)


def _team_episode_summary(
    return_sums: dict[str, float],
    shaping_sums: dict[str, float],
    terminal_sums: dict[str, float],
    steps: int,
    winner: str,
    winner_type: str,
    controlled_team: str,
) -> dict:
    return {
        "return": float(np.mean(list(return_sums.values()))) if return_sums else 0.0,
        "shaping": float(np.mean(list(shaping_sums.values()))) if shaping_sums else 0.0,
        "terminal": float(np.mean(list(terminal_sums.values()))) if terminal_sums else 0.0,
        "steps": int(steps),
        "winner": winner,
        "winner_type": winner_type,
        "win": 1 if winner == controlled_team else 0,
        "loss": 1 if winner not in {controlled_team, "DRAW", None} else 0,
        "draw": 1 if winner in {"DRAW", None} else 0,
        "normal_win": 1 if winner == controlled_team and winner_type == "normal" else 0,
        "death_time_win": 1 if winner == controlled_team and winner_type == "death_time" else 0,
    }


def _empty_agent_buffers(config: TeamPPOConfig) -> dict[str, dict]:
    return {
        player_id: {
            "observations": [],
            "masks": [],
            "actions": [],
            "logprobs": [],
            "values": [],
            "rewards": [],
            "dones": [],
            "entropies": [],
        }
        for player_id in config.controlled_players
    }


def _collect_rollout_worker(request: dict) -> dict:
    torch.set_num_threads(int(request["torch_threads"]))
    seed = int(request["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = TeamPPOConfig(**request["config"])
    policies = make_team_policies(config)
    for player_id, policy_state in request["policy_states"].items():
        policies[player_id].load_state_dict(policy_state)
        policies[player_id].eval()

    env = TeamTrainingEnv(_env_config(config, seed))
    observations, _ = env.reset(seed=seed)
    controlled_team = env.controlled_team

    buffers = _empty_agent_buffers(config)
    episodes = []
    return_sums = {player_id: 0.0 for player_id in config.controlled_players}
    shaping_sums = {player_id: 0.0 for player_id in config.controlled_players}
    terminal_sums = {player_id: 0.0 for player_id in config.controlled_players}
    episode_steps = 0

    for step_index in range(config.rollout_steps):
        masks = env.action_masks()
        actions = {}
        step_outputs = {}
        for player_id in config.controlled_players:
            step = policies[player_id].act(observations[player_id], masks[player_id], deterministic=False)
            actions[player_id] = step.action
            step_outputs[player_id] = step

        result = env.step(actions)
        done = bool(result.terminal or result.truncated)

        for player_id in config.controlled_players:
            step = step_outputs[player_id]
            reward_components = result.info.get("reward_components", {}).get(player_id, {}) or {}
            buffers[player_id]["observations"].append(observations[player_id])
            buffers[player_id]["masks"].append(masks[player_id])
            buffers[player_id]["actions"].append(step.action)
            buffers[player_id]["logprobs"].append(step.logprob)
            buffers[player_id]["values"].append(step.value)
            buffers[player_id]["rewards"].append(float(result.rewards[player_id]))
            buffers[player_id]["dones"].append(done)
            buffers[player_id]["entropies"].append(step.entropy)
            return_sums[player_id] += float(result.rewards[player_id])
            shaping_sums[player_id] += float(reward_components.get("shaping", 0.0) or 0.0)
            terminal_sums[player_id] += float(reward_components.get("terminal", 0.0) or 0.0)

        observations = result.observations
        episode_steps += 1
        if done:
            episodes.append(_team_episode_summary(
                return_sums=return_sums,
                shaping_sums=shaping_sums,
                terminal_sums=terminal_sums,
                steps=episode_steps,
                winner=result.info.get("winner") or "DRAW",
                winner_type=result.info.get("winner_type") or "unknown",
                controlled_team=controlled_team,
            ))
            reset_seed = seed + 10_000 + step_index
            observations, _ = env.reset(seed=reset_seed)
            controlled_team = env.controlled_team
            return_sums = {player_id: 0.0 for player_id in config.controlled_players}
            shaping_sums = {player_id: 0.0 for player_id in config.controlled_players}
            terminal_sums = {player_id: 0.0 for player_id in config.controlled_players}
            episode_steps = 0

    last_values = {}
    last_masks = env.action_masks()
    for player_id in config.controlled_players:
        last_values[player_id] = policies[player_id].act(
            observations[player_id],
            last_masks[player_id],
            deterministic=True,
        ).value

    return {
        "agents": {
            player_id: {
                key: np.asarray(value, dtype=_buffer_dtype(key))
                for key, value in buffers[player_id].items()
            }
            for player_id in config.controlled_players
        },
        "last_values": last_values,
        "episodes": episodes,
    }


def _buffer_dtype(key: str):
    if key in {"actions"}:
        return np.int64
    if key in {"dones", "masks"}:
        return np.bool_
    return np.float32


def _compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for index in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - float(dones[index])
        next_value = last_value if index == len(rewards) - 1 else values[index + 1]
        delta = rewards[index] + gamma * next_value * next_non_terminal - values[index]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[index] = last_gae
    returns = advantages + values
    return advantages, returns.astype(np.float32)


def _merge_agent_rollouts(worker_results: list[dict], config: TeamPPOConfig, player_id: str) -> dict:
    advantages = []
    returns = []
    for result in worker_results:
        agent = result["agents"][player_id]
        worker_advantages, worker_returns = _compute_gae(
            agent["rewards"],
            agent["dones"],
            agent["values"],
            float(result["last_values"][player_id]),
            config.gamma,
            config.gae_lambda,
        )
        advantages.append(worker_advantages)
        returns.append(worker_returns)

    return {
        "observations": np.concatenate([item["agents"][player_id]["observations"] for item in worker_results]),
        "masks": np.concatenate([item["agents"][player_id]["masks"] for item in worker_results]),
        "actions": np.concatenate([item["agents"][player_id]["actions"] for item in worker_results]),
        "old_logprobs": np.concatenate([item["agents"][player_id]["logprobs"] for item in worker_results]),
        "values": np.concatenate([item["agents"][player_id]["values"] for item in worker_results]),
        "rewards": np.concatenate([item["agents"][player_id]["rewards"] for item in worker_results]),
        "dones": np.concatenate([item["agents"][player_id]["dones"] for item in worker_results]),
        "entropies": np.concatenate([item["agents"][player_id]["entropies"] for item in worker_results]),
        "advantages": np.concatenate(advantages),
        "returns": np.concatenate(returns),
    }


def collect_team_rollouts(
    policies: dict[str, MaskedActorCritic],
    config: TeamPPOConfig,
    update: int,
    pool: Optional[mp.pool.Pool] = None,
) -> dict:
    policy_states = {
        player_id: to_cpu_state_dict(policies[player_id])
        for player_id in config.controlled_players
    }
    requests = [
        {
            "seed": config.seed + update * 1_000_000 + worker_id * 10_000,
            "config": asdict(config),
            "policy_states": policy_states,
            "torch_threads": config.torch_threads,
        }
        for worker_id in range(config.workers)
    ]
    if config.workers == 1:
        worker_results = [_collect_rollout_worker(requests[0])]
    else:
        assert pool is not None
        worker_results = pool.map(_collect_rollout_worker, requests)

    episodes = []
    for result in worker_results:
        episodes.extend(result["episodes"])

    agents = {
        player_id: _merge_agent_rollouts(worker_results, config, player_id)
        for player_id in config.controlled_players
    }
    return {"agents": agents, "episodes": episodes}


def update_team_policies(
    policies: dict[str, MaskedActorCritic],
    optimizers: dict[str, torch.optim.Optimizer],
    rollout: dict,
    config: TeamPPOConfig,
    device: torch.device,
) -> dict:
    return {
        player_id: update_policy(
            policies[player_id],
            optimizers[player_id],
            rollout["agents"][player_id],
            config,
            device,
        )
        for player_id in config.controlled_players
    }


def summarize_team_rollout(rollout: dict, config: TeamPPOConfig) -> dict:
    summary = summarize_training_episodes(rollout["episodes"])
    summary["agents"] = {}
    for player_id in config.controlled_players:
        agent = rollout["agents"][player_id]
        summary["agents"][player_id] = {
            "mean_reward": round(float(np.mean(agent["rewards"])) if len(agent["rewards"]) else 0.0, 6),
            "mean_entropy": round(float(np.mean(agent["entropies"])) if len(agent["entropies"]) else 0.0, 6),
            "action_distribution": summarize_action_distribution(agent["actions"], agent["masks"]),
        }
    return summary


@torch.no_grad()
def evaluate_team_policies(
    policies: dict[str, MaskedActorCritic],
    config: TeamPPOConfig,
    episodes: int,
    seed: int,
    deterministic: bool,
    device: torch.device,
) -> dict:
    for policy in policies.values():
        policy.eval()

    episode_summaries = []
    actions_by_player = {player_id: [] for player_id in config.controlled_players}
    masks_by_player = {player_id: [] for player_id in config.controlled_players}
    entropies_by_player = {player_id: [] for player_id in config.controlled_players}

    for episode_index in range(episodes):
        env_seed = seed + episode_index
        env = TeamTrainingEnv(_env_config(config, env_seed))
        observations, _ = env.reset(seed=env_seed)
        done = False
        return_sums = {player_id: 0.0 for player_id in config.controlled_players}
        shaping_sums = {player_id: 0.0 for player_id in config.controlled_players}
        terminal_sums = {player_id: 0.0 for player_id in config.controlled_players}
        steps = 0

        while not done:
            masks = env.action_masks()
            selected_actions = {}
            for player_id in config.controlled_players:
                step = policies[player_id].act(
                    observations[player_id],
                    masks[player_id],
                    deterministic=deterministic,
                    device=device,
                )
                selected_actions[player_id] = step.action
                actions_by_player[player_id].append(step.action)
                masks_by_player[player_id].append(masks[player_id])
                entropies_by_player[player_id].append(step.entropy)

            result = env.step(selected_actions)
            for player_id in config.controlled_players:
                reward_components = result.info.get("reward_components", {}).get(player_id, {}) or {}
                return_sums[player_id] += float(result.rewards[player_id])
                shaping_sums[player_id] += float(reward_components.get("shaping", 0.0) or 0.0)
                terminal_sums[player_id] += float(reward_components.get("terminal", 0.0) or 0.0)
            observations = result.observations
            steps += 1
            done = bool(result.terminal or result.truncated)

        episode_summaries.append(_team_episode_summary(
            return_sums=return_sums,
            shaping_sums=shaping_sums,
            terminal_sums=terminal_sums,
            steps=steps,
            winner=result.info.get("winner") or "DRAW",
            winner_type=result.info.get("winner_type") or "unknown",
            controlled_team=env.controlled_team,
        ))

    summary = summarize_training_episodes(episode_summaries)
    summary["mode"] = "argmax" if deterministic else "stochastic"
    summary["agents"] = {}
    for player_id in config.controlled_players:
        entropies = entropies_by_player[player_id]
        summary["agents"][player_id] = {
            "mean_policy_entropy": round(float(np.mean(entropies)) if entropies else 0.0, 6),
            "action_distribution": summarize_action_distribution(actions_by_player[player_id], masks_by_player[player_id]),
        }
    summary["episodes_detail"] = episode_summaries
    return summary


def evaluate_random_legal_team(config: TeamPPOConfig, episodes: int, seed: int) -> dict:
    episode_summaries = []
    actions_by_player = {player_id: [] for player_id in config.controlled_players}
    masks_by_player = {player_id: [] for player_id in config.controlled_players}
    rng = np.random.default_rng(seed)

    for episode_index in range(episodes):
        env_seed = seed + episode_index
        env = TeamTrainingEnv(_env_config(config, env_seed))
        _, _ = env.reset(seed=env_seed)
        done = False
        return_sums = {player_id: 0.0 for player_id in config.controlled_players}
        shaping_sums = {player_id: 0.0 for player_id in config.controlled_players}
        terminal_sums = {player_id: 0.0 for player_id in config.controlled_players}
        steps = 0

        while not done:
            masks = env.action_masks()
            selected_actions = {}
            for player_id, mask in masks.items():
                legal = np.flatnonzero(mask)
                action = int(rng.choice(legal)) if len(legal) else 0
                selected_actions[player_id] = action
                actions_by_player[player_id].append(action)
                masks_by_player[player_id].append(mask)
            result = env.step(selected_actions)
            for player_id in config.controlled_players:
                reward_components = result.info.get("reward_components", {}).get(player_id, {}) or {}
                return_sums[player_id] += float(result.rewards[player_id])
                shaping_sums[player_id] += float(reward_components.get("shaping", 0.0) or 0.0)
                terminal_sums[player_id] += float(reward_components.get("terminal", 0.0) or 0.0)
            steps += 1
            done = bool(result.terminal or result.truncated)

        episode_summaries.append(_team_episode_summary(
            return_sums=return_sums,
            shaping_sums=shaping_sums,
            terminal_sums=terminal_sums,
            steps=steps,
            winner=result.info.get("winner") or "DRAW",
            winner_type=result.info.get("winner_type") or "unknown",
            controlled_team=env.controlled_team,
        ))

    summary = summarize_training_episodes(episode_summaries)
    summary["mode"] = "random_legal_team"
    summary["agents"] = {
        player_id: {
            "action_distribution": summarize_action_distribution(actions_by_player[player_id], masks_by_player[player_id]),
        }
        for player_id in config.controlled_players
    }
    summary["episodes_detail"] = episode_summaries
    return summary


def _random_legal_team_worker(request: dict) -> dict:
    np.random.seed(int(request["worker_seed"]))
    config = TeamPPOConfig(**request["config"])
    rng = np.random.default_rng(int(request["worker_seed"]))
    episode_summaries = []
    actions_by_player = {player_id: [] for player_id in config.controlled_players}
    masks_by_player = {player_id: [] for player_id in config.controlled_players}

    for env_seed in request["env_seeds"]:
        env = TeamTrainingEnv(_env_config(config, int(env_seed)))
        _, _ = env.reset(seed=int(env_seed))
        done = False
        return_sums = {player_id: 0.0 for player_id in config.controlled_players}
        shaping_sums = {player_id: 0.0 for player_id in config.controlled_players}
        terminal_sums = {player_id: 0.0 for player_id in config.controlled_players}
        steps = 0

        while not done:
            masks = env.action_masks()
            selected_actions = {}
            for player_id, mask in masks.items():
                legal = np.flatnonzero(mask)
                action = int(rng.choice(legal)) if len(legal) else 0
                selected_actions[player_id] = action
                actions_by_player[player_id].append(action)
                masks_by_player[player_id].append(mask)
            result = env.step(selected_actions)
            for player_id in config.controlled_players:
                reward_components = result.info.get("reward_components", {}).get(player_id, {}) or {}
                return_sums[player_id] += float(result.rewards[player_id])
                shaping_sums[player_id] += float(reward_components.get("shaping", 0.0) or 0.0)
                terminal_sums[player_id] += float(reward_components.get("terminal", 0.0) or 0.0)
            steps += 1
            done = bool(result.terminal or result.truncated)

        episode_summaries.append(_team_episode_summary(
            return_sums=return_sums,
            shaping_sums=shaping_sums,
            terminal_sums=terminal_sums,
            steps=steps,
            winner=result.info.get("winner") or "DRAW",
            winner_type=result.info.get("winner_type") or "unknown",
            controlled_team=env.controlled_team,
        ))

    return {
        "episodes": episode_summaries,
        "actions_by_player": actions_by_player,
        "masks_by_player": masks_by_player,
    }


def evaluate_random_legal_team_parallel(
    config: TeamPPOConfig,
    episodes: int,
    seed: int,
    workers: int,
) -> dict:
    if episodes <= 0:
        return evaluate_random_legal_team(config, episodes=0, seed=seed)
    if workers <= 1 or episodes <= 1:
        return evaluate_random_legal_team(config, episodes=episodes, seed=seed)

    env_seeds = [int(seed + index) for index in range(episodes)]
    chunk_count = max(1, min(int(workers), len(env_seeds)))
    chunks = [env_seeds[index::chunk_count] for index in range(chunk_count)]
    requests = [
        {
            "config": asdict(config),
            "env_seeds": chunk,
            "worker_seed": int(seed + worker_index * 9973),
        }
        for worker_index, chunk in enumerate(chunks)
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=chunk_count) as pool:
        results = pool.map(_random_legal_team_worker, requests)

    episode_summaries = []
    actions_by_player = {player_id: [] for player_id in config.controlled_players}
    masks_by_player = {player_id: [] for player_id in config.controlled_players}
    for result in results:
        episode_summaries.extend(result["episodes"])
        for player_id in config.controlled_players:
            actions_by_player[player_id].extend(result["actions_by_player"][player_id])
            masks_by_player[player_id].extend(result["masks_by_player"][player_id])

    summary = summarize_training_episodes(episode_summaries)
    summary["mode"] = "random_legal_team"
    summary["agents"] = {
        player_id: {
            "action_distribution": summarize_action_distribution(actions_by_player[player_id], masks_by_player[player_id]),
        }
        for player_id in config.controlled_players
    }
    summary["episodes_detail"] = episode_summaries
    return summary


def evaluate_scripted_control_team(config: TeamPPOConfig, episodes: int, seed: int) -> dict:
    from src.core.runner import run_match

    summaries = []
    controlled_team = team_of(config.controlled_players[0])
    for episode_index in range(episodes):
        result = run_match(seed=seed + episode_index, bot_type="scripted", mode="training")
        summaries.append(_team_episode_summary(
            return_sums={player_id: 0.0 for player_id in config.controlled_players},
            shaping_sums={player_id: 0.0 for player_id in config.controlled_players},
            terminal_sums={player_id: 0.0 for player_id in config.controlled_players},
            steps=result.total_ticks,
            winner=result.winner,
            winner_type=result.winner_type,
            controlled_team=controlled_team,
        ))
    summary = summarize_training_episodes(summaries)
    summary["mode"] = "scripted_control_team"
    summary["episodes_detail"] = summaries
    return summary


def save_team_checkpoint(
    path: Path,
    policies: dict[str, MaskedActorCritic],
    optimizers: dict[str, torch.optim.Optimizer],
    config: TeamPPOConfig,
    update: int,
    total_steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_states": {
                player_id: policy.state_dict()
                for player_id, policy in policies.items()
            },
            "optimizer_states": {
                player_id: optimizer.state_dict()
                for player_id, optimizer in optimizers.items()
            },
            "config": asdict(config),
            "update": int(update),
            "total_steps": int(total_steps),
        },
        path,
    )


def load_team_checkpoint(
    path: Path,
    policies: dict[str, MaskedActorCritic],
    optimizers: Optional[dict[str, torch.optim.Optimizer]] = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    for player_id, policy_state in checkpoint["policy_states"].items():
        if player_id in policies:
            policies[player_id].load_state_dict(policy_state)
    if optimizers is not None:
        for player_id, optimizer_state in checkpoint.get("optimizer_states", {}).items():
            if player_id in optimizers:
                optimizers[player_id].load_state_dict(optimizer_state)
    return checkpoint


def latest_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                result.append(json.loads(line))
    return result
