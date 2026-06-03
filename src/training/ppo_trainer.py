from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from src.training.env import TrainingEnv, TrainingEnvConfig
from src.training.policy import MaskedActorCritic, to_cpu_state_dict
from src.training.spaces import ACTION_DIM, ACTION_SPECS, OBSERVATION_DIM, OBSERVATION_VERSION, team_of


@dataclass
class PPOConfig:
    seed: int = 20260602
    controlled_player: str = "A1"
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


@dataclass
class EpisodeMetrics:
    return_sum: float = 0.0
    shaping_sum: float = 0.0
    terminal_sum: float = 0.0
    steps: int = 0
    winner: Optional[str] = None
    winner_type: Optional[str] = None


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _env_config(config: PPOConfig, seed: int) -> TrainingEnvConfig:
    return TrainingEnvConfig(
        controlled_player=config.controlled_player,
        seed=seed,
        opponent_bot=config.opponent_bot,
        ally_bot=config.ally_bot,
        max_ticks=config.max_ticks,
        record_replay=False,
        record_trace=False,
    )


def _episode_summary(metrics: EpisodeMetrics, controlled_player: str) -> dict:
    own_team = team_of(controlled_player)
    winner = metrics.winner or "DRAW"
    winner_type = metrics.winner_type or "unknown"
    return {
        "return": float(metrics.return_sum),
        "shaping": float(metrics.shaping_sum),
        "terminal": float(metrics.terminal_sum),
        "steps": int(metrics.steps),
        "winner": winner,
        "winner_type": winner_type,
        "win": 1 if winner == own_team else 0,
        "loss": 1 if winner not in {own_team, "DRAW", None} else 0,
        "draw": 1 if winner in {"DRAW", None} else 0,
        "normal_win": 1 if winner == own_team and winner_type == "normal" else 0,
        "death_time_win": 1 if winner == own_team and winner_type == "death_time" else 0,
    }


def _collect_rollout_worker(request: dict) -> dict:
    torch.set_num_threads(int(request["torch_threads"]))
    seed = int(request["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = PPOConfig(**request["config"])
    policy = MaskedActorCritic(hidden_size=config.hidden_size)
    policy.load_state_dict(request["policy_state"])
    policy.eval()

    env = TrainingEnv(_env_config(config, seed))
    obs, info = env.reset(seed=seed)

    observations = []
    masks = []
    actions = []
    logprobs = []
    values = []
    rewards = []
    dones = []
    entropies = []
    episodes = []
    episode = EpisodeMetrics()

    for step_index in range(config.rollout_steps):
        mask = env.action_mask()
        step = policy.act(obs, mask, deterministic=False)
        next_obs, reward, terminal, truncated, info = env.step(step.action)
        done = bool(terminal or truncated)
        reward_components = info.get("reward_components", {}) or {}

        observations.append(obs)
        masks.append(mask)
        actions.append(step.action)
        logprobs.append(step.logprob)
        values.append(step.value)
        rewards.append(float(reward))
        dones.append(done)
        entropies.append(step.entropy)

        episode.return_sum += float(reward)
        episode.shaping_sum += float(reward_components.get("shaping", 0.0) or 0.0)
        episode.terminal_sum += float(reward_components.get("terminal", 0.0) or 0.0)
        episode.steps += 1

        obs = next_obs
        if done:
            episode.winner = info.get("winner") or "DRAW"
            episode.winner_type = info.get("winner_type") or "unknown"
            episodes.append(_episode_summary(episode, config.controlled_player))
            episode = EpisodeMetrics()
            reset_seed = seed + 10_000 + step_index
            obs, info = env.reset(seed=reset_seed)

    last_mask = env.action_mask()
    last_value = policy.act(obs, last_mask, deterministic=True).value

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "masks": np.asarray(masks, dtype=np.bool_),
        "actions": np.asarray(actions, dtype=np.int64),
        "logprobs": np.asarray(logprobs, dtype=np.float32),
        "values": np.asarray(values, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "entropies": np.asarray(entropies, dtype=np.float32),
        "last_value": float(last_value),
        "episodes": episodes,
    }


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


def _merge_rollouts(worker_results: list[dict], config: PPOConfig) -> dict:
    advantages = []
    returns = []
    episodes = []
    for result in worker_results:
        worker_advantages, worker_returns = _compute_gae(
            result["rewards"],
            result["dones"],
            result["values"],
            result["last_value"],
            config.gamma,
            config.gae_lambda,
        )
        advantages.append(worker_advantages)
        returns.append(worker_returns)
        episodes.extend(result["episodes"])

    return {
        "observations": np.concatenate([item["observations"] for item in worker_results]),
        "masks": np.concatenate([item["masks"] for item in worker_results]),
        "actions": np.concatenate([item["actions"] for item in worker_results]),
        "old_logprobs": np.concatenate([item["logprobs"] for item in worker_results]),
        "values": np.concatenate([item["values"] for item in worker_results]),
        "rewards": np.concatenate([item["rewards"] for item in worker_results]),
        "dones": np.concatenate([item["dones"] for item in worker_results]),
        "entropies": np.concatenate([item["entropies"] for item in worker_results]),
        "advantages": np.concatenate(advantages),
        "returns": np.concatenate(returns),
        "episodes": episodes,
    }


def collect_rollouts(
    policy: MaskedActorCritic,
    config: PPOConfig,
    update: int,
    pool: Optional[mp.pool.Pool] = None,
) -> dict:
    policy_state = to_cpu_state_dict(policy)
    requests = [
        {
            "seed": config.seed + update * 1_000_000 + worker_id * 10_000,
            "config": asdict(config),
            "policy_state": policy_state,
            "torch_threads": config.torch_threads,
        }
        for worker_id in range(config.workers)
    ]
    if config.workers == 1:
        results = [_collect_rollout_worker(requests[0])]
    else:
        assert pool is not None
        results = pool.map(_collect_rollout_worker, requests)
    return _merge_rollouts(results, config)


def update_policy(
    policy: MaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: dict,
    config: PPOConfig,
    device: torch.device,
) -> dict:
    observations = torch.as_tensor(rollout["observations"], dtype=torch.float32, device=device)
    masks = torch.as_tensor(rollout["masks"], dtype=torch.bool, device=device)
    actions = torch.as_tensor(rollout["actions"], dtype=torch.long, device=device)
    old_logprobs = torch.as_tensor(rollout["old_logprobs"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=device)

    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)
    batch_size = observations.shape[0]
    minibatch_size = min(config.minibatch_size, batch_size)
    indexes = np.arange(batch_size)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "batches": 0}

    policy.train()
    for _ in range(config.ppo_epochs):
        np.random.shuffle(indexes)
        for start in range(0, batch_size, minibatch_size):
            batch_indexes = indexes[start:start + minibatch_size]
            dist, values = policy.distribution(observations[batch_indexes], masks[batch_indexes])
            new_logprobs = dist.log_prob(actions[batch_indexes])
            entropy = dist.entropy().mean()

            logratio = new_logprobs - old_logprobs[batch_indexes]
            ratio = logratio.exp()
            clipped_advantages = torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef) * advantages[batch_indexes]
            policy_loss = -torch.min(ratio * advantages[batch_indexes], clipped_advantages).mean()
            value_loss = 0.5 * ((returns[batch_indexes] - values) ** 2).mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - logratio).mean().item()
            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["entropy"] += float(entropy.item())
            stats["approx_kl"] += float(approx_kl)
            stats["batches"] += 1

    count = max(1, stats.pop("batches"))
    return {key: round(value / count, 6) for key, value in stats.items()}


def _rate(items: list[dict], key: str) -> float:
    if not items:
        return 0.0
    return float(sum(int(item.get(key, 0) or 0) for item in items) / len(items))


def _mean(items: list[dict], key: str) -> float:
    if not items:
        return 0.0
    return float(sum(float(item.get(key, 0.0) or 0.0) for item in items) / len(items))


def action_label(action_index: int) -> str:
    if action_index < 0 or action_index >= ACTION_DIM:
        return f"{action_index}:INVALID"
    spec = ACTION_SPECS[int(action_index)]
    if spec.kind == "HOLD":
        return f"{action_index}:HOLD"
    if spec.kind == "PRODUCE" and spec.unit_type is not None:
        return f"{action_index}:PRODUCE:{spec.unit_type.value}"
    if spec.kind == "MOVE" and spec.source and spec.destination:
        return f"{action_index}:MOVE:{spec.intent}:{spec.source}->{spec.destination}"
    return f"{action_index}:{spec.kind}"


def action_kind_label(action_index: int) -> str:
    if action_index < 0 or action_index >= ACTION_DIM:
        return "INVALID"
    spec = ACTION_SPECS[int(action_index)]
    if spec.kind == "MOVE":
        return f"MOVE_{spec.intent}"
    return spec.kind


def _invalid_action_count(actions: np.ndarray, masks: Optional[list[np.ndarray] | np.ndarray]) -> int:
    if masks is None:
        return 0
    mask_array = np.asarray(masks, dtype=np.bool_)
    if mask_array.shape[0] != actions.shape[0]:
        raise ValueError("Action/mask count mismatch while summarizing actions")

    invalid = 0
    for index, action in enumerate(actions):
        action_index = int(action)
        if action_index < 0 or action_index >= ACTION_DIM or not bool(mask_array[index, action_index]):
            invalid += 1
    return invalid


def summarize_action_distribution(
    actions: list[int] | np.ndarray,
    masks: Optional[list[np.ndarray] | np.ndarray] = None,
    top_n: int = 8,
) -> dict:
    action_array = np.asarray(actions, dtype=np.int64)
    total = int(action_array.size)
    if total <= 0:
        return {
            "total_actions": 0,
            "unique_actions": 0,
            "invalid_actions": 0,
            "invalid_action_rate": 0.0,
            "dominant_action": None,
            "dominant_action_rate": 0.0,
            "kind_counts": {},
            "top_actions": [],
        }

    counts = np.bincount(action_array, minlength=ACTION_DIM)
    nonzero = np.flatnonzero(counts)
    kind_counts: dict[str, int] = {}
    for action_index in action_array:
        key = action_kind_label(int(action_index))
        kind_counts[key] = kind_counts.get(key, 0) + 1

    top_indexes = sorted(nonzero, key=lambda idx: (-int(counts[idx]), int(idx)))[:top_n]
    top_actions = [
        {
            "action": action_label(int(index)),
            "count": int(counts[index]),
            "rate": round(float(counts[index]) / total, 6),
        }
        for index in top_indexes
    ]
    dominant = top_actions[0] if top_actions else None
    invalid_actions = _invalid_action_count(action_array, masks)

    return {
        "total_actions": total,
        "unique_actions": int(nonzero.size),
        "invalid_actions": invalid_actions,
        "invalid_action_rate": round(float(invalid_actions) / total, 6),
        "dominant_action": dominant["action"] if dominant else None,
        "dominant_action_rate": dominant["rate"] if dominant else 0.0,
        "kind_counts": {
            key: {
                "count": value,
                "rate": round(float(value) / total, 6),
            }
            for key, value in sorted(kind_counts.items())
        },
        "top_actions": top_actions,
    }


def summarize_training_episodes(
    episodes: list[dict],
    actions: Optional[list[int] | np.ndarray] = None,
    masks: Optional[list[np.ndarray] | np.ndarray] = None,
) -> dict:
    summary = {
        "episodes": len(episodes),
        "mean_reward": round(_mean(episodes, "return"), 6),
        "mean_shaping": round(_mean(episodes, "shaping"), 6),
        "mean_terminal": round(_mean(episodes, "terminal"), 6),
        "win_rate": round(_rate(episodes, "win"), 6),
        "loss_rate": round(_rate(episodes, "loss"), 6),
        "draw_rate": round(_rate(episodes, "draw"), 6),
        "normal_win_rate": round(_rate(episodes, "normal_win"), 6),
        "death_time_win_rate": round(_rate(episodes, "death_time_win"), 6),
        "mean_episode_steps": round(_mean(episodes, "steps"), 3),
    }
    if actions is not None:
        summary["action_distribution"] = summarize_action_distribution(actions, masks)
    return summary


@torch.no_grad()
def evaluate_policy(
    policy: MaskedActorCritic,
    config: PPOConfig,
    episodes: int,
    seed: int,
    deterministic: bool,
    device: torch.device,
) -> dict:
    policy.eval()
    summaries = []
    actions = []
    masks = []
    for episode_index in range(episodes):
        env_seed = seed + episode_index
        env = TrainingEnv(_env_config(config, env_seed))
        obs, info = env.reset(seed=env_seed)
        metrics = EpisodeMetrics()
        done = False
        while not done:
            mask = env.action_mask()
            step = policy.act(obs, mask, deterministic=deterministic, device=device)
            masks.append(mask)
            actions.append(step.action)
            obs, reward, terminal, truncated, info = env.step(step.action)
            reward_components = info.get("reward_components", {}) or {}
            metrics.return_sum += float(reward)
            metrics.shaping_sum += float(reward_components.get("shaping", 0.0) or 0.0)
            metrics.terminal_sum += float(reward_components.get("terminal", 0.0) or 0.0)
            metrics.steps += 1
            done = bool(terminal or truncated)
        metrics.winner = info.get("winner") or "DRAW"
        metrics.winner_type = info.get("winner_type") or "unknown"
        summaries.append(_episode_summary(metrics, config.controlled_player))

    summary = summarize_training_episodes(summaries, actions, masks)
    summary["mode"] = "argmax" if deterministic else "stochastic"
    summary["episodes_detail"] = summaries
    return summary


def evaluate_random_legal_policy(config: PPOConfig, episodes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    summaries = []
    actions = []
    masks = []
    for episode_index in range(episodes):
        env_seed = seed + episode_index
        env = TrainingEnv(_env_config(config, env_seed))
        obs, info = env.reset(seed=env_seed)
        metrics = EpisodeMetrics()
        done = False
        while not done:
            mask = env.action_mask()
            legal = np.flatnonzero(mask)
            action = int(rng.choice(legal)) if len(legal) else 0
            masks.append(mask)
            actions.append(action)
            obs, reward, terminal, truncated, info = env.step(action)
            reward_components = info.get("reward_components", {}) or {}
            metrics.return_sum += float(reward)
            metrics.shaping_sum += float(reward_components.get("shaping", 0.0) or 0.0)
            metrics.terminal_sum += float(reward_components.get("terminal", 0.0) or 0.0)
            metrics.steps += 1
            done = bool(terminal or truncated)
        metrics.winner = info.get("winner") or "DRAW"
        metrics.winner_type = info.get("winner_type") or "unknown"
        summaries.append(_episode_summary(metrics, config.controlled_player))

    summary = summarize_training_episodes(summaries, actions, masks)
    summary["mode"] = "random_legal"
    summary["episodes_detail"] = summaries
    return summary


def save_checkpoint(
    path: Path,
    policy: MaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    update: int,
    total_steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state": policy.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "observation_version": OBSERVATION_VERSION,
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "update": int(update),
            "total_steps": int(total_steps),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    policy: MaskedActorCritic,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    _validate_checkpoint_interface(checkpoint, path)
    policy.load_state_dict(checkpoint["policy_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def _validate_checkpoint_interface(checkpoint: dict, path: Path) -> None:
    if (
        checkpoint.get("observation_version") != OBSERVATION_VERSION
        or int(checkpoint.get("observation_dim") or -1) != OBSERVATION_DIM
        or int(checkpoint.get("action_dim") or -1) != ACTION_DIM
    ):
        raise ValueError(
            f"Checkpoint {path} is incompatible with current policy interface: "
            f"checkpoint=({checkpoint.get('observation_version')}, "
            f"{checkpoint.get('observation_dim')}, {checkpoint.get('action_dim')}), "
            f"current=({OBSERVATION_VERSION}, {OBSERVATION_DIM}, {ACTION_DIM})"
        )


def append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def latest_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip().lstrip("\ufeff")
            if line:
                result.append(json.loads(line))
    return result
