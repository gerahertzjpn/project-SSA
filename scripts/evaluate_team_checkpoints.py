import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.runner import run_match
from src.training.team_env import TeamTrainingEnv
from src.training.team_ppo_trainer import (
    TeamPPOConfig,
    _env_config,
    _team_episode_summary,
    make_team_policies,
)
from src.training.ppo_trainer import summarize_action_distribution, summarize_training_episodes
from src.training.spaces import sample_legal_action as sample_legal_factorized_action, team_of


def parse_csv_ints(value: str) -> list[int]:
    result = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    if not result:
        raise ValueError("Expected at least one integer")
    return result


def parse_csv_strings(value: str) -> list[str]:
    result = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            result.append(item)
    if not result:
        raise ValueError("Expected at least one value")
    return result


def parse_players(value: str) -> tuple[str, ...]:
    players = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not players:
        raise argparse.ArgumentTypeError("Expected at least one player id")
    return players


def checkpoint_paths(run_dir: Path, updates: list[str]) -> list[Path]:
    paths = []
    for update in updates:
        if update == "latest":
            paths.append(run_dir / "checkpoints" / "latest.pt")
        else:
            paths.append(run_dir / "checkpoints" / f"checkpoint_update_{int(update)}.pt")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {', '.join(missing)}")
    return paths


def episode_specs(seed_bases: list[int], episodes_per_seed: int) -> list[dict]:
    specs = []
    for seed_base in seed_bases:
        for episode_index in range(episodes_per_seed):
            specs.append({
                "seed_block": int(seed_base),
                "env_seed": int(seed_base + episode_index),
            })
    return specs


def chunks(items: list[dict], chunk_count: int) -> list[list[dict]]:
    chunk_count = max(1, min(chunk_count, len(items)))
    return [items[index::chunk_count] for index in range(chunk_count)]


def _action_payload(action) -> list[int]:
    action_array = np.asarray(action, dtype=np.int64).reshape(-1)
    return [int(value) for value in action_array]


def sequence_hash(actions: list) -> str:
    payload = json.dumps(
        [_action_payload(action) for action in actions],
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha1(payload).hexdigest()[:12]


def load_checkpoint_payload(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {}) or {}
    return {
        "path": str(path),
        "update": int(checkpoint.get("update", -1)),
        "total_steps": int(checkpoint.get("total_steps", 0) or 0),
        "hidden_size": int(config.get("hidden_size", 128)),
        "policy_states": checkpoint["policy_states"],
    }


def team_policy_eval_worker(request: dict) -> dict:
    torch.set_num_threads(int(request["torch_threads"]))
    np.random.seed(int(request["worker_seed"]))
    torch.manual_seed(int(request["worker_seed"]))

    config = TeamPPOConfig(**request["config"])
    policies = make_team_policies(config)
    for player_id, state in request["policy_states"].items():
        policies[player_id].load_state_dict(state)
        policies[player_id].eval()

    episodes = []
    actions_by_player = {player_id: [] for player_id in config.controlled_players}
    masks_by_player = {player_id: [] for player_id in config.controlled_players}
    entropies_by_player = {player_id: [] for player_id in config.controlled_players}
    sequence_hashes = []
    opening_hashes = []

    for spec in request["episodes"]:
        env_seed = int(spec["env_seed"])
        torch.manual_seed(int(request["sample_seed_offset"]) + env_seed)
        np.random.seed((int(request["sample_seed_offset"]) + env_seed) % (2**32 - 1))

        env = TeamTrainingEnv(_env_config(config, env_seed))
        observations, _ = env.reset(seed=env_seed)
        return_sums = {player_id: 0.0 for player_id in config.controlled_players}
        shaping_sums = {player_id: 0.0 for player_id in config.controlled_players}
        terminal_sums = {player_id: 0.0 for player_id in config.controlled_players}
        episode_actions = []
        steps = 0
        done = False

        while not done:
            masks = env.action_masks()
            selected_actions = {}
            for player_id in config.controlled_players:
                step = policies[player_id].act(
                    observations[player_id],
                    masks[player_id],
                    deterministic=bool(request["deterministic"]),
                    device="cpu",
                )
                selected_actions[player_id] = step.action
                actions_by_player[player_id].append(step.action)
                masks_by_player[player_id].append(masks[player_id])
                entropies_by_player[player_id].append(float(step.entropy))
                episode_actions.append(step.action)

            result = env.step(selected_actions)
            for player_id in config.controlled_players:
                reward_components = result.info.get("reward_components", {}).get(player_id, {}) or {}
                return_sums[player_id] += float(result.rewards[player_id])
                shaping_sums[player_id] += float(reward_components.get("shaping", 0.0) or 0.0)
                terminal_sums[player_id] += float(reward_components.get("terminal", 0.0) or 0.0)
            observations = result.observations
            steps += 1
            done = bool(result.terminal or result.truncated)

        ep = _team_episode_summary(
            return_sums=return_sums,
            shaping_sums=shaping_sums,
            terminal_sums=terminal_sums,
            steps=steps,
            winner=result.info.get("winner") or "DRAW",
            winner_type=result.info.get("winner_type") or "unknown",
            controlled_team=env.controlled_team,
        )
        ep["seed_block"] = int(spec["seed_block"])
        ep["env_seed"] = env_seed
        ep["sequence_hash"] = sequence_hash(episode_actions)
        ep["opening_hash"] = sequence_hash(episode_actions[:60])
        episodes.append(ep)
        sequence_hashes.append(ep["sequence_hash"])
        opening_hashes.append(ep["opening_hash"])

    return {
        "episodes": episodes,
        "actions_by_player": actions_by_player,
        "masks_by_player": masks_by_player,
        "entropies_by_player": entropies_by_player,
        "sequence_hashes": sequence_hashes,
        "opening_hashes": opening_hashes,
    }


def random_legal_worker(request: dict) -> dict:
    np.random.seed(int(request["worker_seed"]))
    config = TeamPPOConfig(**request["config"])
    rng = np.random.default_rng(int(request["worker_seed"]))
    episodes = []
    actions_by_player = {player_id: [] for player_id in config.controlled_players}
    masks_by_player = {player_id: [] for player_id in config.controlled_players}
    sequence_hashes = []
    opening_hashes = []

    for spec in request["episodes"]:
        env_seed = int(spec["env_seed"])
        env = TeamTrainingEnv(_env_config(config, env_seed))
        env.reset(seed=env_seed)
        return_sums = {player_id: 0.0 for player_id in config.controlled_players}
        shaping_sums = {player_id: 0.0 for player_id in config.controlled_players}
        terminal_sums = {player_id: 0.0 for player_id in config.controlled_players}
        episode_actions = []
        steps = 0
        done = False

        while not done:
            masks = env.action_masks()
            selected_actions = {}
            for player_id, mask in masks.items():
                action = sample_legal_factorized_action(mask, rng)
                selected_actions[player_id] = action
                actions_by_player[player_id].append(action)
                masks_by_player[player_id].append(mask)
                episode_actions.append(action)
            result = env.step(selected_actions)
            for player_id in config.controlled_players:
                reward_components = result.info.get("reward_components", {}).get(player_id, {}) or {}
                return_sums[player_id] += float(result.rewards[player_id])
                shaping_sums[player_id] += float(reward_components.get("shaping", 0.0) or 0.0)
                terminal_sums[player_id] += float(reward_components.get("terminal", 0.0) or 0.0)
            steps += 1
            done = bool(result.terminal or result.truncated)

        ep = _team_episode_summary(
            return_sums=return_sums,
            shaping_sums=shaping_sums,
            terminal_sums=terminal_sums,
            steps=steps,
            winner=result.info.get("winner") or "DRAW",
            winner_type=result.info.get("winner_type") or "unknown",
            controlled_team=env.controlled_team,
        )
        ep["seed_block"] = int(spec["seed_block"])
        ep["env_seed"] = env_seed
        ep["sequence_hash"] = sequence_hash(episode_actions)
        ep["opening_hash"] = sequence_hash(episode_actions[:60])
        episodes.append(ep)
        sequence_hashes.append(ep["sequence_hash"])
        opening_hashes.append(ep["opening_hash"])

    return {
        "episodes": episodes,
        "actions_by_player": actions_by_player,
        "masks_by_player": masks_by_player,
        "sequence_hashes": sequence_hashes,
        "opening_hashes": opening_hashes,
    }


def scripted_control_worker(request: dict) -> dict:
    controlled_team = team_of(request["controlled_players"][0])
    episodes = []
    for spec in request["episodes"]:
        result = run_match(seed=int(spec["env_seed"]), bot_type="scripted", mode="training")
        ep = _team_episode_summary(
            return_sums={player_id: 0.0 for player_id in request["controlled_players"]},
            shaping_sums={player_id: 0.0 for player_id in request["controlled_players"]},
            terminal_sums={player_id: 0.0 for player_id in request["controlled_players"]},
            steps=result.total_ticks,
            winner=result.winner,
            winner_type=result.winner_type,
            controlled_team=controlled_team,
        )
        ep["seed_block"] = int(spec["seed_block"])
        ep["env_seed"] = int(spec["env_seed"])
        episodes.append(ep)
    return {"episodes": episodes}


def pool_map(worker, requests: list[dict], workers: int) -> list[dict]:
    if workers <= 1 or len(requests) <= 1:
        return [worker(request) for request in requests]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(workers, len(requests))) as pool:
        return pool.map(worker, requests)


def add_counts(summary: dict, episodes: list[dict]) -> None:
    winners = {}
    winner_types = {}
    for item in episodes:
        winners[item["winner"]] = winners.get(item["winner"], 0) + 1
        winner_types[item["winner_type"]] = winner_types.get(item["winner_type"], 0) + 1
    summary["winner_counts"] = dict(sorted(winners.items()))
    summary["winner_type_counts"] = dict(sorted(winner_types.items()))


def seed_block_summaries(episodes: list[dict]) -> dict:
    result = {}
    for seed_block in sorted({int(item["seed_block"]) for item in episodes}):
        block_items = [item for item in episodes if int(item["seed_block"]) == seed_block]
        summary = summarize_training_episodes(block_items)
        add_counts(summary, block_items)
        result[str(seed_block)] = summary
    return result


def aggregate_eval(results: list[dict], mode: str, controlled_players: tuple[str, ...]) -> dict:
    episodes = []
    actions_by_player = {player_id: [] for player_id in controlled_players}
    masks_by_player = {player_id: [] for player_id in controlled_players}
    entropies_by_player = {player_id: [] for player_id in controlled_players}
    sequence_hashes = []
    opening_hashes = []
    for result in results:
        episodes.extend(result.get("episodes", []))
        sequence_hashes.extend(result.get("sequence_hashes", []))
        opening_hashes.extend(result.get("opening_hashes", []))
        for player_id in controlled_players:
            actions_by_player[player_id].extend(result.get("actions_by_player", {}).get(player_id, []))
            masks_by_player[player_id].extend(result.get("masks_by_player", {}).get(player_id, []))
            entropies_by_player[player_id].extend(result.get("entropies_by_player", {}).get(player_id, []))

    summary = summarize_training_episodes(episodes)
    add_counts(summary, episodes)
    summary["mode"] = mode
    summary["seed_blocks"] = seed_block_summaries(episodes)
    summary["agents"] = {}
    for player_id in controlled_players:
        agent_summary = {
            "action_distribution": summarize_action_distribution(
                actions_by_player[player_id],
                masks_by_player[player_id] if masks_by_player[player_id] else None,
            ),
        }
        if entropies_by_player[player_id]:
            agent_summary["mean_policy_entropy"] = round(float(np.mean(entropies_by_player[player_id])), 6)
        summary["agents"][player_id] = agent_summary
    if sequence_hashes:
        summary["unique_action_sequences"] = len(set(sequence_hashes))
        summary["sequence_diversity_rate"] = round(len(set(sequence_hashes)) / max(1, len(sequence_hashes)), 6)
    if opening_hashes:
        summary["unique_opening_sequences"] = len(set(opening_hashes))
        summary["opening_diversity_rate"] = round(len(set(opening_hashes)) / max(1, len(opening_hashes)), 6)
    return summary


def print_record(record: dict) -> None:
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)


def write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate synchronized team PPO checkpoints")
    parser.add_argument("--run-dir", default="training_runs/sprint3_team_scripted_warm_v1")
    parser.add_argument("--updates", default="latest")
    parser.add_argument("--modes", default="argmax,stochastic")
    parser.add_argument("--seed-bases", default="20260603,20270603,20280603,20290603")
    parser.add_argument("--episodes-per-seed", type=int, default=50)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--controlled-players", type=parse_players, default=("A1", "A2", "A3"))
    scripted_choices = ["scripted", "scripted_core", "scripted_core_raider", "scripted_core_bmp"]
    parser.add_argument("--opponent-bot", choices=scripted_choices + ["random", "policy"], default="scripted")
    parser.add_argument("--ally-bot", choices=scripted_choices + ["random"], default="scripted")
    parser.add_argument("--opponent-policy-checkpoint", default=None, help="Frozen team checkpoint for opponent_bot=policy")
    parser.add_argument(
        "--opponent-policy-mode",
        choices=["stochastic", "argmax"],
        default="stochastic",
        help="How the frozen policy opponent chooses actions",
    )
    parser.add_argument("--max-ticks", type=int, default=300)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--include-random-legal", action="store_true")
    parser.add_argument("--include-scripted-control", action="store_true")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.opponent_bot == "policy" and not args.opponent_policy_checkpoint:
        raise ValueError("--opponent-policy-checkpoint is required with --opponent-bot policy")
    if args.opponent_bot != "policy" and args.opponent_policy_checkpoint:
        raise ValueError("--opponent-policy-checkpoint requires --opponent-bot policy")
    started = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"training_runs/evals/team_eval_{started}")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    summary_path = out_dir / "summary.json"

    update_items = parse_csv_strings(args.updates)
    modes = parse_csv_strings(args.modes)
    seed_bases = parse_csv_ints(args.seed_bases)
    controlled_players = tuple(args.controlled_players)
    specs = episode_specs(seed_bases, int(args.episodes_per_seed))
    specs_chunks = chunks(specs, int(args.workers))
    paths = checkpoint_paths(Path(args.run_dir), update_items)
    config = TeamPPOConfig(
        controlled_players=controlled_players,
        opponent_bot=args.opponent_bot,
        ally_bot=args.ally_bot,
        opponent_policy_checkpoint=args.opponent_policy_checkpoint,
        opponent_policy_deterministic=args.opponent_policy_mode == "argmax",
        max_ticks=int(args.max_ticks),
        hidden_size=int(args.hidden_size),
        workers=int(args.workers),
        torch_threads=int(args.torch_threads),
    )

    config_record = {
        "kind": "team_eval_started",
        "out_dir": str(out_dir),
        "run_dir": args.run_dir,
        "updates": update_items,
        "modes": modes,
        "seed_bases": seed_bases,
        "episodes_per_seed": int(args.episodes_per_seed),
        "total_episodes_per_eval": len(specs),
        "workers": int(args.workers),
        "controlled_players": list(controlled_players),
        "ally_bot": args.ally_bot,
        "opponent_bot": args.opponent_bot,
        "opponent_policy_checkpoint": args.opponent_policy_checkpoint,
        "opponent_policy_mode": args.opponent_policy_mode,
        "max_ticks": int(args.max_ticks),
    }
    print_record(config_record)
    write_jsonl(metrics_path, config_record)

    records = []

    if args.include_scripted_control:
        requests = [{"episodes": chunk, "controlled_players": controlled_players} for chunk in specs_chunks]
        started_at = time.perf_counter()
        result = aggregate_eval(
            pool_map(scripted_control_worker, requests, int(args.workers)),
            mode="scripted_control_team",
            controlled_players=controlled_players,
        )
        record = {
            "kind": "scripted_control_team_eval",
            "elapsed_sec": round(time.perf_counter() - started_at, 3),
            "summary": result,
        }
        records.append(record)
        print_record(record)
        write_jsonl(metrics_path, record)

    if args.include_random_legal:
        requests = []
        for worker_index, chunk in enumerate(specs_chunks):
            requests.append({
                "episodes": chunk,
                "worker_seed": seed_bases[0] + worker_index * 9973,
                "config": config.__dict__,
            })
        started_at = time.perf_counter()
        result = aggregate_eval(
            pool_map(random_legal_worker, requests, int(args.workers)),
            mode="random_legal_team",
            controlled_players=controlled_players,
        )
        record = {
            "kind": "random_legal_team_eval",
            "elapsed_sec": round(time.perf_counter() - started_at, 3),
            "summary": result,
        }
        records.append(record)
        print_record(record)
        write_jsonl(metrics_path, record)

    for path in paths:
        payload = load_checkpoint_payload(path)
        for mode in modes:
            deterministic = mode == "argmax"
            if mode not in {"argmax", "stochastic"}:
                raise ValueError(f"Unknown eval mode: {mode}")
            requests = []
            for worker_index, chunk in enumerate(specs_chunks):
                requests.append({
                    "episodes": chunk,
                    "worker_seed": seed_bases[0] + worker_index * 10_007 + payload["update"] * 101,
                    "sample_seed_offset": seed_bases[0] + payload["update"] * 1_000_003,
                    "policy_states": payload["policy_states"],
                    "deterministic": deterministic,
                    "config": {
                        **config.__dict__,
                        "hidden_size": int(payload["hidden_size"]),
                    },
                    "torch_threads": int(args.torch_threads),
                })
            started_at = time.perf_counter()
            result = aggregate_eval(
                pool_map(team_policy_eval_worker, requests, int(args.workers)),
                mode=mode,
                controlled_players=controlled_players,
            )
            record = {
                "kind": "team_checkpoint_eval",
                "checkpoint": str(path),
                "update": payload["update"],
                "total_steps": payload["total_steps"],
                "mode": mode,
                "deterministic": deterministic,
                "elapsed_sec": round(time.perf_counter() - started_at, 3),
                "summary": result,
            }
            records.append(record)
            print_record(record)
            write_jsonl(metrics_path, record)

    final = {
        "config": config_record,
        "records": records,
        "metrics_path": str(metrics_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(final, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print_record({
        "kind": "team_eval_finished",
        "summary_path": str(summary_path),
        "metrics_path": str(metrics_path),
        "records": len(records),
    })


if __name__ == "__main__":
    main()
