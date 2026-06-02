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
from src.training.env import TrainingEnv, TrainingEnvConfig
from src.training.policy import MaskedActorCritic
from src.training.ppo_trainer import (
    summarize_action_distribution,
    summarize_training_episodes,
)
from src.training.spaces import team_of


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


def checkpoint_paths(run_dir: Path, updates: list[int]) -> list[Path]:
    paths = [run_dir / "checkpoints" / f"checkpoint_update_{update}.pt" for update in updates]
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


def episode_result(return_sum: float, shaping_sum: float, terminal_sum: float, steps: int, winner: str, winner_type: str, controlled_player: str) -> dict:
    own_team = team_of(controlled_player)
    return {
        "return": float(return_sum),
        "shaping": float(shaping_sum),
        "terminal": float(terminal_sum),
        "steps": int(steps),
        "winner": winner,
        "winner_type": winner_type,
        "win": 1 if winner == own_team else 0,
        "loss": 1 if winner not in {own_team, "DRAW", None} else 0,
        "draw": 1 if winner in {"DRAW", None} else 0,
        "normal_win": 1 if winner == own_team and winner_type == "normal" else 0,
        "death_time_win": 1 if winner == own_team and winner_type == "death_time" else 0,
    }


def sequence_hash(actions: list[int]) -> str:
    payload = ",".join(str(int(action)) for action in actions).encode("ascii")
    return hashlib.sha1(payload).hexdigest()[:12]


def policy_eval_worker(request: dict) -> dict:
    torch.set_num_threads(int(request["torch_threads"]))
    np.random.seed(int(request["worker_seed"]))
    torch.manual_seed(int(request["worker_seed"]))

    policy = MaskedActorCritic(hidden_size=int(request["hidden_size"]))
    policy.load_state_dict(request["policy_state"])
    policy.eval()

    episodes = []
    actions = []
    entropies = []
    invalid_actions = 0
    sequence_hashes = []
    opening_hashes = []
    controlled_player = request["controlled_player"]

    for spec in request["episodes"]:
        env_seed = int(spec["env_seed"])
        torch.manual_seed(int(request["sample_seed_offset"]) + env_seed)
        np.random.seed((int(request["sample_seed_offset"]) + env_seed) % (2**32 - 1))

        env = TrainingEnv(TrainingEnvConfig(
            controlled_player=controlled_player,
            seed=env_seed,
            opponent_bot=request["opponent_bot"],
            ally_bot=request["ally_bot"],
            max_ticks=int(request["max_ticks"]),
            record_replay=False,
            record_trace=False,
        ))
        obs, info = env.reset(seed=env_seed)

        return_sum = 0.0
        shaping_sum = 0.0
        terminal_sum = 0.0
        steps = 0
        done = False
        episode_actions = []

        while not done:
            mask = env.action_mask()
            step = policy.act(
                obs,
                mask,
                deterministic=bool(request["deterministic"]),
                device="cpu",
            )
            action = int(step.action)
            if action < 0 or action >= len(mask) or not bool(mask[action]):
                invalid_actions += 1
            actions.append(action)
            entropies.append(float(step.entropy))
            episode_actions.append(action)

            obs, reward, terminal, truncated, info = env.step(action)
            reward_components = info.get("reward_components", {}) or {}
            return_sum += float(reward)
            shaping_sum += float(reward_components.get("shaping", 0.0) or 0.0)
            terminal_sum += float(reward_components.get("terminal", 0.0) or 0.0)
            steps += 1
            done = bool(terminal or truncated)

        ep = episode_result(
            return_sum=return_sum,
            shaping_sum=shaping_sum,
            terminal_sum=terminal_sum,
            steps=steps,
            winner=info.get("winner") or "DRAW",
            winner_type=info.get("winner_type") or "unknown",
            controlled_player=controlled_player,
        )
        ep["seed_block"] = int(spec["seed_block"])
        ep["env_seed"] = env_seed
        ep["sequence_hash"] = sequence_hash(episode_actions)
        ep["opening_hash"] = sequence_hash(episode_actions[:20])
        episodes.append(ep)
        sequence_hashes.append(ep["sequence_hash"])
        opening_hashes.append(ep["opening_hash"])

    return {
        "episodes": episodes,
        "actions": actions,
        "entropies": entropies,
        "invalid_actions": invalid_actions,
        "sequence_hashes": sequence_hashes,
        "opening_hashes": opening_hashes,
    }


def random_legal_worker(request: dict) -> dict:
    np.random.seed(int(request["worker_seed"]))
    episodes = []
    actions = []
    sequence_hashes = []
    opening_hashes = []
    controlled_player = request["controlled_player"]

    for spec in request["episodes"]:
        env_seed = int(spec["env_seed"])
        env = TrainingEnv(TrainingEnvConfig(
            controlled_player=controlled_player,
            seed=env_seed,
            opponent_bot=request["opponent_bot"],
            ally_bot=request["ally_bot"],
            max_ticks=int(request["max_ticks"]),
            record_replay=False,
            record_trace=False,
        ))
        obs, info = env.reset(seed=env_seed)

        return_sum = 0.0
        shaping_sum = 0.0
        terminal_sum = 0.0
        steps = 0
        done = False
        episode_actions = []

        while not done:
            action = env.sample_legal_action()
            actions.append(action)
            episode_actions.append(action)
            obs, reward, terminal, truncated, info = env.step(action)
            reward_components = info.get("reward_components", {}) or {}
            return_sum += float(reward)
            shaping_sum += float(reward_components.get("shaping", 0.0) or 0.0)
            terminal_sum += float(reward_components.get("terminal", 0.0) or 0.0)
            steps += 1
            done = bool(terminal or truncated)

        ep = episode_result(
            return_sum=return_sum,
            shaping_sum=shaping_sum,
            terminal_sum=terminal_sum,
            steps=steps,
            winner=info.get("winner") or "DRAW",
            winner_type=info.get("winner_type") or "unknown",
            controlled_player=controlled_player,
        )
        ep["seed_block"] = int(spec["seed_block"])
        ep["env_seed"] = env_seed
        ep["sequence_hash"] = sequence_hash(episode_actions)
        ep["opening_hash"] = sequence_hash(episode_actions[:20])
        episodes.append(ep)
        sequence_hashes.append(ep["sequence_hash"])
        opening_hashes.append(ep["opening_hash"])

    return {
        "episodes": episodes,
        "actions": actions,
        "invalid_actions": 0,
        "sequence_hashes": sequence_hashes,
        "opening_hashes": opening_hashes,
    }


def scripted_control_worker(request: dict) -> dict:
    episodes = []
    controlled_player = request["controlled_player"]
    for spec in request["episodes"]:
        result = run_match(seed=int(spec["env_seed"]), bot_type="scripted", mode="training")
        ep = episode_result(
            return_sum=0.0,
            shaping_sum=0.0,
            terminal_sum=0.0,
            steps=result.total_ticks,
            winner=result.winner,
            winner_type=result.winner_type,
            controlled_player=controlled_player,
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


def aggregate_eval(results: list[dict], mode: str) -> dict:
    episodes = []
    actions = []
    entropies = []
    invalid_actions = 0
    sequence_hashes = []
    opening_hashes = []
    for result in results:
        episodes.extend(result.get("episodes", []))
        actions.extend(result.get("actions", []))
        entropies.extend(result.get("entropies", []))
        invalid_actions += int(result.get("invalid_actions", 0) or 0)
        sequence_hashes.extend(result.get("sequence_hashes", []))
        opening_hashes.extend(result.get("opening_hashes", []))

    summary = summarize_training_episodes(episodes, actions if actions else None)
    add_counts(summary, episodes)
    if actions:
        action_summary = summary.get("action_distribution") or summarize_action_distribution(actions)
        action_summary["invalid_actions"] = invalid_actions
        action_summary["invalid_action_rate"] = round(invalid_actions / max(1, len(actions)), 6)
        summary["action_distribution"] = action_summary
    summary["mode"] = mode
    summary["seed_blocks"] = seed_block_summaries(episodes)
    if entropies:
        summary["mean_policy_entropy"] = round(float(np.mean(entropies)), 6)
    if sequence_hashes:
        summary["unique_action_sequences"] = len(set(sequence_hashes))
        summary["sequence_diversity_rate"] = round(len(set(sequence_hashes)) / max(1, len(sequence_hashes)), 6)
    if opening_hashes:
        summary["unique_opening_sequences"] = len(set(opening_hashes))
        summary["opening_diversity_rate"] = round(len(set(opening_hashes)) / max(1, len(opening_hashes)), 6)
    return summary


def load_checkpoint_payload(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {}) or {}
    return {
        "path": str(path),
        "update": int(checkpoint.get("update", -1)),
        "total_steps": int(checkpoint.get("total_steps", 0) or 0),
        "hidden_size": int(config.get("hidden_size", 128)),
        "policy_state": checkpoint["policy_state"],
    }


def print_record(record: dict) -> None:
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)


def write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoints against baseline bots")
    parser.add_argument("--run-dir", default="training_runs/sprint2_random_first_real_v1")
    parser.add_argument("--updates", default="170,180,190,380,390,400")
    parser.add_argument("--modes", default="argmax,stochastic")
    parser.add_argument("--seed-bases", default="20260602,20270602,20280602,20290602")
    parser.add_argument("--episodes-per-seed", type=int, default=50)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--controlled-player", default="A1")
    parser.add_argument("--ally-bot", choices=["scripted", "random"], default="scripted")
    parser.add_argument("--opponent-bot", choices=["scripted", "random"], default="scripted")
    parser.add_argument("--max-ticks", type=int, default=300)
    parser.add_argument("--include-random-legal", action="store_true")
    parser.add_argument("--skip-scripted-control", action="store_true")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"training_runs/evals/scripted_eval_{started}")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    summary_path = out_dir / "summary.json"

    updates = parse_csv_ints(args.updates)
    modes = parse_csv_strings(args.modes)
    seed_bases = parse_csv_ints(args.seed_bases)
    specs = episode_specs(seed_bases, int(args.episodes_per_seed))
    specs_chunks = chunks(specs, int(args.workers))
    paths = checkpoint_paths(Path(args.run_dir), updates)

    config_record = {
        "kind": "eval_started",
        "out_dir": str(out_dir),
        "run_dir": args.run_dir,
        "updates": updates,
        "modes": modes,
        "seed_bases": seed_bases,
        "episodes_per_seed": int(args.episodes_per_seed),
        "total_episodes_per_eval": len(specs),
        "workers": int(args.workers),
        "controlled_player": args.controlled_player,
        "ally_bot": args.ally_bot,
        "opponent_bot": args.opponent_bot,
        "max_ticks": int(args.max_ticks),
    }
    print_record(config_record)
    write_jsonl(metrics_path, config_record)

    records = []

    if not args.skip_scripted_control:
        requests = [{"episodes": chunk, "controlled_player": args.controlled_player} for chunk in specs_chunks]
        started_at = time.perf_counter()
        result = aggregate_eval(
            pool_map(scripted_control_worker, requests, int(args.workers)),
            mode="scripted_control",
        )
        record = {
            "kind": "scripted_control_eval",
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
                "controlled_player": args.controlled_player,
                "opponent_bot": args.opponent_bot,
                "ally_bot": args.ally_bot,
                "max_ticks": int(args.max_ticks),
            })
        started_at = time.perf_counter()
        result = aggregate_eval(
            pool_map(random_legal_worker, requests, int(args.workers)),
            mode="random_legal",
        )
        record = {
            "kind": "random_legal_eval",
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
                    "policy_state": payload["policy_state"],
                    "hidden_size": payload["hidden_size"],
                    "deterministic": deterministic,
                    "controlled_player": args.controlled_player,
                    "opponent_bot": args.opponent_bot,
                    "ally_bot": args.ally_bot,
                    "max_ticks": int(args.max_ticks),
                    "torch_threads": int(args.torch_threads),
                })
            started_at = time.perf_counter()
            result = aggregate_eval(
                pool_map(policy_eval_worker, requests, int(args.workers)),
                mode=mode,
            )
            record = {
                "kind": "checkpoint_eval",
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
        "kind": "eval_finished",
        "summary_path": str(summary_path),
        "metrics_path": str(metrics_path),
        "records": len(records),
    })


if __name__ == "__main__":
    main()
