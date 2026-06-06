from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.constants import TICK_LIMIT
from src.training.spaces import ACTION_DIM, OBSERVATION_DIM, OBSERVATION_VERSION
from src.training.team_env import TeamTrainingEnv, TeamTrainingEnvConfig


def _run_worker(args: dict) -> dict:
    worker_id = int(args["worker_id"])
    duration_sec = float(args["duration_sec"])
    base_seed = int(args["seed"])
    max_ticks = int(args["max_ticks"])

    env = TeamTrainingEnv(TeamTrainingEnvConfig(
        seed=base_seed + worker_id,
        opponent_bot=args["opponent_bot"],
        ally_bot=args["ally_bot"],
        opponent_policy_checkpoint=args.get("opponent_policy_checkpoint"),
        opponent_policy_deterministic=bool(args.get("opponent_policy_deterministic", False)),
        max_ticks=max_ticks,
        record_replay=False,
        record_trace=False,
    ))

    deadline = time.perf_counter() + duration_sec
    steps = 0
    episodes = 0
    winners: dict[str, int] = {}
    observations, info = env.reset(seed=base_seed + worker_id)
    terminal = False
    truncated = False

    start = time.perf_counter()
    while time.perf_counter() < deadline:
        if terminal or truncated:
            winners[info["winner"]] = winners.get(info["winner"], 0) + 1
            episodes += 1
            episode_seed = base_seed + worker_id * 1_000_000 + episodes
            observations, info = env.reset(seed=episode_seed)
            terminal = False
            truncated = False

        result = env.step(env.sample_legal_actions())
        observations = result.observations
        info = result.info
        terminal = result.terminal
        truncated = result.truncated
        steps += 1

    elapsed = max(0.001, time.perf_counter() - start)
    if terminal or truncated:
        winners[info["winner"]] = winners.get(info["winner"], 0) + 1
        episodes += 1

    return {
        "worker": worker_id,
        "world_steps": steps,
        "agent_decisions": steps * 3,
        "episodes": episodes,
        "elapsed_sec": elapsed,
        "world_steps_per_sec": steps / elapsed,
        "agent_decisions_per_sec": steps * 3 / elapsed,
        "winners": winners,
        "last_observation_dim": int(next(iter(observations.values())).shape[0]),
    }


def _sum_winners(worker_results: list[dict]) -> dict:
    winners: dict[str, int] = {}
    for item in worker_results:
        for winner, count in item["winners"].items():
            winners[winner] = winners.get(winner, 0) + int(count)
    return dict(sorted(winners.items()))


def run_benchmark(args: argparse.Namespace) -> dict:
    workers = max(1, int(args.workers))
    max_ticks = args.max_ticks or TICK_LIMIT
    worker_args = [
        {
            "worker_id": worker_id,
            "duration_sec": args.duration_sec,
            "seed": args.seed,
            "opponent_bot": args.opponent_bot,
            "ally_bot": args.ally_bot,
            "opponent_policy_checkpoint": args.opponent_policy_checkpoint,
            "opponent_policy_deterministic": args.opponent_policy_mode == "argmax",
            "max_ticks": max_ticks,
        }
        for worker_id in range(workers)
    ]

    started = time.perf_counter()
    if workers == 1:
        worker_results = [_run_worker(worker_args[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            worker_results = pool.map(_run_worker, worker_args)
    elapsed = max(0.001, time.perf_counter() - started)
    world_steps = sum(item["world_steps"] for item in worker_results)
    agent_decisions = sum(item["agent_decisions"] for item in worker_results)
    episodes = sum(item["episodes"] for item in worker_results)

    return {
        "kind": "team_training_env_step_benchmark_no_learning",
        "duration_sec_requested": float(args.duration_sec),
        "elapsed_sec": round(elapsed, 3),
        "workers": workers,
        "cpu_count": os.cpu_count(),
        "opponent_bot": args.opponent_bot,
        "opponent_policy_checkpoint": args.opponent_policy_checkpoint,
        "opponent_policy_mode": args.opponent_policy_mode,
        "ally_bot": args.ally_bot,
        "max_ticks": max_ticks,
        "record_replay": False,
        "record_trace": False,
        "observation_version": OBSERVATION_VERSION,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "world_steps": world_steps,
        "world_steps_per_sec": round(world_steps / elapsed, 3),
        "world_steps_per_hour": round(world_steps * 3600.0 / elapsed),
        "agent_decisions": agent_decisions,
        "agent_decisions_per_sec": round(agent_decisions / elapsed, 3),
        "agent_decisions_per_hour": round(agent_decisions * 3600.0 / elapsed),
        "episodes": episodes,
        "winners": _sum_winners(worker_results),
        "workers_detail": [
            {
                "worker": item["worker"],
                "world_steps": item["world_steps"],
                "agent_decisions": item["agent_decisions"],
                "episodes": item["episodes"],
                "world_steps_per_sec": round(item["world_steps_per_sec"], 3),
                "agent_decisions_per_sec": round(item["agent_decisions_per_sec"], 3),
                "winners": item["winners"],
                "last_observation_dim": item["last_observation_dim"],
            }
            for item in worker_results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark TeamTrainingEnv step throughput without learning")
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260604)
    scripted_choices = ["scripted", "scripted_core", "scripted_core_raider", "scripted_core_bmp"]
    parser.add_argument("--opponent-bot", choices=scripted_choices + ["random", "policy"], default="scripted")
    parser.add_argument("--ally-bot", choices=scripted_choices + ["random"], default="scripted")
    parser.add_argument("--opponent-policy-checkpoint", default=None)
    parser.add_argument("--opponent-policy-mode", choices=["stochastic", "argmax"], default="stochastic")
    parser.add_argument("--max-ticks", type=int, default=None)
    args = parser.parse_args()
    if args.opponent_bot == "policy" and not args.opponent_policy_checkpoint:
        raise ValueError("--opponent-policy-checkpoint is required with --opponent-bot policy")
    if args.opponent_bot != "policy" and args.opponent_policy_checkpoint:
        raise ValueError("--opponent-policy-checkpoint requires --opponent-bot policy")

    print(json.dumps(run_benchmark(args), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
