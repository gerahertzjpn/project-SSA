import argparse
import json
import multiprocessing as mp
import os
import sys
import time


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.constants import TICK_LIMIT
from src.training.env import TrainingEnv, TrainingEnvConfig


def _run_worker(args: dict) -> dict:
    worker_id = int(args["worker_id"])
    duration_sec = float(args["duration_sec"])
    base_seed = int(args["seed"])
    max_ticks = int(args["max_ticks"])
    progress_sec = float(args["progress_sec"])
    progress = bool(args["progress"])

    config = TrainingEnvConfig(
        controlled_player=args["controlled_player"],
        seed=base_seed + worker_id,
        opponent_bot=args["opponent_bot"],
        ally_bot=args["ally_bot"],
        max_ticks=max_ticks,
        record_replay=False,
        record_trace=False,
    )
    env = TrainingEnv(config)

    deadline = time.perf_counter() + duration_sec
    next_progress = time.perf_counter() + progress_sec
    steps = 0
    episodes = 0
    winners: dict[str, int] = {}
    obs, info = env.reset(seed=base_seed + worker_id)
    terminal = False
    truncated = False

    start = time.perf_counter()
    while time.perf_counter() < deadline:
        if terminal or truncated:
            winners[info["winner"]] = winners.get(info["winner"], 0) + 1
            episodes += 1
            episode_seed = base_seed + worker_id * 1_000_000 + episodes
            obs, info = env.reset(seed=episode_seed)
            terminal = False
            truncated = False

        action = env.sample_legal_action()
        obs, reward, terminal, truncated, info = env.step(action)
        steps += 1

        now = time.perf_counter()
        if progress and now >= next_progress:
            elapsed = max(0.001, now - start)
            print(
                json.dumps({
                    "kind": "benchmark_progress",
                    "worker": worker_id,
                    "elapsed_sec": round(elapsed, 3),
                    "steps": steps,
                    "steps_per_sec": round(steps / elapsed, 3),
                    "episodes": episodes,
                }, ensure_ascii=False),
                flush=True,
            )
            next_progress = now + progress_sec

    elapsed = max(0.001, time.perf_counter() - start)
    if terminal or truncated:
        winners[info["winner"]] = winners.get(info["winner"], 0) + 1
        episodes += 1

    return {
        "worker": worker_id,
        "steps": steps,
        "episodes": episodes,
        "elapsed_sec": elapsed,
        "steps_per_sec": steps / elapsed,
        "winners": winners,
        "last_observation_dim": int(obs.shape[0]),
    }


def _sum_winners(worker_results: list[dict]) -> dict:
    winners: dict[str, int] = {}
    for item in worker_results:
        for winner, count in item["winners"].items():
            winners[winner] = winners.get(winner, 0) + int(count)
    return winners


def run_benchmark(args: argparse.Namespace) -> dict:
    workers = max(1, int(args.workers))
    max_ticks = args.max_ticks or TICK_LIMIT
    worker_args = [
        {
            "worker_id": worker_id,
            "duration_sec": args.duration_sec,
            "seed": args.seed,
            "controlled_player": args.player,
            "opponent_bot": args.opponent_bot,
            "ally_bot": args.ally_bot,
            "max_ticks": max_ticks,
            "progress": args.progress and workers == 1,
            "progress_sec": args.progress_sec,
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
    total_steps = sum(item["steps"] for item in worker_results)
    total_episodes = sum(item["episodes"] for item in worker_results)

    return {
        "kind": "training_env_step_benchmark_no_learning",
        "duration_sec_requested": float(args.duration_sec),
        "elapsed_sec": round(elapsed, 3),
        "workers": workers,
        "cpu_count": os.cpu_count(),
        "controlled_player": args.player,
        "opponent_bot": args.opponent_bot,
        "ally_bot": args.ally_bot,
        "max_ticks": max_ticks,
        "record_replay": False,
        "record_trace": False,
        "steps": total_steps,
        "steps_per_sec": round(total_steps / elapsed, 3),
        "steps_per_20_min": round(total_steps * 1200.0 / elapsed),
        "episodes": total_episodes,
        "winners": _sum_winners(worker_results),
        "workers_detail": [
            {
                "worker": item["worker"],
                "steps": item["steps"],
                "episodes": item["episodes"],
                "steps_per_sec": round(item["steps_per_sec"], 3),
                "winners": item["winners"],
                "last_observation_dim": item["last_observation_dim"],
            }
            for item in worker_results
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TrainingEnv step throughput without learning")
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--player", default="A1")
    parser.add_argument("--opponent-bot", choices=["scripted", "random"], default="scripted")
    parser.add_argument("--ally-bot", choices=["scripted", "random"], default="scripted")
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-sec", type=float, default=60.0)
    args = parser.parse_args()

    summary = run_benchmark(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
