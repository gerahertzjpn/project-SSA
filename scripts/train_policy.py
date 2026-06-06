import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training.policy import MaskedActorCritic
from src.training.ppo_trainer import (
    PPOConfig,
    append_jsonl,
    collect_rollouts,
    evaluate_policy,
    evaluate_random_legal_policy,
    latest_metrics,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    summarize_training_episodes,
    update_policy,
)


def _run_dir_default() -> str:
    return time.strftime("training_runs/run_%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first masked PPO policy against baseline bots")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", default=None, help="Checkpoint path to load before training")
    parser.add_argument("--updates", type=int, default=10, help="Additional PPO updates to run in this invocation")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--rollout-steps", type=int, default=300)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--player", default="A1")
    scripted_choices = ["scripted", "scripted_core", "scripted_core_raider", "scripted_core_bmp"]
    parser.add_argument("--opponent-bot", choices=scripted_choices + ["random"], default="scripted")
    parser.add_argument("--ally-bot", choices=scripted_choices + ["random"], default="scripted")
    parser.add_argument("--max-ticks", type=int, default=300)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-baseline-episodes", type=int, default=50)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> PPOConfig:
    return PPOConfig(
        seed=args.seed,
        controlled_player=args.player,
        opponent_bot=args.opponent_bot,
        ally_bot=args.ally_bot,
        max_ticks=args.max_ticks,
        workers=max(1, args.workers),
        rollout_steps=args.rollout_steps,
        updates=args.updates,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        eval_episodes=args.eval_episodes,
        eval_interval=max(1, args.eval_interval),
        checkpoint_interval=max(1, args.checkpoint_interval),
        torch_threads=max(1, args.torch_threads),
        device=args.device,
    )


def prepare_run_dir(path: Path, overwrite: bool, resume: str | None) -> None:
    if path.exists() and overwrite:
        resolved = path.resolve()
        root = Path.cwd().resolve()
        if root not in resolved.parents and resolved != root:
            raise ValueError(f"Refusing to remove run dir outside workspace: {resolved}")
        shutil.rmtree(resolved)
    if path.exists() and any(path.iterdir()) and not resume and not overwrite:
        raise FileExistsError(f"Run dir already exists: {path}. Use --overwrite or --resume.")
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(parents=True, exist_ok=True)


def print_json(item: dict) -> None:
    print(json.dumps(item, ensure_ascii=False, sort_keys=True), flush=True)


def git_version_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status)}


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir or _run_dir_default())
    prepare_run_dir(run_dir, args.overwrite, args.resume)

    config = config_from_args(args)
    torch.set_num_threads(config.torch_threads)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = resolve_device(config.device)
    version_info = git_version_info()
    policy = MaskedActorCritic(hidden_size=config.hidden_size).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    start_update = 1
    total_steps = 0
    if args.resume:
        checkpoint = load_checkpoint(Path(args.resume), policy, optimizer, map_location=device)
        start_update = int(checkpoint["update"]) + 1
        total_steps = int(checkpoint.get("total_steps", 0))

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump({
            "config": config.__dict__,
            "device": str(device),
            "resume": args.resume,
            "start_update": start_update,
            **version_info,
        }, handle, ensure_ascii=False, indent=2, sort_keys=True)

    metrics_path = run_dir / "metrics.jsonl"
    print_json({
        "kind": "training_started",
        "run_dir": str(run_dir),
        "device": str(device),
        "workers": config.workers,
        "rollout_steps": config.rollout_steps,
        "updates_this_invocation": config.updates,
        "start_update": start_update,
        "resume": args.resume,
        **version_info,
    })

    if start_update == 1 and args.random_baseline_episodes > 0:
        random_eval = evaluate_random_legal_policy(
            config,
            episodes=args.random_baseline_episodes,
            seed=config.seed + 900_000_000,
        )
        item = {
            "kind": "random_legal_baseline_eval",
            "episodes": args.random_baseline_episodes,
            "eval": {key: value for key, value in random_eval.items() if key != "episodes_detail"},
        }
        append_jsonl(metrics_path, item)
        print_json(item)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=config.workers) if config.workers > 1 else None
    try:
        end_update = start_update + config.updates
        for update in range(start_update, end_update):
            started = time.perf_counter()
            rollout = collect_rollouts(policy, config, update, pool=pool)
            total_steps += int(len(rollout["actions"]))
            train_summary = summarize_training_episodes(rollout["episodes"], rollout["actions"], rollout["masks"])
            loss_summary = update_policy(policy, optimizer, rollout, config, device)

            eval_summary = None
            if update % config.eval_interval == 0:
                eval_summary = evaluate_policy(
                    policy,
                    config,
                    episodes=config.eval_episodes,
                    seed=config.seed + update * 50_000,
                    deterministic=True,
                    device=device,
                )

            elapsed = time.perf_counter() - started
            metrics = {
                "kind": "ppo_update",
                "update": update,
                "total_steps": total_steps,
                "elapsed_sec": round(elapsed, 3),
                "steps_per_sec": round(len(rollout["actions"]) / max(0.001, elapsed), 3),
                "train": train_summary,
                "loss": loss_summary,
                "eval_argmax": (
                    {key: value for key, value in eval_summary.items() if key != "episodes_detail"}
                    if eval_summary
                    else None
                ),
            }
            append_jsonl(metrics_path, metrics)
            print_json(metrics)

            if update % config.checkpoint_interval == 0:
                checkpoint_path = run_dir / "checkpoints" / f"checkpoint_update_{update}.pt"
                latest_path = run_dir / "checkpoints" / "latest.pt"
                save_checkpoint(checkpoint_path, policy, optimizer, config, update, total_steps)
                save_checkpoint(latest_path, policy, optimizer, config, update, total_steps)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    final_metrics = latest_metrics(metrics_path)
    print_json({
        "kind": "training_finished",
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "checkpoints_dir": str(run_dir / "checkpoints"),
        "records": len(final_metrics),
        "total_steps": total_steps,
        **version_info,
    })


if __name__ == "__main__":
    main()
