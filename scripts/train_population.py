"""Launch population self-play training for the 18-net free-agent draft pool.

This is the single entry point another agent runs to train the population and
monitor it. It does NOT cull or select nets -- all 18 are trained and
checkpointed every round.

Smoke (fast end-to-end sanity, a few CPU-minutes):
    python scripts/train_population.py --smoke

Full run (size with --updates / --episodes-per-net / --workers):
    python scripts/train_population.py --run-dir training_runs/population_18_v1 \
        --updates 400 --episodes-per-net 3 --workers 16

Resume a stopped run (re-run the same command; latest per-net checkpoints load
automatically). Inspect only, without training:
    python scripts/train_population.py --init-only
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.training.population import (
    PopulationConfig,
    PopulationTrainer,
    build_default_population,
    count_total_parameters,
    parameter_count,
)
from src.training.lineage import BranchingConfig, run_branch_select_cycle
from src.training.ppo_trainer import append_jsonl
from src.training.spaces import ACTION_DIM, OBSERVATION_DIM, OBSERVATION_VERSION


def _init_rollout_worker(n_threads: int) -> None:
    """Pool initializer: set PyTorch thread count before any work arrives."""
    torch.set_num_threads(n_threads)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 18-net population via frozen-snapshot self-play")
    parser.add_argument("--run-dir", default="training_runs/population_18_v1")
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--episodes-per-net", type=int, default=3,
                        help="Episodes each net plays as a controlled learner per update")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--max-ticks", type=int, default=300)
    parser.add_argument("--snapshot-interval", type=int, default=5,
                        help="Refresh the frozen opponent pool every N updates")
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--master-seed", type=int, default=20260609)
    parser.add_argument("--opponent-mode", choices=["stochastic", "argmax"], default="stochastic")
    parser.add_argument("--init-only", action="store_true",
                        help="Write the population spec, registry and initial checkpoints, then exit")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny end-to-end run for validation (overrides sizing args)")
    # --- per-lineage branching & selection ---
    parser.add_argument("--branch-interval", type=int, default=0,
                        help="Run per-lineage branch-train-eval-select every N updates "
                             "(0 = disabled).  Roots preserved; pool always stays at 18.")
    parser.add_argument("--branch-n", type=int, default=3,
                        help="Branches per player per branching cycle")
    parser.add_argument("--branch-mini-updates", type=int, default=3,
                        help="PPO mini-updates per branch before eval")
    parser.add_argument("--branch-mini-episodes", type=int, default=3,
                        help="Episodes per mini-update during branch training")
    parser.add_argument("--branch-eval-episodes", type=int, default=12,
                        help="Evaluation episodes per branch (same opponents → fair)")
    parser.add_argument("--mutation-scale", type=float, default=0.004,
                        help="Gaussian weight-perturbation std for branch creation")
    return parser.parse_args()


def apply_smoke(args: argparse.Namespace) -> None:
    args.run_dir = args.run_dir if args.run_dir != "training_runs/population_18_v1" else "training_runs/population_18_smoke"
    args.updates = 2
    args.episodes_per_net = 1
    args.workers = 1
    args.max_ticks = 60
    args.snapshot_interval = 1
    args.checkpoint_interval = 1


def make_config(args: argparse.Namespace) -> PopulationConfig:
    return PopulationConfig(
        run_dir=args.run_dir,
        nets=build_default_population(),
        master_seed=args.master_seed,
        updates=args.updates,
        episodes_per_net_per_update=args.episodes_per_net,
        workers=max(1, args.workers),
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        max_ticks=args.max_ticks,
        snapshot_interval=max(1, args.snapshot_interval),
        checkpoint_interval=max(1, args.checkpoint_interval),
        torch_threads=max(1, args.torch_threads),
        device=args.device,
        opponent_deterministic=(args.opponent_mode == "argmax"),
        mixed_faction=True,
    )


def git_version_info() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True,
                                             stderr=subprocess.DEVNULL).strip())
    except Exception:
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def build_registry(config: PopulationConfig, trainer: PopulationTrainer, update: int) -> dict:
    entries = []
    for spec in config.nets:
        net = trainer.nets[spec.net_id]
        checkpoint = trainer.net_dir(spec.net_id) / "latest.pt"
        checkpoint_rel = (
            checkpoint.as_posix() if not checkpoint.is_absolute()
            else checkpoint.relative_to(ROOT).as_posix()
        )
        entries.append({
            "net_id": spec.net_id,
            "faction": spec.faction,
            "seed": spec.seed,
            "recipe": spec.recipe,
            "genotype_hash": spec.genotype_hash(),
            "hidden_size": spec.hidden_size,
            "learning_rate": spec.learning_rate,
            "entropy_coef": spec.entropy_coef,
            "clip_coef": spec.clip_coef,
            "parameter_count": parameter_count(net),
            "checkpoint": checkpoint_rel if checkpoint.exists() else None,
            "policy_state_id": spec.net_id,
            "generation": 0,
            "parent_ids": [],
            "status": "training" if update < config.updates else "trained",
            "updates_trained": update,
        })
    return {
        "kind": "population_18_registry",
        "run_dir": config.run_dir,
        "observation_version": OBSERVATION_VERSION,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "net_count": len(entries),
        "factions": {faction: sum(1 for e in entries if e["faction"] == faction)
                     for faction in sorted({e["faction"] for e in entries})},
        "unique_genotypes": len({e["genotype_hash"] for e in entries}),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "entries": entries,
    }


def write_registry_markdown(path: Path, registry: dict) -> None:
    lines = [
        "# Population 18 — Model Registry",
        "",
        f"- run_dir: `{registry['run_dir']}`",
        f"- observation: `{registry['observation_version']}` (obs={registry['observation_dim']}, act={registry['action_dim']})",
        f"- nets: {registry['net_count']} | unique genotypes: {registry['unique_genotypes']} | factions: {registry['factions']}",
        f"- updated_at: {registry['updated_at']}",
        "",
        "| net_id | faction | recipe | seed | genotype | params | updates | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for e in registry["entries"]:
        lines.append(
            f"| {e['net_id']} | {e['faction']} | {e['recipe']} | {e['seed']} | "
            f"`{e['genotype_hash']}` | {e['parameter_count']} | {e['updates_trained']} | {e['status']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def print_json(item: dict) -> None:
    print(json.dumps(item, ensure_ascii=False, sort_keys=True), flush=True)


def persist_registry(config: PopulationConfig, trainer: PopulationTrainer, run_dir: Path, update: int) -> dict:
    registry = build_registry(config, trainer, update)
    write_json(run_dir / "registry.json", registry)
    write_registry_markdown(run_dir / "MODEL_REGISTRY.md", registry)
    return registry


def main() -> int:
    args = parse_args()
    if args.smoke:
        apply_smoke(args)

    config = make_config(args)
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    version_info = git_version_info()
    write_json(run_dir / "population.json", {
        "kind": "population_18_spec",
        "master_seed": config.master_seed,
        "nets": [asdict(spec) for spec in config.nets],
        "genotype_hashes": {spec.net_id: spec.genotype_hash() for spec in config.nets},
    })
    write_json(run_dir / "config.json", {
        "config": {key: value for key, value in asdict(config).items() if key != "nets"},
        "args": vars(args),
        **version_info,
    })

    trainer = PopulationTrainer(config)
    trainer.resume()

    if trainer.start_update == 1:
        trainer.save_all(update=0, milestone=False)
    registry = persist_registry(config, trainer, run_dir, trainer.start_update - 1)

    print_json({
        "kind": "population_training_prepared",
        "run_dir": str(run_dir),
        "device": str(trainer.device),
        "nets": len(config.nets),
        "unique_genotypes": registry["unique_genotypes"],
        "total_parameters": count_total_parameters(trainer),
        "start_update": trainer.start_update,
        "planned_updates": config.updates,
        "workers": config.workers,
        **version_info,
    })

    if args.init_only:
        print_json({"kind": "population_init_only_done", "run_dir": str(run_dir)})
        return 0

    metrics_path = run_dir / "metrics.jsonl"
    ctx = mp.get_context("spawn")
    pool = None
    current_pool_path = None
    last_refresh = -10**9

    # Per-lineage branching config (disabled when branch_interval == 0).
    branching_cfg: BranchingConfig | None = None
    if args.branch_interval > 0:
        branching_cfg = BranchingConfig(
            n_branches=args.branch_n,
            mini_updates=args.branch_mini_updates,
            mini_episodes_per_update=args.branch_mini_episodes,
            eval_episodes=args.branch_eval_episodes,
            mutation_scale=args.mutation_scale,
            branch_interval=args.branch_interval,
        )
    branch_cycle_num = 0

    def ensure_pool(refresh: bool):
        nonlocal pool
        if refresh and pool is not None:
            pool.close()
            pool.join()
            pool = None
        if pool is None and config.workers > 1:
            pool = ctx.Pool(
                processes=config.workers,
                initializer=_init_rollout_worker,
                initargs=(config.torch_threads,),
            )

    try:
        end_update = trainer.start_update + config.updates
        for update in range(trainer.start_update, end_update):
            refresh = (current_pool_path is None) or (update - last_refresh >= config.snapshot_interval)
            if refresh:
                current_pool_path = trainer.refresh_frozen_pool(version=update)
                last_refresh = update
                ensure_pool(refresh=True)  # rebuild workers so frozen-state caches stay bounded
                _cleanup_old_pools(run_dir, keep_version=update)
            elif pool is None and config.workers > 1:
                ensure_pool(refresh=False)

            started = time.perf_counter()
            summary = trainer.run_update(update, current_pool_path, pool)
            summary["elapsed_sec"] = round(time.perf_counter() - started, 3)
            summary["frozen_pool_version"] = last_refresh
            summary.update(version_info)
            append_jsonl(metrics_path, summary)
            print_json({key: value for key, value in summary.items() if key != "loss_by_net"})

            if update % config.checkpoint_interval == 0 or update == end_update - 1:
                trainer.save_all(update=update, milestone=True)
                persist_registry(config, trainer, run_dir, update)

            # Per-lineage branch-train-eval-select cycle.
            if branching_cfg is not None and update % branching_cfg.branch_interval == 0:
                branch_cycle_num += 1
                # Pause the worker pool during branching (single-threaded per branch).
                if pool is not None:
                    pool.close()
                    pool.join()
                    pool = None

                branch_summary = run_branch_select_cycle(
                    trainer=trainer,
                    branching_config=branching_cfg,
                    frozen_pool_path=str(current_pool_path),
                    cycle_num=branch_cycle_num,
                    update=update,
                )
                append_jsonl(metrics_path, branch_summary)
                print_json({k: v for k, v in branch_summary.items() if k != "player_results"})

                # After branching, refresh the frozen pool so opponents reflect new winner weights.
                current_pool_path = trainer.refresh_frozen_pool(version=update)
                last_refresh = update
                _cleanup_old_pools(run_dir, keep_version=update)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    persist_registry(config, trainer, run_dir, end_update - 1)
    print_json({
        "kind": "population_training_finished",
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "total_steps": trainer.total_steps,
        "registry": str(run_dir / "registry.json"),
        **version_info,
    })
    return 0


def _cleanup_old_pools(run_dir: Path, keep_version: int) -> None:
    """Keep only the current frozen-pool file to bound disk usage."""
    pool_dir = run_dir / "frozen_pool"
    if not pool_dir.exists():
        return
    for path in pool_dir.glob("frozen_pool_v*.pt"):
        try:
            version = int(path.stem.split("_v")[-1])
        except ValueError:
            continue
        if version != keep_version:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
