from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bots.policy_bot import checkpoint_is_compatible


TMIX08 = "training_runs/fresh_population_v31_teammate_mix_tmix08_hard345_s20264108/checkpoints/checkpoint_update_100.pt"
TMIX09 = "training_runs/fresh_population_v31_teammate_mix_tmix09_hard561_s20264109/checkpoints/checkpoint_update_60.pt"
TMIX10 = "training_runs/fresh_population_v31_teammate_mix_tmix10_hard987_s20264110/checkpoints/checkpoint_update_80.pt"
TMIX11 = "training_runs/fresh_population_v31_teammate_mix_tmix11_hard978_s20264111/checkpoints/checkpoint_update_80.pt"


@dataclass(frozen=True)
class SelfPlayRun:
    run_id: str
    run_dir: str
    resume: str
    opponent_policy_checkpoint: str
    seed: int
    updates: int
    entropy_coef: float = 0.006
    learning_rate: float = 5.0e-5
    description: str = ""


SELFPLAY_RUNS: tuple[SelfPlayRun, ...] = (
    SelfPlayRun(
        run_id="spv3_01_tmix10_vs_tmix11",
        run_dir="training_runs/fresh_population_v31_selfplay_v3_tmix10_vs_tmix11_s20264201",
        resume=TMIX10,
        opponent_policy_checkpoint=TMIX11,
        seed=20264201,
        updates=160,
        description="continue hard987 recovery line against hard978 stochastic peer",
    ),
    SelfPlayRun(
        run_id="spv3_02_tmix11_vs_tmix08",
        run_dir="training_runs/fresh_population_v31_selfplay_v3_tmix11_vs_tmix08_s20264202",
        resume=TMIX11,
        opponent_policy_checkpoint=TMIX08,
        seed=20264202,
        updates=160,
        description="continue hard978 line against stable hard345 stochastic peer",
    ),
    SelfPlayRun(
        run_id="spv3_03_tmix08_vs_tmix09",
        run_dir="training_runs/fresh_population_v31_selfplay_v3_tmix08_vs_tmix09_s20264203",
        resume=TMIX08,
        opponent_policy_checkpoint=TMIX09,
        seed=20264203,
        updates=140,
        description="keep hard345 line training against hard561 stochastic peer",
    ),
    SelfPlayRun(
        run_id="spv3_04_tmix09_vs_tmix10",
        run_dir="training_runs/fresh_population_v31_selfplay_v3_tmix09_vs_tmix10_s20264204",
        resume=TMIX09,
        opponent_policy_checkpoint=TMIX10,
        seed=20264204,
        updates=180,
        entropy_coef=0.007,
        description="stabilize weakest native stochastic hard561 line against hard987 peer",
    ),
    SelfPlayRun(
        run_id="spv3_05_tmix10_vs_tmix08",
        run_dir="training_runs/fresh_population_v31_selfplay_v3_tmix10_vs_tmix08_s20264205",
        resume=TMIX10,
        opponent_policy_checkpoint=TMIX08,
        seed=20264205,
        updates=140,
        description="second hard987 branch against stable hard345 peer",
    ),
    SelfPlayRun(
        run_id="spv3_06_tmix11_vs_tmix09",
        run_dir="training_runs/fresh_population_v31_selfplay_v3_tmix11_vs_tmix09_s20264206",
        resume=TMIX11,
        opponent_policy_checkpoint=TMIX09,
        seed=20264206,
        updates=140,
        description="second hard978 branch against hard561 peer",
    ),
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def validate_checkpoint(label: str, checkpoint: str) -> None:
    path = ROOT / checkpoint
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {checkpoint}")
    if not checkpoint_is_compatible(path):
        raise ValueError(f"{label} incompatible: {checkpoint}")


def command_for_run(run: SelfPlayRun, args: argparse.Namespace) -> list[str]:
    updates = int(args.smoke_updates or run.updates)
    return [
        sys.executable,
        "scripts/train_team_policy.py",
        "--run-dir",
        run.run_dir,
        "--overwrite",
        "--resume",
        run.resume,
        "--updates",
        str(updates),
        "--workers",
        str(args.workers),
        "--rollout-steps",
        str(args.rollout_steps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--eval-interval",
        str(args.eval_interval),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--seed",
        str(run.seed),
        "--opponent-bot",
        "policy",
        "--opponent-policy-checkpoint",
        run.opponent_policy_checkpoint,
        "--opponent-policy-mode",
        "stochastic",
        "--entropy-coef",
        str(run.entropy_coef),
        "--learning-rate",
        str(run.learning_rate),
        "--torch-threads",
        str(args.torch_threads),
        "--role-assignment-mode",
        "shuffle",
        "--random-baseline-episodes",
        "0",
    ]


def validate_runs(runs: tuple[SelfPlayRun, ...]) -> None:
    for run in runs:
        validate_checkpoint(f"{run.run_id} resume", run.resume)
        validate_checkpoint(f"{run.run_id} opponent", run.opponent_policy_checkpoint)


def build_manifest(batch_dir: Path, runs: tuple[SelfPlayRun, ...]) -> dict:
    entries = []
    for run in runs:
        checkpoint = ROOT / run.run_dir / "checkpoints" / "latest.pt"
        entries.append({
            "run_id": run.run_id,
            "description": run.description,
            "run_dir": run.run_dir,
            "checkpoint": checkpoint.relative_to(ROOT).as_posix() if checkpoint.exists() else None,
            "compatible": checkpoint_is_compatible(checkpoint) if checkpoint.exists() else False,
            "role_assignment_mode": "shuffle",
            "resume": run.resume,
            "opponent_policy_checkpoint": run.opponent_policy_checkpoint,
            "opponent_policy_mode": "stochastic",
            "seed": run.seed,
            "planned_updates": run.updates,
        })
    manifest = {
        "kind": "fresh_population_v31_selfplay_v3_pool_manifest",
        "batch_dir": batch_dir.relative_to(ROOT).as_posix(),
        "entries": entries,
    }
    write_json(batch_dir / "manifest.json", manifest)
    write_json(ROOT / "training_runs" / "self_play_pools" / f"{batch_dir.name}_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v3.1 self-play continuations from teammate-mix v3 pool")
    parser.add_argument("--batch-id", default="fresh_population_v31_selfplay_v3_pool_v1")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0, help="0-based run index to start from")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validate_runs(SELFPLAY_RUNS)
    selected_runs = SELFPLAY_RUNS[max(0, int(args.start_index)) :]
    batch_dir = ROOT / "training_runs" / "fresh_population_v31_selfplay_batches" / args.batch_id
    logs_dir = batch_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "kind": "fresh_population_v31_selfplay_v3_pool_plan",
        "batch_id": args.batch_id,
        "args": vars(args),
        "runs": [asdict(run) for run in selected_runs],
        "commands": [command_for_run(run, args) for run in selected_runs],
    }
    write_json(batch_dir / "plan.json", plan)
    print(json.dumps({"kind": "fresh_population_v31_selfplay_v3_pool_plan_written", "path": str(batch_dir / "plan.json")}, sort_keys=True), flush=True)

    if args.dry_run:
        build_manifest(batch_dir, SELFPLAY_RUNS)
        return 0

    status = {
        "kind": "fresh_population_v31_selfplay_v3_pool_status",
        "batch_id": args.batch_id,
        "runs": [],
    }
    for run in selected_runs:
        command = command_for_run(run, args)
        log_path = logs_dir / f"{run.run_id}.log"
        print(json.dumps({
            "kind": "fresh_population_v31_selfplay_v3_pool_run_started",
            "run_id": run.run_id,
            "run_dir": run.run_dir,
            "log_path": str(log_path),
        }, sort_keys=True), flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        entry = {
            "run_id": run.run_id,
            "run_dir": run.run_dir,
            "returncode": process.returncode,
            "log_path": log_path.relative_to(ROOT).as_posix(),
        }
        status["runs"].append(entry)
        write_json(batch_dir / "status.json", status)
        print(json.dumps({"kind": "fresh_population_v31_selfplay_v3_pool_run_finished", **entry}, sort_keys=True), flush=True)
        if process.returncode != 0:
            return process.returncode

    manifest = build_manifest(batch_dir, SELFPLAY_RUNS)
    print(json.dumps({
        "kind": "fresh_population_v31_selfplay_v3_pool_finished",
        "batch_id": args.batch_id,
        "compatible": sum(1 for entry in manifest["entries"] if entry["compatible"]),
        "runs": len(manifest["entries"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
