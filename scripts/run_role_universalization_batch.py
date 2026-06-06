from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bots.policy_bot import checkpoint_is_compatible


@dataclass(frozen=True)
class UniversalizationRun:
    run_id: str
    run_dir: str
    resume: str
    seed: int
    opponent_bot: str = "scripted"
    opponent_policy_checkpoint: str | None = None
    opponent_policy_mode: str = "stochastic"
    entropy_coef: float = 0.015
    learning_rate: float = 2.0e-4


SEED_RUNS: tuple[UniversalizationRun, ...] = (
    UniversalizationRun(
        run_id="role_universal_seed01_scripted_std",
        run_dir="training_runs/role_universal_seed01_scripted_std_s20260701",
        resume="training_runs/population_s3_seed01_scripted_std_s20260651/checkpoints/latest.pt",
        seed=20260701,
    ),
    UniversalizationRun(
        run_id="role_universal_seed02_scripted_lowent",
        run_dir="training_runs/role_universal_seed02_scripted_lowent_s20260702",
        resume="training_runs/population_s3_seed02_scripted_lowent_s20260652/checkpoints/latest.pt",
        seed=20260702,
    ),
    UniversalizationRun(
        run_id="role_universal_seed03_scripted_highent",
        run_dir="training_runs/role_universal_seed03_scripted_highent_s20260703",
        resume="training_runs/population_s3_seed03_scripted_highent_s20260653/checkpoints/latest.pt",
        seed=20260703,
    ),
    UniversalizationRun(
        run_id="role_universal_seed04_warmstart_seed03_vs_u280",
        run_dir="training_runs/role_universal_seed04_warmstart_seed03_vs_u280_s20260704",
        resume="training_runs/population_s3_seed04_warmstart_seed03_vs_u280_s20260664/checkpoints/latest.pt",
        seed=20260704,
    ),
    UniversalizationRun(
        run_id="role_universal_seed05_resume_u280_vs_u380",
        run_dir="training_runs/role_universal_seed05_resume_u280_vs_u380_s20260705",
        resume="training_runs/population_s3_seed05_resume_u280_vs_u380_s20260655/checkpoints/latest.pt",
        seed=20260705,
    ),
    UniversalizationRun(
        run_id="role_universal_seed06_resume_u380_vs_u240_argmax",
        run_dir="training_runs/role_universal_seed06_resume_u380_vs_u240_argmax_s20260706",
        resume="training_runs/population_s3_seed06_resume_u380_vs_u240_argmax_s20260656/checkpoints/latest.pt",
        seed=20260706,
    ),
    UniversalizationRun(
        run_id="role_universal_seed07_scripted_altent",
        run_dir="training_runs/role_universal_seed07_scripted_altent_s20260707",
        resume="training_runs/population_s3_seed07_scripted_altent_s20260657/checkpoints/latest.pt",
        seed=20260707,
    ),
)


def command_for_run(run: UniversalizationRun, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_team_policy.py",
        "--run-dir",
        run.run_dir,
        "--overwrite",
        "--resume",
        run.resume,
        "--updates",
        str(args.updates),
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
        run.opponent_bot,
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
    if run.opponent_policy_checkpoint:
        command.extend([
            "--opponent-policy-checkpoint",
            run.opponent_policy_checkpoint,
            "--opponent-policy-mode",
            run.opponent_policy_mode,
        ])
    return command


def validate_runs(runs: tuple[UniversalizationRun, ...]) -> None:
    for run in runs:
        checkpoint = ROOT / run.resume
        if not checkpoint.exists():
            raise FileNotFoundError(f"{run.run_id} references missing checkpoint: {run.resume}")
        if not checkpoint_is_compatible(checkpoint):
            raise ValueError(f"{run.run_id} references incompatible checkpoint: {run.resume}")
        if run.opponent_policy_checkpoint:
            opponent = ROOT / run.opponent_policy_checkpoint
            if not opponent.exists():
                raise FileNotFoundError(
                    f"{run.run_id} references missing opponent checkpoint: {run.opponent_policy_checkpoint}"
                )
            if not checkpoint_is_compatible(opponent):
                raise ValueError(
                    f"{run.run_id} references incompatible opponent checkpoint: {run.opponent_policy_checkpoint}"
                )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def build_manifest(batch_dir: Path, runs: tuple[UniversalizationRun, ...]) -> dict:
    entries = []
    for run in runs:
        checkpoint = ROOT / run.run_dir / "checkpoints" / "latest.pt"
        entries.append({
            "run_id": run.run_id,
            "run_dir": run.run_dir,
            "checkpoint": checkpoint.relative_to(ROOT).as_posix() if checkpoint.exists() else None,
            "compatible": checkpoint_is_compatible(checkpoint) if checkpoint.exists() else False,
            "role_assignment_mode": "shuffle",
            "root_resume": run.resume,
        })
    manifest = {
        "kind": "role_universalization_manifest",
        "batch_dir": batch_dir.relative_to(ROOT).as_posix(),
        "entries": entries,
    }
    write_json(batch_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Universalize population roots with role-shuffle PPO continuation")
    parser.add_argument("--batch-id", default="sprint3_role_universalization_v1")
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validate_runs(SEED_RUNS)
    batch_dir = ROOT / "training_runs" / "role_universalization_batches" / args.batch_id
    logs_dir = batch_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "kind": "role_universalization_plan",
        "batch_id": args.batch_id,
        "args": vars(args),
        "runs": [asdict(run) for run in SEED_RUNS],
        "commands": [
            command_for_run(run, args)
            for run in SEED_RUNS
        ],
    }
    write_json(batch_dir / "plan.json", plan)
    print(json.dumps({"kind": "role_universalization_plan_written", "path": str(batch_dir / "plan.json")}, sort_keys=True), flush=True)

    if args.dry_run:
        build_manifest(batch_dir, SEED_RUNS)
        return 0

    status = {
        "kind": "role_universalization_status",
        "batch_id": args.batch_id,
        "runs": [],
    }
    for run in SEED_RUNS:
        command = command_for_run(run, args)
        log_path = logs_dir / f"{run.run_id}.log"
        print(json.dumps({
            "kind": "role_universalization_run_started",
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
        print(json.dumps({"kind": "role_universalization_run_finished", **entry}, sort_keys=True), flush=True)
        if process.returncode != 0:
            build_manifest(batch_dir, SEED_RUNS)
            return process.returncode

    manifest = build_manifest(batch_dir, SEED_RUNS)
    print(json.dumps({
        "kind": "role_universalization_batch_finished",
        "batch_id": args.batch_id,
        "manifest_path": str(batch_dir / "manifest.json"),
        "compatible": sum(1 for item in manifest["entries"] if item["compatible"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
