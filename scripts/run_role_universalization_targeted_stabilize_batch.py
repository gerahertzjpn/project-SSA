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


@dataclass(frozen=True)
class TargetedStabilizeRun:
    run_id: str
    run_dir: str
    resume: str
    seed: int
    entropy_coef: float = 0.003
    learning_rate: float = 8.0e-5


OPPONENT_CHECKPOINT = "training_runs/sprint3_team_scripted_fresh_v1/checkpoints/checkpoint_update_280.pt"


TARGETED_STABILIZE_RUNS: tuple[TargetedStabilizeRun, ...] = (
    TargetedStabilizeRun(
        run_id="role_universal_v4_seed02_u280_stoch_lowent",
        run_dir="training_runs/role_universal_v4_seed02_u280_stoch_lowent_s20261002",
        resume="training_runs/role_universal_stabilize_seed02_u280_stoch_s20260902/checkpoints/checkpoint_update_400.pt",
        seed=20261002,
    ),
    TargetedStabilizeRun(
        run_id="role_universal_v4_seed04_u280_stoch_lowent",
        run_dir="training_runs/role_universal_v4_seed04_u280_stoch_lowent_s20261004",
        resume="training_runs/role_universal_stabilize_seed04_u280_stoch_s20260904/checkpoints/latest.pt",
        seed=20261004,
    ),
    TargetedStabilizeRun(
        run_id="role_universal_v4_seed06_u280_stoch_lowent",
        run_dir="training_runs/role_universal_v4_seed06_u280_stoch_lowent_s20261006",
        resume="training_runs/role_universal_stabilize_seed06_u280_stoch_s20260906/checkpoints/latest.pt",
        seed=20261006,
    ),
    TargetedStabilizeRun(
        run_id="role_universal_v4_seed07_u280_stoch_lowent",
        run_dir="training_runs/role_universal_v4_seed07_u280_stoch_lowent_s20261007",
        resume="training_runs/role_universal_stabilize_seed07_u280_stoch_s20260907/checkpoints/latest.pt",
        seed=20261007,
    ),
)


def command_for_run(run: TargetedStabilizeRun, args: argparse.Namespace) -> list[str]:
    return [
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
        "policy",
        "--opponent-policy-checkpoint",
        OPPONENT_CHECKPOINT,
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


def validate_runs(runs: tuple[TargetedStabilizeRun, ...]) -> None:
    opponent = ROOT / OPPONENT_CHECKPOINT
    if not opponent.exists():
        raise FileNotFoundError(f"Missing opponent checkpoint: {OPPONENT_CHECKPOINT}")
    if not checkpoint_is_compatible(opponent):
        raise ValueError(f"Incompatible opponent checkpoint: {OPPONENT_CHECKPOINT}")
    for run in runs:
        resume = ROOT / run.resume
        if not resume.exists():
            raise FileNotFoundError(f"{run.run_id} references missing checkpoint: {run.resume}")
        if not checkpoint_is_compatible(resume):
            raise ValueError(f"{run.run_id} references incompatible checkpoint: {run.resume}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def build_manifest(batch_dir: Path, runs: tuple[TargetedStabilizeRun, ...]) -> dict:
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
            "opponent_policy_checkpoint": OPPONENT_CHECKPOINT,
            "opponent_policy_mode": "stochastic",
        })
    manifest = {
        "kind": "role_universalization_targeted_stabilize_manifest",
        "batch_dir": batch_dir.relative_to(ROOT).as_posix(),
        "entries": entries,
    }
    write_json(batch_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Targeted low-entropy role-universal stabilization for weak V3 roots"
    )
    parser.add_argument("--batch-id", default="sprint3_role_universalization_targeted_v4")
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validate_runs(TARGETED_STABILIZE_RUNS)
    batch_dir = ROOT / "training_runs" / "role_universalization_batches" / args.batch_id
    logs_dir = batch_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "kind": "role_universalization_targeted_stabilize_plan",
        "batch_id": args.batch_id,
        "args": vars(args),
        "runs": [asdict(run) for run in TARGETED_STABILIZE_RUNS],
        "commands": [command_for_run(run, args) for run in TARGETED_STABILIZE_RUNS],
    }
    write_json(batch_dir / "plan.json", plan)
    print(json.dumps({"kind": "role_universalization_targeted_plan_written", "path": str(batch_dir / "plan.json")}, sort_keys=True), flush=True)

    if args.dry_run:
        build_manifest(batch_dir, TARGETED_STABILIZE_RUNS)
        return 0

    status = {
        "kind": "role_universalization_targeted_stabilize_status",
        "batch_id": args.batch_id,
        "runs": [],
    }
    for run in TARGETED_STABILIZE_RUNS:
        command = command_for_run(run, args)
        log_path = logs_dir / f"{run.run_id}.log"
        print(json.dumps({
            "kind": "role_universalization_targeted_run_started",
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
        print(json.dumps({"kind": "role_universalization_targeted_run_finished", **entry}, sort_keys=True), flush=True)
        if process.returncode != 0:
            build_manifest(batch_dir, TARGETED_STABILIZE_RUNS)
            return process.returncode

    manifest = build_manifest(batch_dir, TARGETED_STABILIZE_RUNS)
    print(json.dumps({
        "kind": "role_universalization_targeted_batch_finished",
        "batch_id": args.batch_id,
        "manifest_path": str(batch_dir / "manifest.json"),
        "compatible": sum(1 for item in manifest["entries"] if item["compatible"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
