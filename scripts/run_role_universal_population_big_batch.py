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


U_SEED01 = "training_runs/role_universal_stabilize_seed01_u280_stoch_s20260901/checkpoints/latest.pt"
U_SEED02 = "training_runs/role_universal_v4_seed02_u280_stoch_lowent_s20261002/checkpoints/checkpoint_update_520.pt"
U_SEED03 = "training_runs/role_universal_stabilize_seed03_u280_stoch_s20260903/checkpoints/latest.pt"
U_SEED04 = "training_runs/role_universal_v4_seed04_u280_stoch_lowent_s20261004/checkpoints/checkpoint_update_560.pt"
U_SEED05 = "training_runs/role_universal_stabilize_seed05_u280_stoch_s20260905/checkpoints/checkpoint_update_640.pt"
U_SEED06 = "training_runs/role_universal_v4_seed06_u280_stoch_lowent_s20261006/checkpoints/checkpoint_update_820.pt"
U_SEED07 = "training_runs/role_universal_v4_seed07_u280_stoch_lowent_s20261007/checkpoints/checkpoint_update_320.pt"


@dataclass(frozen=True)
class PopulationRun:
    run_id: str
    run_dir: str
    resume: str
    opponent_policy_checkpoint: str
    seed: int
    updates: int
    opponent_policy_mode: str = "stochastic"
    entropy_coef: float = 0.005
    learning_rate: float = 8.0e-5
    description: str = ""


POPULATION_RUNS: tuple[PopulationRun, ...] = (
    PopulationRun(
        run_id="unipop_seed01_vs_seed03_stoch",
        run_dir="training_runs/universal_population_seed01_vs_seed03_stoch_s20261101",
        resume=U_SEED01,
        opponent_policy_checkpoint=U_SEED03,
        seed=20261101,
        updates=180,
        entropy_coef=0.006,
        description="seed01 universal root continued against strong seed03 stochastic peer",
    ),
    PopulationRun(
        run_id="unipop_seed02_vs_seed05_stoch",
        run_dir="training_runs/universal_population_seed02_vs_seed05_stoch_s20261102",
        resume=U_SEED02,
        opponent_policy_checkpoint=U_SEED05,
        seed=20261102,
        updates=180,
        entropy_coef=0.004,
        description="seed02 V4 root continued against strong seed05 stochastic peer",
    ),
    PopulationRun(
        run_id="unipop_seed03_vs_seed01_argmax",
        run_dir="training_runs/universal_population_seed03_vs_seed01_argmax_s20261103",
        resume=U_SEED03,
        opponent_policy_checkpoint=U_SEED01,
        opponent_policy_mode="argmax",
        seed=20261103,
        updates=160,
        entropy_coef=0.008,
        description="seed03 root hardened against deterministic seed01 peer",
    ),
    PopulationRun(
        run_id="unipop_seed04_vs_seed02_stoch",
        run_dir="training_runs/universal_population_seed04_vs_seed02_stoch_s20261104",
        resume=U_SEED04,
        opponent_policy_checkpoint=U_SEED02,
        seed=20261104,
        updates=180,
        entropy_coef=0.005,
        description="seed04 V4 root continued against seed02 V4 stochastic peer",
    ),
    PopulationRun(
        run_id="unipop_seed05_vs_seed07_stoch",
        run_dir="training_runs/universal_population_seed05_vs_seed07_stoch_s20261105",
        resume=U_SEED05,
        opponent_policy_checkpoint=U_SEED07,
        seed=20261105,
        updates=180,
        entropy_coef=0.006,
        description="seed05 root continued against weakest accepted seed07 stochastic peer",
    ),
    PopulationRun(
        run_id="unipop_seed06_vs_seed03_stoch",
        run_dir="training_runs/universal_population_seed06_vs_seed03_stoch_s20261106",
        resume=U_SEED06,
        opponent_policy_checkpoint=U_SEED03,
        seed=20261106,
        updates=200,
        entropy_coef=0.004,
        description="seed06 V4 root strengthened against strong seed03 stochastic peer",
    ),
    PopulationRun(
        run_id="unipop_seed07_vs_seed05_stoch",
        run_dir="training_runs/universal_population_seed07_vs_seed05_stoch_s20261107",
        resume=U_SEED07,
        opponent_policy_checkpoint=U_SEED05,
        seed=20261107,
        updates=200,
        entropy_coef=0.004,
        description="seed07 V4 root strengthened against strong seed05 stochastic peer",
    ),
)


def command_for_run(run: PopulationRun, args: argparse.Namespace) -> list[str]:
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
        run.opponent_policy_mode,
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


def validate_runs(runs: tuple[PopulationRun, ...]) -> None:
    for run in runs:
        for field_name, checkpoint in (
            ("resume", run.resume),
            ("opponent_policy_checkpoint", run.opponent_policy_checkpoint),
        ):
            path = ROOT / checkpoint
            if not path.exists():
                raise FileNotFoundError(f"{run.run_id} {field_name} missing: {checkpoint}")
            if not checkpoint_is_compatible(path):
                raise ValueError(f"{run.run_id} {field_name} incompatible: {checkpoint}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def build_manifest(batch_dir: Path, runs: tuple[PopulationRun, ...]) -> dict:
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
            "opponent_policy_mode": run.opponent_policy_mode,
            "seed": run.seed,
        })
    manifest = {
        "kind": "role_universal_population_manifest",
        "batch_dir": batch_dir.relative_to(ROOT).as_posix(),
        "entries": entries,
    }
    write_json(batch_dir / "manifest.json", manifest)
    write_json(ROOT / "training_runs" / "self_play_pools" / f"{batch_dir.name}_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the post-gate role-universal frozen self-play population continuation"
    )
    parser.add_argument("--batch-id", default="sprint3_role_universal_population_big_v1")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validate_runs(POPULATION_RUNS)
    batch_dir = ROOT / "training_runs" / "role_universal_population_batches" / args.batch_id
    logs_dir = batch_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "kind": "role_universal_population_plan",
        "batch_id": args.batch_id,
        "args": vars(args),
        "runs": [asdict(run) for run in POPULATION_RUNS],
        "commands": [command_for_run(run, args) for run in POPULATION_RUNS],
    }
    write_json(batch_dir / "plan.json", plan)
    print(json.dumps({"kind": "role_universal_population_plan_written", "path": str(batch_dir / "plan.json")}, sort_keys=True), flush=True)

    if args.dry_run:
        build_manifest(batch_dir, POPULATION_RUNS)
        return 0

    status = {
        "kind": "role_universal_population_status",
        "batch_id": args.batch_id,
        "runs": [],
    }
    for run in POPULATION_RUNS:
        command = command_for_run(run, args)
        log_path = logs_dir / f"{run.run_id}.log"
        print(json.dumps({
            "kind": "role_universal_population_run_started",
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
        print(json.dumps({"kind": "role_universal_population_run_finished", **entry}, sort_keys=True), flush=True)
        if process.returncode != 0:
            build_manifest(batch_dir, POPULATION_RUNS)
            return process.returncode

    manifest = build_manifest(batch_dir, POPULATION_RUNS)
    print(json.dumps({
        "kind": "role_universal_population_batch_finished",
        "batch_id": args.batch_id,
        "manifest_path": str(batch_dir / "manifest.json"),
        "compatible": sum(1 for item in manifest["entries"] if item["compatible"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
