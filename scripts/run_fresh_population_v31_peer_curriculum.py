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


BRIDGE01 = "training_runs/fresh_population_v31_bridge_seed01_vs_seed06_s20261501/checkpoints/checkpoint_update_440.pt"
BRIDGE02 = "training_runs/fresh_population_v31_bridge_seed02_vs_seed04_s20261502/checkpoints/checkpoint_update_460.pt"
BRIDGE03 = "training_runs/fresh_population_v31_bridge_seed03_vs_seed06_s20261503/checkpoints/checkpoint_update_400.pt"
BRIDGE04 = "training_runs/fresh_population_v31_bridge_seed04_vs_seed02_s20261504/checkpoints/checkpoint_update_380.pt"
BRIDGE05 = "training_runs/fresh_population_v31_bridge_seed05_vs_seed01_s20261505/checkpoints/checkpoint_update_420.pt"
BRIDGE06 = "training_runs/fresh_population_v31_bridge_seed06_vs_seed03_s20261506/checkpoints/checkpoint_update_440.pt"


@dataclass(frozen=True)
class PeerRun:
    run_id: str
    run_dir: str
    resume: str
    opponent_policy_checkpoint: str
    seed: int
    updates: int
    opponent_policy_mode: str = "stochastic"
    entropy_coef: float = 0.006
    learning_rate: float = 6.0e-5
    description: str = ""


PEER_RUNS: tuple[PeerRun, ...] = (
    PeerRun(
        run_id="peer01_bridge01_vs_bridge03",
        run_dir="training_runs/fresh_population_v31_peer_seed01_vs_bridge03_s20262001",
        resume=BRIDGE01,
        opponent_policy_checkpoint=BRIDGE03,
        seed=20262001,
        updates=220,
        entropy_coef=0.007,
        description="bridge01 continued against strong bridge03 stochastic peer",
    ),
    PeerRun(
        run_id="peer02_bridge02_vs_bridge03",
        run_dir="training_runs/fresh_population_v31_peer_seed02_vs_bridge03_s20262002",
        resume=BRIDGE02,
        opponent_policy_checkpoint=BRIDGE03,
        seed=20262002,
        updates=240,
        entropy_coef=0.006,
        description="bridge02 continued against strong bridge03 stochastic peer",
    ),
    PeerRun(
        run_id="peer03_bridge03_vs_bridge06",
        run_dir="training_runs/fresh_population_v31_peer_seed03_vs_bridge06_s20262003",
        resume=BRIDGE03,
        opponent_policy_checkpoint=BRIDGE06,
        seed=20262003,
        updates=180,
        entropy_coef=0.006,
        description="bridge03 continued against bridge06 stochastic peer",
    ),
    PeerRun(
        run_id="peer04_bridge04_vs_bridge03",
        run_dir="training_runs/fresh_population_v31_peer_seed04_vs_bridge03_s20262004",
        resume=BRIDGE04,
        opponent_policy_checkpoint=BRIDGE03,
        seed=20262004,
        updates=220,
        entropy_coef=0.007,
        description="bridge04 continued against strong bridge03 stochastic peer",
    ),
    PeerRun(
        run_id="peer05_bridge05_vs_bridge03",
        run_dir="training_runs/fresh_population_v31_peer_seed05_vs_bridge03_s20262005",
        resume=BRIDGE05,
        opponent_policy_checkpoint=BRIDGE03,
        seed=20262005,
        updates=240,
        entropy_coef=0.007,
        description="bridge05 continued against strong bridge03 stochastic peer",
    ),
    PeerRun(
        run_id="peer06_bridge06_vs_bridge02",
        run_dir="training_runs/fresh_population_v31_peer_seed06_vs_bridge02_s20262006",
        resume=BRIDGE06,
        opponent_policy_checkpoint=BRIDGE02,
        seed=20262006,
        updates=180,
        entropy_coef=0.005,
        description="bridge06 continued against bridge02 stochastic peer for pool diversity",
    ),
)


def command_for_run(run: PeerRun, args: argparse.Namespace) -> list[str]:
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


def validate_runs(runs: tuple[PeerRun, ...]) -> None:
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


def build_manifest(batch_dir: Path, runs: tuple[PeerRun, ...]) -> dict:
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
            "planned_updates": run.updates,
        })
    manifest = {
        "kind": "fresh_population_v31_peer_curriculum_manifest",
        "batch_dir": batch_dir.relative_to(ROOT).as_posix(),
        "entries": entries,
    }
    write_json(batch_dir / "manifest.json", manifest)
    write_json(ROOT / "training_runs" / "self_play_pools" / f"{batch_dir.name}_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fresh v3.1 universal trained-peer curriculum")
    parser.add_argument("--batch-id", default="fresh_population_v31_peer_curriculum_v1")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0, help="0-based peer run index to start from")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validate_runs(PEER_RUNS)
    selected_runs = PEER_RUNS[max(0, int(args.start_index)) :]
    batch_dir = ROOT / "training_runs" / "fresh_population_v31_peer_batches" / args.batch_id
    logs_dir = batch_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "kind": "fresh_population_v31_peer_curriculum_plan",
        "batch_id": args.batch_id,
        "args": vars(args),
        "runs": [asdict(run) for run in selected_runs],
        "commands": [command_for_run(run, args) for run in selected_runs],
    }
    write_json(batch_dir / "plan.json", plan)
    print(json.dumps({"kind": "fresh_population_v31_peer_curriculum_plan_written", "path": str(batch_dir / "plan.json")}, sort_keys=True), flush=True)

    if args.dry_run:
        build_manifest(batch_dir, PEER_RUNS)
        return 0

    status = {
        "kind": "fresh_population_v31_peer_curriculum_status",
        "batch_id": args.batch_id,
        "runs": [],
    }
    for run in selected_runs:
        command = command_for_run(run, args)
        log_path = logs_dir / f"{run.run_id}.log"
        print(json.dumps({
            "kind": "fresh_population_v31_peer_curriculum_run_started",
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
        print(json.dumps({"kind": "fresh_population_v31_peer_curriculum_run_finished", **entry}, sort_keys=True), flush=True)
        if process.returncode != 0:
            build_manifest(batch_dir, PEER_RUNS)
            return int(process.returncode)

    manifest = build_manifest(batch_dir, PEER_RUNS)
    print(json.dumps({
        "kind": "fresh_population_v31_peer_curriculum_batch_finished",
        "batch_id": args.batch_id,
        "manifest_path": str(batch_dir / "manifest.json"),
        "compatible": sum(1 for item in manifest["entries"] if item["compatible"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
