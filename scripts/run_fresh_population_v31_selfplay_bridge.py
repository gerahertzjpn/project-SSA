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


FRESH01 = "training_runs/fresh_population_v31_seed01_scripted_std_s20261201/checkpoints/checkpoint_update_300.pt"
FRESH02 = "training_runs/fresh_population_v31_seed02_scripted_lowent_s20261202/checkpoints/checkpoint_update_320.pt"
FRESH03 = "training_runs/fresh_population_v31_seed03_scripted_highent_s20261203/checkpoints/checkpoint_update_240.pt"
FRESH04 = "training_runs/fresh_population_v31_seed04_scripted_fastlr_s20261204/checkpoints/checkpoint_update_240.pt"
FRESH05 = "training_runs/fresh_population_v31_seed05_scripted_slow_highent_s20261205/checkpoints/checkpoint_update_280.pt"
FRESH06 = "training_runs/fresh_population_v31_seed06_scripted_wide_minibatch_s20261206/checkpoints/checkpoint_update_320.pt"


@dataclass(frozen=True)
class BridgeRun:
    run_id: str
    run_dir: str
    resume: str
    opponent_policy_checkpoint: str
    seed: int
    updates: int
    opponent_policy_mode: str = "stochastic"
    entropy_coef: float = 0.006
    learning_rate: float = 8.0e-5
    description: str = ""


BRIDGE_RUNS: tuple[BridgeRun, ...] = (
    BridgeRun(
        run_id="bridge01_fresh01_vs_fresh06_stoch",
        run_dir="training_runs/fresh_population_v31_bridge_seed01_vs_seed06_s20261501",
        resume=FRESH01,
        opponent_policy_checkpoint=FRESH06,
        seed=20261501,
        updates=140,
        entropy_coef=0.006,
        description="fresh01 universal root continued against strong fresh06 stochastic peer",
    ),
    BridgeRun(
        run_id="bridge02_fresh02_vs_fresh04_stoch",
        run_dir="training_runs/fresh_population_v31_bridge_seed02_vs_seed04_s20261502",
        resume=FRESH02,
        opponent_policy_checkpoint=FRESH04,
        seed=20261502,
        updates=140,
        entropy_coef=0.004,
        description="fresh02 universal root continued against fresh04 stochastic peer",
    ),
    BridgeRun(
        run_id="bridge03_fresh03_vs_fresh06_stoch",
        run_dir="training_runs/fresh_population_v31_bridge_seed03_vs_seed06_s20261503",
        resume=FRESH03,
        opponent_policy_checkpoint=FRESH06,
        seed=20261503,
        updates=180,
        entropy_coef=0.008,
        learning_rate=6.0e-5,
        description="weak fresh03 universal root stabilized against strong fresh06 stochastic peer",
    ),
    BridgeRun(
        run_id="bridge04_fresh04_vs_fresh02_stoch",
        run_dir="training_runs/fresh_population_v31_bridge_seed04_vs_seed02_s20261504",
        resume=FRESH04,
        opponent_policy_checkpoint=FRESH02,
        seed=20261504,
        updates=140,
        entropy_coef=0.005,
        description="fresh04 universal root continued against fresh02 stochastic peer",
    ),
    BridgeRun(
        run_id="bridge05_fresh05_vs_fresh01_stoch",
        run_dir="training_runs/fresh_population_v31_bridge_seed05_vs_seed01_s20261505",
        resume=FRESH05,
        opponent_policy_checkpoint=FRESH01,
        seed=20261505,
        updates=180,
        entropy_coef=0.008,
        learning_rate=6.0e-5,
        description="weak fresh05 universal root stabilized against fresh01 stochastic peer",
    ),
    BridgeRun(
        run_id="bridge06_fresh06_vs_fresh03_stoch",
        run_dir="training_runs/fresh_population_v31_bridge_seed06_vs_seed03_s20261506",
        resume=FRESH06,
        opponent_policy_checkpoint=FRESH03,
        seed=20261506,
        updates=120,
        entropy_coef=0.004,
        description="fresh06 strong root continued against fresh03 stochastic peer",
    ),
)


def command_for_run(run: BridgeRun, args: argparse.Namespace) -> list[str]:
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


def validate_runs(runs: tuple[BridgeRun, ...]) -> None:
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


def build_manifest(batch_dir: Path, runs: tuple[BridgeRun, ...]) -> dict:
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
        "kind": "fresh_population_v31_selfplay_bridge_manifest",
        "batch_dir": batch_dir.relative_to(ROOT).as_posix(),
        "entries": entries,
    }
    write_json(batch_dir / "manifest.json", manifest)
    write_json(ROOT / "training_runs" / "self_play_pools" / f"{batch_dir.name}_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fresh v3.1 universal self-play bridge continuations")
    parser.add_argument("--batch-id", default="fresh_population_v31_selfplay_bridge_v1")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0, help="0-based bridge run index to start from")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validate_runs(BRIDGE_RUNS)
    selected_runs = BRIDGE_RUNS[max(0, int(args.start_index)) :]
    batch_dir = ROOT / "training_runs" / "fresh_population_v31_bridge_batches" / args.batch_id
    logs_dir = batch_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "kind": "fresh_population_v31_selfplay_bridge_plan",
        "batch_id": args.batch_id,
        "args": vars(args),
        "runs": [asdict(run) for run in selected_runs],
        "commands": [command_for_run(run, args) for run in selected_runs],
    }
    write_json(batch_dir / "plan.json", plan)
    print(json.dumps({"kind": "fresh_population_v31_selfplay_bridge_plan_written", "path": str(batch_dir / "plan.json")}, sort_keys=True), flush=True)

    if args.dry_run:
        build_manifest(batch_dir, BRIDGE_RUNS)
        return 0

    status = {
        "kind": "fresh_population_v31_selfplay_bridge_status",
        "batch_id": args.batch_id,
        "runs": [],
    }
    for run in selected_runs:
        command = command_for_run(run, args)
        log_path = logs_dir / f"{run.run_id}.log"
        print(json.dumps({
            "kind": "fresh_population_v31_selfplay_bridge_run_started",
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
        print(json.dumps({"kind": "fresh_population_v31_selfplay_bridge_run_finished", **entry}, sort_keys=True), flush=True)
        if process.returncode != 0:
            build_manifest(batch_dir, BRIDGE_RUNS)
            return int(process.returncode)

    manifest = build_manifest(batch_dir, BRIDGE_RUNS)
    print(json.dumps({
        "kind": "fresh_population_v31_selfplay_bridge_batch_finished",
        "batch_id": args.batch_id,
        "manifest_path": str(batch_dir / "manifest.json"),
        "compatible": sum(1 for item in manifest["entries"] if item["compatible"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
