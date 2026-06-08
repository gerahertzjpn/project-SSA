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


MIX01 = "training_runs/fresh_population_v31_rotmix_mix01_peer01_stage2_s20263101/checkpoints/checkpoint_update_840.pt"
MIX02 = "training_runs/fresh_population_v31_rotmix_mix02_peer02_stage2_s20263102/checkpoints/checkpoint_update_900.pt"
MIX03 = "training_runs/fresh_population_v31_rotmix_mix03_peer03_stage2_s20263103/checkpoints/checkpoint_update_780.pt"
MIX04 = "training_runs/fresh_population_v31_rotmix_mix04_peer04_stage1_s20263004/checkpoints/checkpoint_update_720.pt"
MIX05 = "training_runs/fresh_population_v31_rotmix_mix05_peer05_stage1_s20263005/checkpoints/checkpoint_update_760.pt"
MIX06 = "training_runs/fresh_population_v31_rotmix_mix06_peer06_stage2_s20263106/checkpoints/checkpoint_update_820.pt"

PEER01 = "training_runs/fresh_population_v31_peer_seed01_vs_bridge03_s20262001/checkpoints/checkpoint_update_660.pt"


@dataclass(frozen=True)
class SlotSource:
    target_player: str
    checkpoint: str
    source_player: str


@dataclass(frozen=True)
class TeammateMixRun:
    run_id: str
    run_dir: str
    init_checkpoint: str
    opponent_policy_checkpoint: str
    seed: int
    updates: int
    slots: tuple[SlotSource, SlotSource, SlotSource]
    entropy_coef: float = 0.008
    learning_rate: float = 5.0e-5
    description: str = ""


MIX_RUNS: tuple[TeammateMixRun, ...] = (
    TeammateMixRun(
        run_id="tmix01_worst_456",
        run_dir="training_runs/fresh_population_v31_teammate_mix_tmix01_worst456_s20264001",
        init_checkpoint="training_runs/fresh_population_v31_teammate_mix_inits/tmix01_worst456.pt",
        opponent_policy_checkpoint=PEER01,
        seed=20264001,
        updates=100,
        slots=(
            SlotSource("A1", MIX04, "A1"),
            SlotSource("A2", MIX05, "A2"),
            SlotSource("A3", MIX06, "A3"),
        ),
        description="directly adapts the worst observed mix04+mix05+mix06 slot-mixed team",
    ),
    TeammateMixRun(
        run_id="tmix02_worst_465",
        run_dir="training_runs/fresh_population_v31_teammate_mix_tmix02_worst465_s20264002",
        init_checkpoint="training_runs/fresh_population_v31_teammate_mix_inits/tmix02_worst465.pt",
        opponent_policy_checkpoint=PEER01,
        seed=20264002,
        updates=100,
        slots=(
            SlotSource("A1", MIX04, "A1"),
            SlotSource("A2", MIX06, "A2"),
            SlotSource("A3", MIX05, "A3"),
        ),
        description="adapts the alternate worst mix04+mix06+mix05 team",
    ),
    TeammateMixRun(
        run_id="tmix03_peer_pressure_132",
        run_dir="training_runs/fresh_population_v31_teammate_mix_tmix03_132_s20264003",
        init_checkpoint="training_runs/fresh_population_v31_teammate_mix_inits/tmix03_132.pt",
        opponent_policy_checkpoint=PEER01,
        seed=20264003,
        updates=80,
        slots=(
            SlotSource("A1", MIX01, "A1"),
            SlotSource("A2", MIX03, "A2"),
            SlotSource("A3", MIX02, "A3"),
        ),
        description="mixes strong peer01 descendant with weaker mix03 and stable mix02",
    ),
    TeammateMixRun(
        run_id="tmix04_peer_pressure_354",
        run_dir="training_runs/fresh_population_v31_teammate_mix_tmix04_354_s20264004",
        init_checkpoint="training_runs/fresh_population_v31_teammate_mix_inits/tmix04_354.pt",
        opponent_policy_checkpoint=PEER01,
        seed=20264004,
        updates=80,
        slots=(
            SlotSource("A1", MIX03, "A1"),
            SlotSource("A2", MIX05, "A2"),
            SlotSource("A3", MIX04, "A3"),
        ),
        description="cross-trains the weak mix03 with mix05/mix04 teammates",
    ),
    TeammateMixRun(
        run_id="tmix05_peer_pressure_621",
        run_dir="training_runs/fresh_population_v31_teammate_mix_tmix05_621_s20264005",
        init_checkpoint="training_runs/fresh_population_v31_teammate_mix_inits/tmix05_621.pt",
        opponent_policy_checkpoint=PEER01,
        seed=20264005,
        updates=80,
        slots=(
            SlotSource("A1", MIX06, "A1"),
            SlotSource("A2", MIX02, "A2"),
            SlotSource("A3", MIX01, "A3"),
        ),
        entropy_coef=0.009,
        description="cross-trains weak mix06 with stronger mix02/mix01 teammates",
    ),
    TeammateMixRun(
        run_id="tmix06_peer_pressure_243",
        run_dir="training_runs/fresh_population_v31_teammate_mix_tmix06_243_s20264006",
        init_checkpoint="training_runs/fresh_population_v31_teammate_mix_inits/tmix06_243.pt",
        opponent_policy_checkpoint=PEER01,
        seed=20264006,
        updates=80,
        slots=(
            SlotSource("A1", MIX02, "A1"),
            SlotSource("A2", MIX04, "A2"),
            SlotSource("A3", MIX03, "A3"),
        ),
        description="balances stable mix02 with mix04/mix03 teammates",
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


def create_init_command(run: TeammateMixRun) -> list[str]:
    command = [
        sys.executable,
        "scripts/create_mixed_team_checkpoint.py",
        "--output",
        run.init_checkpoint,
        "--label",
        run.run_id,
    ]
    for slot in run.slots:
        command.extend([
            "--slot",
            f"{slot.target_player}={slot.checkpoint}:{slot.source_player}",
        ])
    return command


def train_command(run: TeammateMixRun, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/train_team_policy.py",
        "--run-dir",
        run.run_dir,
        "--overwrite",
        "--resume",
        run.init_checkpoint,
        "--updates",
        str(int(args.smoke_updates or run.updates)),
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


def validate_runs(runs: tuple[TeammateMixRun, ...]) -> None:
    for run in runs:
        validate_checkpoint(f"{run.run_id} opponent", run.opponent_policy_checkpoint)
        for slot in run.slots:
            validate_checkpoint(f"{run.run_id} {slot.target_player}", slot.checkpoint)


def planned_runs(args: argparse.Namespace) -> list[dict]:
    result = []
    for run in MIX_RUNS:
        result.append({
            "run": asdict(run),
            "create_init_command": create_init_command(run),
            "train_command": train_command(run, args),
            "planned_updates": int(args.smoke_updates or run.updates),
        })
    return result


def build_manifest(batch_dir: Path) -> dict:
    entries = []
    for run in MIX_RUNS:
        checkpoint = ROOT / run.run_dir / "checkpoints" / "latest.pt"
        entries.append({
            "run_id": run.run_id,
            "description": run.description,
            "run_dir": run.run_dir,
            "checkpoint": checkpoint.relative_to(ROOT).as_posix() if checkpoint.exists() else None,
            "compatible": checkpoint_is_compatible(checkpoint) if checkpoint.exists() else False,
            "role_assignment_mode": "shuffle",
            "init_checkpoint": run.init_checkpoint,
            "opponent_policy_checkpoint": run.opponent_policy_checkpoint,
            "seed": run.seed,
            "planned_updates": run.updates,
            "slots": [asdict(slot) for slot in run.slots],
        })
    manifest = {
        "kind": "fresh_population_v31_teammate_mix_adapt_manifest",
        "batch_dir": batch_dir.relative_to(ROOT).as_posix(),
        "entries": entries,
    }
    write_json(batch_dir / "manifest.json", manifest)
    write_json(ROOT / "training_runs" / "self_play_pools" / f"{batch_dir.name}_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run targeted v3.1 teammate-mix adaptation")
    parser.add_argument("--batch-id", default="fresh_population_v31_teammate_mix_adapt_v1")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0, help="0-based planned run index to start from")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validate_runs(MIX_RUNS)
    batch_dir = ROOT / "training_runs" / "fresh_population_v31_teammate_mix_batches" / args.batch_id
    logs_dir = batch_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "kind": "fresh_population_v31_teammate_mix_adapt_plan",
        "batch_id": args.batch_id,
        "args": vars(args),
        "runs": planned_runs(args),
    }
    write_json(batch_dir / "plan.json", plan)
    print(json.dumps({"kind": "fresh_population_v31_teammate_mix_adapt_plan_written", "path": str(batch_dir / "plan.json")}, sort_keys=True), flush=True)

    if args.dry_run:
        build_manifest(batch_dir)
        return 0

    status = {
        "kind": "fresh_population_v31_teammate_mix_adapt_status",
        "batch_id": args.batch_id,
        "runs": [],
    }
    for index, run in enumerate(MIX_RUNS):
        if index < max(0, int(args.start_index)):
            continue
        init_log_path = logs_dir / f"{run.run_id}_init.log"
        with init_log_path.open("w", encoding="utf-8") as log:
            init_process = subprocess.run(
                create_init_command(run),
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if init_process.returncode != 0:
            status["runs"].append({
                "run_id": run.run_id,
                "returncode": init_process.returncode,
                "stage": "init",
                "log_path": init_log_path.relative_to(ROOT).as_posix(),
            })
            write_json(batch_dir / "status.json", status)
            build_manifest(batch_dir)
            return int(init_process.returncode)

        validate_checkpoint(f"{run.run_id} init", run.init_checkpoint)
        train_log_path = logs_dir / f"{run.run_id}.log"
        print(json.dumps({
            "kind": "fresh_population_v31_teammate_mix_adapt_run_started",
            "run_id": run.run_id,
            "run_dir": run.run_dir,
            "init_checkpoint": run.init_checkpoint,
            "opponent_policy_checkpoint": run.opponent_policy_checkpoint,
            "log_path": str(train_log_path),
        }, sort_keys=True), flush=True)
        with train_log_path.open("w", encoding="utf-8") as log:
            train_process = subprocess.run(
                train_command(run, args),
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        entry = {
            "run_id": run.run_id,
            "run_dir": run.run_dir,
            "returncode": train_process.returncode,
            "init_log_path": init_log_path.relative_to(ROOT).as_posix(),
            "log_path": train_log_path.relative_to(ROOT).as_posix(),
        }
        status["runs"].append(entry)
        write_json(batch_dir / "status.json", status)
        print(json.dumps({"kind": "fresh_population_v31_teammate_mix_adapt_run_finished", **entry}, sort_keys=True), flush=True)
        if train_process.returncode != 0:
            build_manifest(batch_dir)
            return int(train_process.returncode)

    manifest = build_manifest(batch_dir)
    print(json.dumps({
        "kind": "fresh_population_v31_teammate_mix_adapt_batch_finished",
        "batch_id": args.batch_id,
        "manifest_path": str(batch_dir / "manifest.json"),
        "compatible": sum(1 for item in manifest["entries"] if item["compatible"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
