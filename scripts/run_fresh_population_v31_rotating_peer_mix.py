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


PEER01 = "training_runs/fresh_population_v31_peer_seed01_vs_bridge03_s20262001/checkpoints/checkpoint_update_660.pt"
PEER02 = "training_runs/fresh_population_v31_peer_seed02_vs_bridge03_s20262002/checkpoints/checkpoint_update_700.pt"
PEER03 = "training_runs/fresh_population_v31_peer_seed03_vs_bridge06_s20262003/checkpoints/checkpoint_update_580.pt"
PEER04 = "training_runs/fresh_population_v31_peer_seed04_vs_bridge03_s20262004/checkpoints/checkpoint_update_600.pt"
PEER05 = "training_runs/fresh_population_v31_peer_seed05_vs_bridge03_s20262005/checkpoints/checkpoint_update_660.pt"
PEER06 = "training_runs/fresh_population_v31_peer_seed06_vs_bridge02_s20262006/checkpoints/checkpoint_update_620.pt"


@dataclass(frozen=True)
class MixLine:
    line_id: str
    seed: int
    resume: str
    stage1_opponent: str
    stage1_updates: int
    stage2_opponent_line: str
    stage2_updates: int
    entropy_coef: float
    learning_rate: float
    description: str


MIX_LINES: tuple[MixLine, ...] = (
    MixLine(
        line_id="mix01_peer01",
        seed=20263001,
        resume=PEER01,
        stage1_opponent=PEER05,
        stage1_updates=120,
        stage2_opponent_line="mix02_peer02",
        stage2_updates=80,
        entropy_coef=0.006,
        learning_rate=5.0e-5,
        description="strong peer01 continued against peer05, then rotated against updated peer02",
    ),
    MixLine(
        line_id="mix02_peer02",
        seed=20263002,
        resume=PEER02,
        stage1_opponent=PEER01,
        stage1_updates=120,
        stage2_opponent_line="mix05_peer05",
        stage2_updates=80,
        entropy_coef=0.006,
        learning_rate=5.0e-5,
        description="peer02 continued against hard peer01, then rotated against updated peer05",
    ),
    MixLine(
        line_id="mix03_peer03",
        seed=20263003,
        resume=PEER03,
        stage1_opponent=PEER01,
        stage1_updates=120,
        stage2_opponent_line="mix04_peer04",
        stage2_updates=80,
        entropy_coef=0.008,
        learning_rate=5.0e-5,
        description="weak-vs-peer01 peer03 continued against hard peer01, then updated peer04",
    ),
    MixLine(
        line_id="mix04_peer04",
        seed=20263004,
        resume=PEER04,
        stage1_opponent=PEER01,
        stage1_updates=120,
        stage2_opponent_line="mix06_peer06",
        stage2_updates=80,
        entropy_coef=0.006,
        learning_rate=5.0e-5,
        description="peer04 continued against hard peer01, then rotated against updated peer06",
    ),
    MixLine(
        line_id="mix05_peer05",
        seed=20263005,
        resume=PEER05,
        stage1_opponent=PEER01,
        stage1_updates=120,
        stage2_opponent_line="mix01_peer01",
        stage2_updates=80,
        entropy_coef=0.006,
        learning_rate=5.0e-5,
        description="best stochastic challenger peer05 continued against hard peer01, then updated peer01",
    ),
    MixLine(
        line_id="mix06_peer06",
        seed=20263006,
        resume=PEER06,
        stage1_opponent=PEER01,
        stage1_updates=120,
        stage2_opponent_line="mix03_peer03",
        stage2_updates=80,
        entropy_coef=0.008,
        learning_rate=5.0e-5,
        description="weak-vs-peer01 peer06 continued against hard peer01, then updated peer03",
    ),
)


def stage1_run_dir(line: MixLine) -> str:
    return f"training_runs/fresh_population_v31_rotmix_{line.line_id}_stage1_s{line.seed}"


def stage2_run_dir(line: MixLine) -> str:
    return f"training_runs/fresh_population_v31_rotmix_{line.line_id}_stage2_s{line.seed + 100}"


def latest_checkpoint(run_dir: str) -> str:
    return f"{run_dir}/checkpoints/latest.pt"


def checkpoint_update_path(run_dir: str, update: int) -> str:
    return f"{run_dir}/checkpoints/checkpoint_update_{int(update)}.pt"


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


def validate_base_lines(lines: tuple[MixLine, ...]) -> None:
    for line in lines:
        validate_checkpoint(f"{line.line_id} resume", line.resume)
        validate_checkpoint(f"{line.line_id} stage1_opponent", line.stage1_opponent)
    line_ids = {line.line_id for line in lines}
    for line in lines:
        if line.stage2_opponent_line not in line_ids:
            raise ValueError(f"{line.line_id} unknown stage2 opponent line: {line.stage2_opponent_line}")


def train_command(
    *,
    run_dir: str,
    resume: str,
    opponent: str,
    updates: int,
    seed: int,
    entropy_coef: float,
    learning_rate: float,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        "scripts/train_team_policy.py",
        "--run-dir",
        run_dir,
        "--overwrite",
        "--resume",
        resume,
        "--updates",
        str(int(args.smoke_updates or updates)),
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
        str(seed),
        "--opponent-bot",
        "policy",
        "--opponent-policy-checkpoint",
        opponent,
        "--opponent-policy-mode",
        "stochastic",
        "--entropy-coef",
        str(entropy_coef),
        "--learning-rate",
        str(learning_rate),
        "--torch-threads",
        str(args.torch_threads),
        "--role-assignment-mode",
        "shuffle",
        "--random-baseline-episodes",
        "0",
    ]


def planned_runs(args: argparse.Namespace) -> list[dict]:
    stage1 = []
    stage2 = []
    by_id = {line.line_id: line for line in MIX_LINES}
    for line in MIX_LINES:
        run_dir = stage1_run_dir(line)
        stage1.append({
            "stage": 1,
            "line_id": line.line_id,
            "run_id": f"{line.line_id}_stage1",
            "run_dir": run_dir,
            "resume": line.resume,
            "opponent_policy_checkpoint": line.stage1_opponent,
            "seed": line.seed,
            "planned_updates": int(args.smoke_updates or line.stage1_updates),
            "description": line.description,
            "command": train_command(
                run_dir=run_dir,
                resume=line.resume,
                opponent=line.stage1_opponent,
                updates=line.stage1_updates,
                seed=line.seed,
                entropy_coef=line.entropy_coef,
                learning_rate=line.learning_rate,
                args=args,
            ),
        })
    for line in MIX_LINES:
        opponent_line = by_id[line.stage2_opponent_line]
        run_dir = stage2_run_dir(line)
        resume = latest_checkpoint(stage1_run_dir(line))
        opponent = latest_checkpoint(stage1_run_dir(opponent_line))
        stage2.append({
            "stage": 2,
            "line_id": line.line_id,
            "run_id": f"{line.line_id}_stage2",
            "run_dir": run_dir,
            "resume": resume,
            "opponent_policy_checkpoint": opponent,
            "seed": line.seed + 100,
            "planned_updates": int(args.smoke_updates or line.stage2_updates),
            "description": f"{line.line_id} rotated against updated {opponent_line.line_id}",
            "command": train_command(
                run_dir=run_dir,
                resume=resume,
                opponent=opponent,
                updates=line.stage2_updates,
                seed=line.seed + 100,
                entropy_coef=line.entropy_coef,
                learning_rate=line.learning_rate,
                args=args,
            ),
        })
    return stage1 + stage2


def build_manifest(batch_dir: Path) -> dict:
    entries = []
    for line in MIX_LINES:
        stage1_latest = ROOT / latest_checkpoint(stage1_run_dir(line))
        stage2_latest = ROOT / latest_checkpoint(stage2_run_dir(line))
        final = stage2_latest if stage2_latest.exists() else stage1_latest
        entries.append({
            "line_id": line.line_id,
            "description": line.description,
            "stage1_run_dir": stage1_run_dir(line),
            "stage2_run_dir": stage2_run_dir(line),
            "checkpoint": final.relative_to(ROOT).as_posix() if final.exists() else None,
            "compatible": checkpoint_is_compatible(final) if final.exists() else False,
            "role_assignment_mode": "shuffle",
            "seed": line.seed,
            "stage1_opponent": line.stage1_opponent,
            "stage2_opponent_line": line.stage2_opponent_line,
        })
    manifest = {
        "kind": "fresh_population_v31_rotating_peer_mix_manifest",
        "batch_dir": batch_dir.relative_to(ROOT).as_posix(),
        "entries": entries,
    }
    write_json(batch_dir / "manifest.json", manifest)
    write_json(ROOT / "training_runs" / "self_play_pools" / f"{batch_dir.name}_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v3.1 rotating trained-peer mix continuations")
    parser.add_argument("--batch-id", default="fresh_population_v31_rotating_peer_mix_v1")
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

    validate_base_lines(MIX_LINES)
    batch_dir = ROOT / "training_runs" / "fresh_population_v31_rotmix_batches" / args.batch_id
    logs_dir = batch_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    runs = planned_runs(args)
    plan = {
        "kind": "fresh_population_v31_rotating_peer_mix_plan",
        "batch_id": args.batch_id,
        "args": vars(args),
        "lines": [asdict(line) for line in MIX_LINES],
        "runs": runs,
    }
    write_json(batch_dir / "plan.json", plan)
    print(json.dumps({"kind": "fresh_population_v31_rotating_peer_mix_plan_written", "path": str(batch_dir / "plan.json")}, sort_keys=True), flush=True)

    if args.dry_run:
        build_manifest(batch_dir)
        return 0

    status = {
        "kind": "fresh_population_v31_rotating_peer_mix_status",
        "batch_id": args.batch_id,
        "runs": [],
    }
    for index, run in enumerate(runs):
        if index < max(0, int(args.start_index)):
            continue
        validate_checkpoint(f"{run['run_id']} resume", run["resume"])
        validate_checkpoint(f"{run['run_id']} opponent", run["opponent_policy_checkpoint"])
        command = run["command"]
        log_path = logs_dir / f"{run['run_id']}.log"
        print(json.dumps({
            "kind": "fresh_population_v31_rotating_peer_mix_run_started",
            "run_id": run["run_id"],
            "stage": run["stage"],
            "line_id": run["line_id"],
            "run_dir": run["run_dir"],
            "resume": run["resume"],
            "opponent_policy_checkpoint": run["opponent_policy_checkpoint"],
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
            "run_id": run["run_id"],
            "stage": run["stage"],
            "line_id": run["line_id"],
            "run_dir": run["run_dir"],
            "returncode": process.returncode,
            "log_path": log_path.relative_to(ROOT).as_posix(),
        }
        status["runs"].append(entry)
        write_json(batch_dir / "status.json", status)
        print(json.dumps({"kind": "fresh_population_v31_rotating_peer_mix_run_finished", **entry}, sort_keys=True), flush=True)
        if process.returncode != 0:
            build_manifest(batch_dir)
            return int(process.returncode)

    manifest = build_manifest(batch_dir)
    print(json.dumps({
        "kind": "fresh_population_v31_rotating_peer_mix_batch_finished",
        "batch_id": args.batch_id,
        "manifest_path": str(batch_dir / "manifest.json"),
        "compatible": sum(1 for item in manifest["entries"] if item["compatible"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
