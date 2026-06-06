from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bots.policy_bot import checkpoint_is_compatible
from src.training.spaces import ACTION_DIM, OBSERVATION_DIM, OBSERVATION_VERSION


SCRIPTED_U240 = "training_runs/sprint3_team_scripted_fresh_v1/checkpoints/checkpoint_update_240.pt"
SCRIPTED_U280 = "training_runs/sprint3_team_scripted_fresh_v1/checkpoints/checkpoint_update_280.pt"
SELFPLAY_U380 = "training_runs/sprint3_team_selfplay_from_u280_vs_u280_v1/checkpoints/checkpoint_update_380.pt"
POP_SEED03_LATEST = "training_runs/population_s3_seed03_scripted_highent_s20260653/checkpoints/latest.pt"


@dataclass(frozen=True)
class SeedRun:
    run_id: str
    run_dir: str
    seed: int
    updates: int
    description: str
    opponent_bot: str = "scripted"
    opponent_policy_checkpoint: str | None = None
    opponent_policy_mode: str = "stochastic"
    resume: str | None = None
    entropy_coef: float = 0.01
    extra_args: tuple[str, ...] = field(default_factory=tuple)


SEED_RUNS: tuple[SeedRun, ...] = (
    SeedRun(
        run_id="seed01_scripted_std",
        run_dir="training_runs/population_s3_seed01_scripted_std_s20260651",
        seed=20260651,
        updates=120,
        description="fresh scripted anchor, standard entropy",
    ),
    SeedRun(
        run_id="seed02_scripted_low_entropy",
        run_dir="training_runs/population_s3_seed02_scripted_lowent_s20260652",
        seed=20260652,
        updates=160,
        entropy_coef=0.005,
        description="fresh scripted anchor, lower entropy",
    ),
    SeedRun(
        run_id="seed03_scripted_high_entropy",
        run_dir="training_runs/population_s3_seed03_scripted_highent_s20260653",
        seed=20260653,
        updates=140,
        entropy_coef=0.02,
        description="fresh scripted anchor, higher entropy",
    ),
    SeedRun(
        run_id="seed04_warmstart_seed03_vs_u280",
        run_dir="training_runs/population_s3_seed04_warmstart_seed03_vs_u280_s20260664",
        seed=20260664,
        updates=80,
        opponent_bot="policy",
        opponent_policy_checkpoint=SCRIPTED_U280,
        opponent_policy_mode="stochastic",
        resume=POP_SEED03_LATEST,
        entropy_coef=0.015,
        description="population seed03 ancestry continued against frozen scripted-u280 stochastic policy",
    ),
    SeedRun(
        run_id="seed05_resume_u280_vs_u380",
        run_dir="training_runs/population_s3_seed05_resume_u280_vs_u380_s20260655",
        seed=20260655,
        updates=120,
        opponent_bot="policy",
        opponent_policy_checkpoint=SELFPLAY_U380,
        opponent_policy_mode="stochastic",
        resume=SCRIPTED_U280,
        description="scripted-u280 ancestry continued against frozen self-play-u380",
    ),
    SeedRun(
        run_id="seed06_resume_u380_vs_u240_argmax",
        run_dir="training_runs/population_s3_seed06_resume_u380_vs_u240_argmax_s20260656",
        seed=20260656,
        updates=100,
        opponent_bot="policy",
        opponent_policy_checkpoint=SCRIPTED_U240,
        opponent_policy_mode="argmax",
        resume=SELFPLAY_U380,
        description="self-play-u380 ancestry hardened against scripted-u240 argmax",
    ),
    SeedRun(
        run_id="seed07_scripted_alt_entropy",
        run_dir="training_runs/population_s3_seed07_scripted_altent_s20260657",
        seed=20260657,
        updates=120,
        entropy_coef=0.015,
        description="backup fresh scripted anchor with alternate seed and medium-high entropy",
    ),
)


def train_command(seed_run: SeedRun, args: argparse.Namespace) -> list[str]:
    updates = int(args.smoke_updates or seed_run.updates)
    command = [
        sys.executable,
        "scripts/train_team_policy.py",
        "--run-dir",
        seed_run.run_dir,
        "--overwrite",
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
        "--opponent-bot",
        seed_run.opponent_bot,
        "--ally-bot",
        "scripted",
        "--seed",
        str(seed_run.seed),
        "--torch-threads",
        str(args.torch_threads),
        "--random-baseline-episodes",
        str(args.random_baseline_episodes),
        "--entropy-coef",
        str(seed_run.entropy_coef),
    ]
    if seed_run.resume:
        command.extend(["--resume", seed_run.resume])
    if seed_run.opponent_bot == "policy":
        command.extend([
            "--opponent-policy-checkpoint",
            str(seed_run.opponent_policy_checkpoint),
            "--opponent-policy-mode",
            seed_run.opponent_policy_mode,
        ])
    command.extend(seed_run.extra_args)
    return command


def scripted_eval_command(seed_run: SeedRun, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/evaluate_team_checkpoints.py",
        "--run-dir",
        seed_run.run_dir,
        "--updates",
        "latest",
        "--modes",
        "argmax,stochastic",
        "--seed-bases",
        args.post_eval_seed_bases,
        "--episodes-per-seed",
        str(args.post_eval_episodes_per_seed),
        "--workers",
        str(args.workers),
        "--torch-threads",
        str(args.torch_threads),
        "--opponent-bot",
        "scripted",
        "--ally-bot",
        "scripted",
        "--out-dir",
        f"training_runs/evals/{Path(seed_run.run_dir).name}_scripted_latest",
    ]


def validate_inputs(selected_runs: list[SeedRun]) -> None:
    for seed_run in selected_runs:
        for checkpoint in (seed_run.resume, seed_run.opponent_policy_checkpoint):
            if checkpoint is None:
                continue
            path = ROOT / checkpoint
            if not path.exists():
                raise FileNotFoundError(f"{seed_run.run_id} references missing checkpoint: {checkpoint}")
            if not checkpoint_is_compatible(path):
                raise ValueError(f"{seed_run.run_id} references incompatible checkpoint: {checkpoint}")


def run_and_tee(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return int(process.wait())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def build_population_manifest(batch_id: str, selected_runs: list[SeedRun], status_records: list[dict]) -> dict:
    entries = []
    for seed_run, status in zip(selected_runs, status_records):
        checkpoint_path = ROOT / seed_run.run_dir / "checkpoints" / "latest.pt"
        entry = {
            "id": seed_run.run_id,
            "description": seed_run.description,
            "run_dir": seed_run.run_dir,
            "seed": seed_run.seed,
            "planned_updates": seed_run.updates,
            "opponent_bot": seed_run.opponent_bot,
            "opponent_policy_checkpoint": seed_run.opponent_policy_checkpoint,
            "opponent_policy_mode": seed_run.opponent_policy_mode,
            "resume": seed_run.resume,
            "entropy_coef": seed_run.entropy_coef,
            "status": status.get("status"),
            "returncode": status.get("returncode"),
            "checkpoint": None,
            "compatible": False,
        }
        if checkpoint_path.exists():
            entry["checkpoint"] = checkpoint_path.relative_to(ROOT).as_posix()
            entry["compatible"] = checkpoint_is_compatible(checkpoint_path)
        entries.append(entry)
    return {
        "kind": "population_seed_manifest",
        "batch_id": batch_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "observation_version": OBSERVATION_VERSION,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "entries": entries,
    }


def selected_seed_runs(args: argparse.Namespace) -> list[SeedRun]:
    start_index = max(0, int(args.start_index))
    max_runs = int(args.max_runs or len(SEED_RUNS))
    return list(SEED_RUNS[start_index:start_index + max_runs])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first Sprint 3 population seed batch")
    parser.add_argument("--batch-id", default="sprint3_population_seed_batch_v1")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means all remaining runs")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--random-baseline-episodes", type=int, default=50)
    parser.add_argument("--post-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-eval-seed-bases", default="20260605,20270605")
    parser.add_argument("--post-eval-episodes-per-seed", type=int, default=25)
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_runs = selected_seed_runs(args)
    validate_inputs(selected_runs)

    batch_root = ROOT / "training_runs" / "population_batches" / args.batch_id
    log_dir = batch_root / "logs"
    status_path = batch_root / "status.json"
    manifest_path = ROOT / "training_runs" / "self_play_pools" / f"{args.batch_id}_manifest.json"
    status_records: list[dict] = []

    plan = {
        "kind": "population_seed_batch_plan",
        "batch_id": args.batch_id,
        "dry_run": bool(args.dry_run),
        "smoke_updates": int(args.smoke_updates),
        "selected_runs": [
            {
                "run_id": seed_run.run_id,
                "description": seed_run.description,
                "train_command": train_command(seed_run, args),
                "scripted_eval_command": scripted_eval_command(seed_run, args) if args.post_eval else None,
            }
            for seed_run in selected_runs
        ],
    }
    write_json(batch_root / "plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True), flush=True)

    if args.dry_run:
        return 0

    for seed_run in selected_runs:
        started_at = time.perf_counter()
        record = {
            "run_id": seed_run.run_id,
            "run_dir": seed_run.run_dir,
            "status": "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        status_records.append(record)
        write_json(status_path, {"batch_id": args.batch_id, "runs": status_records})

        train_rc = run_and_tee(train_command(seed_run, args), log_dir / f"{seed_run.run_id}_train.log")
        record["returncode"] = train_rc
        record["elapsed_sec"] = round(time.perf_counter() - started_at, 3)
        record["status"] = "trained" if train_rc == 0 else "failed_train"
        write_json(status_path, {"batch_id": args.batch_id, "runs": status_records})
        if train_rc != 0:
            break

        if args.post_eval:
            eval_started = time.perf_counter()
            eval_rc = run_and_tee(scripted_eval_command(seed_run, args), log_dir / f"{seed_run.run_id}_scripted_eval.log")
            record["scripted_eval_returncode"] = eval_rc
            record["scripted_eval_elapsed_sec"] = round(time.perf_counter() - eval_started, 3)
            record["status"] = "complete" if eval_rc == 0 else "failed_scripted_eval"
            write_json(status_path, {"batch_id": args.batch_id, "runs": status_records})
            if eval_rc != 0:
                break

        manifest = build_population_manifest(args.batch_id, selected_runs, status_records)
        write_json(manifest_path, manifest)

    manifest = build_population_manifest(args.batch_id, selected_runs, status_records)
    write_json(manifest_path, manifest)
    print(json.dumps({
        "kind": "population_seed_batch_finished",
        "batch_id": args.batch_id,
        "status_path": str(status_path.relative_to(ROOT)),
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "completed": sum(1 for record in status_records if record.get("status") == "complete"),
        "runs": len(status_records),
    }, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if all(record.get("status") == "complete" for record in status_records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
