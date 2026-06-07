from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bots.policy_bot import checkpoint_is_compatible
from src.training.spaces import ACTION_DIM, OBSERVATION_DIM, OBSERVATION_VERSION


@dataclass(frozen=True)
class FreshSeedRun:
    run_id: str
    run_dir: str
    seed: int
    opponent_bot: str
    entropy_coef: float
    learning_rate: float
    updates: int
    description: str
    ppo_epochs: int = 4
    minibatch_size: int = 2048


FRESH_RUNS: tuple[FreshSeedRun, ...] = (
    FreshSeedRun(
        run_id="fresh01_scripted_std",
        run_dir="training_runs/fresh_population_v31_seed01_scripted_std_s20261201",
        seed=20261201,
        opponent_bot="scripted",
        entropy_coef=0.010,
        learning_rate=1.0e-4,
        updates=300,
        description="fresh root, scripted opponent, standard entropy",
    ),
    FreshSeedRun(
        run_id="fresh02_scripted_lowent",
        run_dir="training_runs/fresh_population_v31_seed02_scripted_lowent_s20261202",
        seed=20261202,
        opponent_bot="scripted",
        entropy_coef=0.005,
        learning_rate=8.0e-5,
        updates=320,
        description="fresh root, scripted opponent, lower entropy and lower LR",
    ),
    FreshSeedRun(
        run_id="fresh03_scripted_highent",
        run_dir="training_runs/fresh_population_v31_seed03_scripted_highent_s20261203",
        seed=20261203,
        opponent_bot="scripted",
        entropy_coef=0.020,
        learning_rate=1.0e-4,
        updates=320,
        description="fresh root, scripted opponent, higher entropy for policy diversity",
    ),
    FreshSeedRun(
        run_id="fresh04_scripted_fastlr",
        run_dir="training_runs/fresh_population_v31_seed04_scripted_fastlr_s20261204",
        seed=20261204,
        opponent_bot="scripted",
        entropy_coef=0.012,
        learning_rate=1.2e-4,
        updates=320,
        description="fresh root, scripted opponent, slightly faster LR",
    ),
    FreshSeedRun(
        run_id="fresh05_scripted_slow_highent",
        run_dir="training_runs/fresh_population_v31_seed05_scripted_slow_highent_s20261205",
        seed=20261205,
        opponent_bot="scripted",
        entropy_coef=0.018,
        learning_rate=7.0e-5,
        updates=340,
        description="fresh root, scripted opponent, slower LR with higher entropy",
    ),
    FreshSeedRun(
        run_id="fresh06_scripted_wide_minibatch",
        run_dir="training_runs/fresh_population_v31_seed06_scripted_wide_minibatch_s20261206",
        seed=20261206,
        opponent_bot="scripted",
        entropy_coef=0.012,
        learning_rate=9.0e-5,
        updates=320,
        description="fresh root, scripted opponent, alternate minibatch/optimization cadence",
        ppo_epochs=5,
        minibatch_size=1024,
    ),
)


def train_command(run: FreshSeedRun, args: argparse.Namespace) -> list[str]:
    updates = int(args.smoke_updates or run.updates)
    return [
        sys.executable,
        "scripts/train_team_policy.py",
        "--run-dir",
        run.run_dir,
        "--overwrite",
        "--updates",
        str(updates),
        "--workers",
        str(args.workers),
        "--rollout-steps",
        str(args.rollout_steps),
        "--ppo-epochs",
        str(run.ppo_epochs),
        "--minibatch-size",
        str(run.minibatch_size),
        "--eval-episodes",
        str(args.eval_episodes),
        "--eval-interval",
        str(args.eval_interval),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--opponent-bot",
        run.opponent_bot,
        "--ally-bot",
        "scripted",
        "--seed",
        str(run.seed),
        "--torch-threads",
        str(args.torch_threads),
        "--random-baseline-episodes",
        str(args.random_baseline_episodes),
        "--entropy-coef",
        str(run.entropy_coef),
        "--learning-rate",
        str(run.learning_rate),
        "--role-assignment-mode",
        "shuffle",
    ]


def eval_command(run: FreshSeedRun, args: argparse.Namespace, opponent_bot: str) -> list[str]:
    out_name = f"{Path(run.run_dir).name}_{opponent_bot}_latest"
    return [
        sys.executable,
        "scripts/evaluate_team_checkpoints.py",
        "--run-dir",
        run.run_dir,
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
        opponent_bot,
        "--ally-bot",
        "scripted",
        "--out-dir",
        f"training_runs/evals/{out_name}",
    ]


def selected_runs(args: argparse.Namespace) -> list[FreshSeedRun]:
    start = max(0, int(args.start_index))
    count = int(args.max_runs or len(FRESH_RUNS))
    return list(FRESH_RUNS[start : start + count])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def run_and_log(command: list[str], log_path: Path, tee: bool) -> int:
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
            log.write(line)
            log.flush()
            if tee:
                print(line, end="", flush=True)
        return int(process.wait())


def latest_checkpoint(run: FreshSeedRun) -> Path:
    return ROOT / run.run_dir / "checkpoints" / "latest.pt"


def build_manifest(batch_dir: Path, runs: list[FreshSeedRun], statuses: list[dict]) -> dict:
    by_id = {status.get("run_id"): status for status in statuses}
    entries = []
    for run in runs:
        checkpoint = latest_checkpoint(run)
        entries.append(
            {
                **asdict(run),
                "checkpoint": checkpoint.relative_to(ROOT).as_posix() if checkpoint.exists() else None,
                "compatible": checkpoint_is_compatible(checkpoint) if checkpoint.exists() else False,
                "role_assignment_mode": "shuffle",
                "status": by_id.get(run.run_id, {}).get("status", "pending"),
            }
        )
    manifest = {
        "kind": "fresh_population_v31_manifest",
        "batch_dir": batch_dir.relative_to(ROOT).as_posix(),
        "observation_version": OBSERVATION_VERSION,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "entries": entries,
    }
    write_json(batch_dir / "manifest.json", manifest)
    write_json(ROOT / "training_runs" / "self_play_pools" / f"{batch_dir.name}_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run six fresh v3.1 population roots from scratch")
    parser.add_argument("--batch-id", default="fresh_population_v31_roots_v1")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means all selected fresh runs")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--random-baseline-episodes", type=int, default=50)
    parser.add_argument("--post-eval-seed-bases", default="20261221,20271221,20281221,20291221")
    parser.add_argument("--post-eval-episodes-per-seed", type=int, default=50)
    parser.add_argument("--post-eval-opponents", default="scripted")
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tee", action="store_true")
    args = parser.parse_args()

    runs = selected_runs(args)
    batch_dir = ROOT / "training_runs" / "fresh_population_batches" / args.batch_id
    logs_dir = batch_dir / "logs"
    status_path = batch_dir / "status.json"
    opponents = [item.strip() for item in args.post_eval_opponents.split(",") if item.strip()]
    statuses: list[dict] = []

    plan = {
        "kind": "fresh_population_v31_plan",
        "batch_id": args.batch_id,
        "args": vars(args),
        "runs": [asdict(run) for run in runs],
        "commands": [train_command(run, args) for run in runs],
        "post_eval_commands": {
            run.run_id: [eval_command(run, args, opponent) for opponent in opponents]
            for run in runs
        },
    }
    write_json(batch_dir / "plan.json", plan)
    build_manifest(batch_dir, runs, statuses)
    print(json.dumps({"kind": "fresh_population_v31_plan_written", "path": str(batch_dir / "plan.json")}, sort_keys=True), flush=True)

    if args.dry_run:
        return 0

    for run in runs:
        started = time.perf_counter()
        status = {
            "run_id": run.run_id,
            "run_dir": run.run_dir,
            "status": "training",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        statuses.append(status)
        write_json(status_path, {"kind": "fresh_population_v31_status", "batch_id": args.batch_id, "runs": statuses})
        build_manifest(batch_dir, runs, statuses)
        print(json.dumps({"kind": "fresh_population_v31_run_started", "run_id": run.run_id, "run_dir": run.run_dir}, sort_keys=True), flush=True)

        train_rc = run_and_log(train_command(run, args), logs_dir / f"{run.run_id}_train.log", args.tee)
        status["train_returncode"] = train_rc
        status["train_elapsed_sec"] = round(time.perf_counter() - started, 3)
        status["status"] = "trained" if train_rc == 0 else "failed_train"
        write_json(status_path, {"kind": "fresh_population_v31_status", "batch_id": args.batch_id, "runs": statuses})
        build_manifest(batch_dir, runs, statuses)
        if train_rc != 0:
            print(json.dumps({"kind": "fresh_population_v31_run_failed", "run_id": run.run_id, "stage": "train"}, sort_keys=True), flush=True)
            return train_rc

        eval_results = []
        for opponent in opponents:
            eval_started = time.perf_counter()
            eval_rc = run_and_log(
                eval_command(run, args, opponent),
                logs_dir / f"{run.run_id}_eval_{opponent}.log",
                args.tee,
            )
            eval_results.append(
                {
                    "opponent_bot": opponent,
                    "returncode": eval_rc,
                    "elapsed_sec": round(time.perf_counter() - eval_started, 3),
                }
            )
            if eval_rc != 0:
                status["eval_results"] = eval_results
                status["status"] = "failed_eval"
                write_json(status_path, {"kind": "fresh_population_v31_status", "batch_id": args.batch_id, "runs": statuses})
                build_manifest(batch_dir, runs, statuses)
                return eval_rc

        status["eval_results"] = eval_results
        status["status"] = "complete"
        status["elapsed_sec"] = round(time.perf_counter() - started, 3)
        write_json(status_path, {"kind": "fresh_population_v31_status", "batch_id": args.batch_id, "runs": statuses})
        build_manifest(batch_dir, runs, statuses)
        print(json.dumps({"kind": "fresh_population_v31_run_finished", "run_id": run.run_id, "status": "complete"}, sort_keys=True), flush=True)

    manifest = build_manifest(batch_dir, runs, statuses)
    print(
        json.dumps(
            {
                "kind": "fresh_population_v31_batch_finished",
                "batch_id": args.batch_id,
                "compatible": sum(1 for item in manifest["entries"] if item["compatible"]),
                "manifest_path": str(batch_dir / "manifest.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
