from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bots.policy_bot import checkpoint_is_compatible
from src.training.spaces import ACTION_DIM, OBSERVATION_DIM, OBSERVATION_VERSION


def parse_updates(value: str) -> list[str]:
    updates = [item.strip() for item in str(value).split(",") if item.strip()]
    if not updates:
        raise argparse.ArgumentTypeError("Expected at least one update id or 'latest'")
    return updates


def checkpoint_path_for_update(run_dir: Path, update: str) -> Path:
    if update == "latest":
        return run_dir / "checkpoints" / "latest.pt"
    return run_dir / "checkpoints" / f"checkpoint_update_{int(update)}.pt"


def checkpoint_entry(path: Path, source_run_dir: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    if not checkpoint_is_compatible(path):
        raise ValueError(
            f"Checkpoint is not compatible with current interface "
            f"({OBSERVATION_VERSION}, {OBSERVATION_DIM}, {ACTION_DIM}): {path}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    try:
        relative_path = path.resolve().relative_to(ROOT)
    except ValueError:
        relative_path = path.resolve()
    return {
        "id": f"{source_run_dir.name}:u{int(checkpoint.get('update', -1))}",
        "path": relative_path.as_posix(),
        "update": int(checkpoint.get("update", -1)),
        "total_steps": int(checkpoint.get("total_steps", 0) or 0),
        "observation_version": checkpoint.get("observation_version"),
        "observation_dim": int(checkpoint.get("observation_dim", -1)),
        "action_dim": int(checkpoint.get("action_dim", -1)),
        "hidden_size": int((checkpoint.get("config") or {}).get("hidden_size", 128) or 128),
    }


def build_pool(run_dir: Path, updates: list[str]) -> dict:
    entries = []
    seen_paths = set()
    for update in updates:
        path = checkpoint_path_for_update(run_dir, update)
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        entries.append(checkpoint_entry(path, run_dir))
    return {
        "kind": "self_play_policy_pool",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_run_dir": str(run_dir),
        "observation_version": OBSERVATION_VERSION,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "selection": "manual_or_uniform_future_training",
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compatible self-play policy pool manifest")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--updates", default="latest", type=parse_updates)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    pool = build_pool(run_dir, args.updates)
    out_path = (
        Path(args.out)
        if args.out
        else Path("training_runs/self_play_pools") / f"{run_dir.name}_pool.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(pool, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({
        "kind": "self_play_pool_built",
        "out": str(out_path),
        "entries": len(pool["entries"]),
        "paths": [entry["path"] for entry in pool["entries"]],
    }, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
