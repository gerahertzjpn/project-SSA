from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.spaces import ACTION_DIM, OBSERVATION_DIM, OBSERVATION_VERSION


CONTROLLED_PLAYERS = ("A1", "A2", "A3")


def parse_slot(value: str) -> tuple[str, str, str]:
    try:
        target, source = value.split("=", 1)
        checkpoint, source_player = source.rsplit(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "slot must look like A1=path/to/checkpoint.pt:A1"
        ) from error
    target = target.strip()
    checkpoint = checkpoint.strip()
    source_player = source_player.strip()
    if target not in CONTROLLED_PLAYERS:
        raise argparse.ArgumentTypeError(f"unknown target player: {target}")
    if source_player not in CONTROLLED_PLAYERS:
        raise argparse.ArgumentTypeError(f"unknown source player: {source_player}")
    if not checkpoint:
        raise argparse.ArgumentTypeError("checkpoint path cannot be empty")
    return target, checkpoint, source_player


def load_team_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("observation_version") != OBSERVATION_VERSION
        or int(checkpoint.get("observation_dim") or -1) != OBSERVATION_DIM
        or int(checkpoint.get("action_dim") or -1) != ACTION_DIM
    ):
        raise ValueError(
            f"Incompatible checkpoint {path}: "
            f"{checkpoint.get('observation_version')}, "
            f"{checkpoint.get('observation_dim')}, {checkpoint.get('action_dim')}"
        )
    policy_states = checkpoint.get("policy_states")
    if not isinstance(policy_states, dict):
        raise ValueError(f"Expected team checkpoint with policy_states: {path}")
    return checkpoint


def resolved_checkpoint(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a team checkpoint from per-slot source checkpoints")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--slot",
        action="append",
        type=parse_slot,
        required=True,
        help="Slot assignment like A1=training_runs/run/checkpoints/u.pt:A1. Repeat for A1,A2,A3.",
    )
    parser.add_argument("--update", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=0)
    parser.add_argument("--label", default="mixed_team_init")
    args = parser.parse_args()

    slots = dict[str, tuple[str, str]]()
    for target, checkpoint, source_player in args.slot:
        if target in slots:
            raise ValueError(f"duplicate target slot: {target}")
        slots[target] = (checkpoint, source_player)
    missing = [player for player in CONTROLLED_PLAYERS if player not in slots]
    if missing:
        raise ValueError(f"missing target slots: {', '.join(missing)}")

    checkpoints: dict[str, dict[str, Any]] = {}
    source_meta: dict[str, dict[str, Any]] = {}
    policy_states = {}
    base_config: dict[str, Any] | None = None
    hidden_size: int | None = None

    for target_player in CONTROLLED_PLAYERS:
        checkpoint_raw, source_player = slots[target_player]
        checkpoint_path = resolved_checkpoint(checkpoint_raw)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        cache_key = str(checkpoint_path)
        if cache_key not in checkpoints:
            checkpoints[cache_key] = load_team_checkpoint(checkpoint_path)
        checkpoint = checkpoints[cache_key]
        config = checkpoint.get("config") or {}
        checkpoint_hidden = int(config.get("hidden_size", 128) or 128)
        if hidden_size is None:
            hidden_size = checkpoint_hidden
            base_config = dict(config)
        elif checkpoint_hidden != hidden_size:
            raise ValueError(
                f"Mixed checkpoint hidden size mismatch: {checkpoint_path} has "
                f"{checkpoint_hidden}, expected {hidden_size}"
            )
        policy_states[target_player] = checkpoint["policy_states"][source_player]
        source_meta[target_player] = {
            "checkpoint": checkpoint_path.relative_to(ROOT).as_posix()
            if checkpoint_path.is_relative_to(ROOT)
            else str(checkpoint_path),
            "source_player": source_player,
            "source_update": int(checkpoint.get("update", -1)),
            "source_total_steps": int(checkpoint.get("total_steps", 0) or 0),
        }

    assert base_config is not None
    base_config["role_assignment_mode"] = "shuffle"
    output = resolved_checkpoint(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "mixed_team_checkpoint",
        "label": args.label,
        "policy_states": policy_states,
        "optimizer_states": {},
        "config": base_config,
        "observation_version": OBSERVATION_VERSION,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "update": int(args.update),
        "total_steps": int(args.total_steps),
        "source_slots": source_meta,
    }
    torch.save(payload, output)

    print(json.dumps({
        "kind": "mixed_team_checkpoint_created",
        "output": output.relative_to(ROOT).as_posix() if output.is_relative_to(ROOT) else str(output),
        "label": args.label,
        "source_slots": source_meta,
        "update": int(args.update),
        "total_steps": int(args.total_steps),
    }, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
