from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.analyze_random_lord_stats import aggregate, analyze_match, chunks
from src.core.runner import current_code_commit, rules_meta
from src.training.team_env import TeamTrainingEnv, TeamTrainingEnvConfig
from src.training.team_ppo_trainer import TeamPPOConfig, make_team_policies


def parse_players(value: str) -> tuple[str, ...]:
    players = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not players:
        raise argparse.ArgumentTypeError("Expected at least one player id")
    return players


def load_team_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {}) or {}
    policy_states = checkpoint.get("policy_states")
    if not isinstance(policy_states, dict) or not policy_states:
        raise ValueError(f"Expected team checkpoint with policy_states: {path}")
    return {
        "path": str(path),
        "update": int(checkpoint.get("update", -1)),
        "total_steps": int(checkpoint.get("total_steps", 0) or 0),
        "hidden_size": int(config.get("hidden_size", 128) or 128),
        "policy_states": policy_states,
    }


def run_one(request: dict[str, Any]) -> dict[str, Any]:
    seed = int(request["seed"])
    torch.set_num_threads(1)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed + int(request["sample_seed_offset"]))

    payload = load_team_checkpoint(Path(request["checkpoint"]))
    controlled_players = tuple(request["controlled_players"])
    config = TeamPPOConfig(
        controlled_players=controlled_players,
        opponent_bot=str(request["opponent_bot"]),
        ally_bot=str(request["ally_bot"]),
        opponent_policy_checkpoint=request.get("opponent_policy_checkpoint"),
        opponent_policy_deterministic=bool(request["opponent_policy_deterministic"]),
        max_ticks=int(request["max_ticks"]),
        hidden_size=int(payload["hidden_size"]),
        torch_threads=1,
    )
    policies = make_team_policies(config)
    for player_id in controlled_players:
        policies[player_id].load_state_dict(payload["policy_states"][player_id])
        policies[player_id].eval()

    env = TeamTrainingEnv(TeamTrainingEnvConfig(
        controlled_players=controlled_players,
        seed=seed,
        opponent_bot=config.opponent_bot,
        ally_bot=config.ally_bot,
        opponent_policy_checkpoint=config.opponent_policy_checkpoint,
        opponent_policy_deterministic=config.opponent_policy_deterministic,
        max_ticks=config.max_ticks,
        record_replay=True,
        record_trace=True,
    ))
    observations, _ = env.reset(seed=seed)
    done = False
    while not done:
        masks = env.action_masks()
        actions = {}
        for player_id in controlled_players:
            step = policies[player_id].act(
                observations[player_id],
                masks[player_id],
                deterministic=bool(request["deterministic"]),
                device="cpu",
            )
            actions[player_id] = step.action
        result = env.step(actions)
        observations = result.observations
        done = bool(result.terminal or result.truncated)

    state = env.state
    engine = env.engine
    if state is None or engine is None:
        raise RuntimeError("TeamTrainingEnv did not produce a final state")
    match = analyze_match(seed, state.winner or "DRAW", state, engine.replay_ticks)
    match["policy_mode"] = "argmax" if bool(request["deterministic"]) else "stochastic"
    match["controlled_players"] = list(controlled_players)
    return match


def run_chunk(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run_one(request) for request in requests]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Lord stats for a synchronized team checkpoint")
    parser.add_argument("--matches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["argmax", "stochastic"], default="argmax")
    parser.add_argument("--controlled-players", type=parse_players, default=("A1", "A2", "A3"))
    scripted_choices = ["scripted", "scripted_core", "scripted_core_raider", "scripted_core_bmp"]
    parser.add_argument("--ally-bot", choices=scripted_choices + ["random"], default="scripted")
    parser.add_argument("--opponent-bot", choices=scripted_choices + ["random", "policy"], default="scripted")
    parser.add_argument("--opponent-policy-checkpoint", default=None)
    parser.add_argument("--opponent-policy-mode", choices=["argmax", "stochastic"], default="stochastic")
    parser.add_argument("--max-ticks", type=int, default=300)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if args.opponent_bot == "policy" and not args.opponent_policy_checkpoint:
        raise ValueError("--opponent-policy-checkpoint is required with --opponent-bot policy")

    requests = [
        {
            "seed": int(args.seed) + index,
            "sample_seed_offset": 700_000_000,
            "checkpoint": str(checkpoint),
            "deterministic": args.mode == "argmax",
            "controlled_players": tuple(args.controlled_players),
            "ally_bot": args.ally_bot,
            "opponent_bot": args.opponent_bot,
            "opponent_policy_checkpoint": args.opponent_policy_checkpoint,
            "opponent_policy_deterministic": args.opponent_policy_mode == "argmax",
            "max_ticks": int(args.max_ticks),
        }
        for index in range(int(args.matches))
    ]

    started_at = time.perf_counter()
    if int(args.workers) <= 1:
        matches = run_chunk(requests)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(int(args.workers), len(requests))) as pool:
            nested = pool.map(run_chunk, chunks(requests, int(args.workers)))
        matches = [item for chunk in nested for item in chunk]
        matches.sort(key=lambda item: item["seed"])

    summary = aggregate(matches)
    payload = {
        "kind": "team_policy_lord_stats",
        "config": {
            "matches": int(args.matches),
            "seed": int(args.seed),
            "workers": int(args.workers),
            "checkpoint": str(checkpoint),
            "policy_mode": args.mode,
            "controlled_players": list(args.controlled_players),
            "ally_bot": args.ally_bot,
            "opponent_bot": args.opponent_bot,
            "opponent_policy_checkpoint": args.opponent_policy_checkpoint,
            "opponent_policy_mode": args.opponent_policy_mode,
            "code_commit": current_code_commit(),
            "rules": rules_meta(),
        },
        "elapsed_sec": round(time.perf_counter() - started_at, 3),
        "summary": summary,
        "matches": matches,
    }

    out = Path(args.out or (
        f"analysis/team_policy_lord_stats_{args.mode}_{int(args.seed)}_{int(args.matches)}.json"
    ))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)

    print(json.dumps({
        "kind": payload["kind"],
        "out": str(out),
        "elapsed_sec": payload["elapsed_sec"],
        "config": payload["config"],
        "summary": summary,
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
