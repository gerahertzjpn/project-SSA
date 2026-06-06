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
from src.bots.policy_bot import PolicyBot
from src.bots.random_bot import RandomBot
from src.bots.scripted_bot import ScriptedBot
from src.core.constants import PLAYER_IDS, TEAM_A_PLAYERS, TEAM_B_PLAYERS
from src.core.engine import GameEngine
from src.core.runner import current_code_commit, rules_meta
from src.core.state import make_initial_state


def make_policy_match_bots(
    seed: int,
    checkpoint: Path,
    controlled_player: str,
    deterministic: bool,
    ally_bot: str,
    opponent_bot: str,
) -> dict:
    bots = {}
    controlled_team = "TEAM_A" if controlled_player in TEAM_A_PLAYERS else "TEAM_B"
    for index, player_id in enumerate(PLAYER_IDS):
        team = "TEAM_A" if player_id in TEAM_A_PLAYERS else "TEAM_B"
        if player_id == controlled_player:
            bots[player_id] = PolicyBot(
                player_id,
                team,
                checkpoint_path=checkpoint,
                deterministic=deterministic,
                device="cpu",
            )
            continue
        bot_kind = ally_bot if team == controlled_team else opponent_bot
        bots[player_id] = make_baseline_bot(bot_kind, player_id, team, seed * 10 + index)
    return bots


def make_baseline_bot(bot_kind: str, player_id: str, team: str, seed: int):
    if bot_kind == "scripted":
        return ScriptedBot(player_id, team)
    if bot_kind == "random":
        return RandomBot(player_id, team, seed=seed)
    raise ValueError(f"Unknown baseline bot: {bot_kind}")


def run_one(request: dict[str, Any]) -> dict[str, Any]:
    seed = int(request["seed"])
    torch.set_num_threads(1)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed + int(request["sample_seed_offset"]))

    state = make_initial_state(seed=seed)
    engine = GameEngine(
        state,
        make_policy_match_bots(
            seed=seed,
            checkpoint=Path(request["checkpoint"]),
            controlled_player=str(request["controlled_player"]),
            deterministic=bool(request["deterministic"]),
            ally_bot=str(request["ally_bot"]),
            opponent_bot=str(request["opponent_bot"]),
        ),
        record_trace=True,
        record_replay=True,
    )
    winner = engine.run()
    match = analyze_match(seed, winner, state, engine.replay_ticks)
    match["controlled_player"] = str(request["controlled_player"])
    match["policy_mode"] = "argmax" if bool(request["deterministic"]) else "stochastic"
    return match


def run_chunk(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run_one(request) for request in requests]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Lord stats for a policy-controlled player")
    parser.add_argument("--matches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--checkpoint", default="training_runs/sprint2_scripted_clean_v1/checkpoints/checkpoint_update_100.pt")
    parser.add_argument("--controlled-player", default="A1")
    parser.add_argument("--mode", choices=["argmax", "stochastic"], default="argmax")
    scripted_choices = ["scripted", "scripted_core", "scripted_core_raider", "scripted_core_bmp"]
    parser.add_argument("--ally-bot", choices=scripted_choices + ["random"], default="scripted")
    parser.add_argument("--opponent-bot", choices=scripted_choices + ["random"], default="scripted")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    requests = [
        {
            "seed": int(args.seed) + index,
            "checkpoint": str(checkpoint),
            "controlled_player": args.controlled_player,
            "deterministic": args.mode == "argmax",
            "sample_seed_offset": 500_000_000,
            "ally_bot": args.ally_bot,
            "opponent_bot": args.opponent_bot,
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
        "kind": "policy_lord_stats",
        "config": {
            "matches": int(args.matches),
            "seed": int(args.seed),
            "workers": int(args.workers),
            "checkpoint": str(checkpoint),
            "controlled_player": args.controlled_player,
            "policy_mode": args.mode,
            "ally_bot": args.ally_bot,
            "opponent_bot": args.opponent_bot,
            "code_commit": current_code_commit(),
            "rules": rules_meta(),
        },
        "elapsed_sec": round(time.perf_counter() - started_at, 3),
        "summary": summary,
        "matches": matches,
    }

    out = Path(args.out or (
        f"analysis/policy_lord_stats_{args.mode}_{args.controlled_player}_"
        f"{int(args.seed)}_{int(args.matches)}.json"
    ))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)

    compact = {
        "kind": payload["kind"],
        "out": str(out),
        "elapsed_sec": payload["elapsed_sec"],
        "config": payload["config"],
        "summary": summary,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
