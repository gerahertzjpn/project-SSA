"""
Entry point: run one or more matches and save replays.

Usage:
    python main.py                # one match, saves replay_0.json
    python main.py --runs 20      # 20 matches, saves replay_0.json ... replay_19.json
    python main.py --seed 42      # specific seed
    python main.py --bot scripted # use scripted bots (default: random)
"""

import argparse
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.core.state import make_initial_state
from src.core.engine import GameEngine
from src.core.replay import save_replay
from src.core.constants import PLAYER_IDS, TEAM_A_PLAYERS, TEAM_B_PLAYERS
from src.bots.random_bot import RandomBot
from src.bots.scripted_bot import ScriptedBot


def make_bots(bot_type: str, seed_offset: int = 0) -> dict:
    bots = {}
    for i, pid in enumerate(PLAYER_IDS):
        team = "TEAM_A" if pid in TEAM_A_PLAYERS else "TEAM_B"
        if bot_type == "scripted":
            bots[pid] = ScriptedBot(pid, team)
        else:
            bots[pid] = RandomBot(pid, team, seed=i + seed_offset)
    return bots


def run_match(seed: int, bot_type: str, replay_path: str) -> str:
    state = make_initial_state(seed=seed)
    bots = make_bots(bot_type, seed_offset=seed * 10)
    engine = GameEngine(state, bots)
    winner = engine.run()

    save_replay(
        winner=winner,
        total_ticks=state.tick,
        ticks=engine.replay_ticks,
        path=replay_path,
    )

    print(f"[seed={seed}] Winner: {winner} | Ticks: {state.tick} | Replay: {replay_path}")
    return winner


def main():
    parser = argparse.ArgumentParser(description="Run autobattler simulation")
    parser.add_argument("--runs", type=int, default=1, help="Number of matches to run")
    parser.add_argument("--seed", type=int, default=42, help="Starting random seed")
    parser.add_argument("--bot", choices=["random", "scripted"], default="random")
    parser.add_argument("--out", default="replays", help="Output directory for replays")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    results = {"TEAM_A": 0, "TEAM_B": 0, "DRAW": 0}
    for i in range(args.runs):
        seed = args.seed + i
        path = os.path.join(args.out, f"replay_{i}.json")
        winner = run_match(seed=seed, bot_type=args.bot, replay_path=path)
        results[winner] = results.get(winner, 0) + 1

    if args.runs > 1:
        print(f"\nResults over {args.runs} matches: {results}")


if __name__ == "__main__":
    main()
