"""
Entry point: run one or more matches and save replays.

Usage:
    python main.py                # one match, saves replay_0.json
    python main.py --runs 20      # 20 matches, saves replay_0.json ... replay_19.json
    python main.py --seed 42      # specific seed
    python main.py --bot scripted # use scripted bots (default: random)
    python main.py --mode training --runs 100  # headless summaries, no replay JSON
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.core.runner import GENERATION_PROFILES, format_match_result, run_match


def main():
    parser = argparse.ArgumentParser(description="Run autobattler simulation")
    parser.add_argument("--runs", type=int, default=1, help="Number of matches to run")
    parser.add_argument("--seed", type=int, default=42, help="Starting random seed")
    parser.add_argument("--bot", choices=["random", "scripted"], default="random")
    parser.add_argument("--mode", choices=sorted(GENERATION_PROFILES), default="replay")
    parser.add_argument("--out", default="replays", help="Output directory for replay mode")
    args = parser.parse_args()

    profile = GENERATION_PROFILES[args.mode]
    if profile.save_replay:
        os.makedirs(args.out, exist_ok=True)

    results = {"TEAM_A": 0, "TEAM_B": 0, "DRAW": 0}
    for i in range(args.runs):
        seed = args.seed + i
        path = os.path.join(args.out, f"replay_{i}.json") if profile.save_replay else None
        result = run_match(seed=seed, bot_type=args.bot, replay_path=path, mode=args.mode)
        print(format_match_result(result))
        results[result.winner] = results.get(result.winner, 0) + 1

    if args.runs > 1:
        print(f"\nResults over {args.runs} matches ({args.mode} mode): {results}")


if __name__ == "__main__":
    main()
