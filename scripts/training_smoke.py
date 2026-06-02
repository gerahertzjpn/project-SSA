import argparse
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training.rollout import run_random_rollouts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training environment smoke rollouts without learning")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--player", default="A1")
    parser.add_argument("--opponent-bot", choices=["scripted", "random"], default="scripted")
    parser.add_argument("--ally-bot", choices=["scripted", "random"], default="scripted")
    parser.add_argument("--max-ticks", type=int, default=None)
    args = parser.parse_args()

    summary = run_random_rollouts(
        episodes=args.episodes,
        seed=args.seed,
        controlled_player=args.player,
        opponent_bot=args.opponent_bot,
        ally_bot=args.ally_bot,
        max_ticks=args.max_ticks,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
