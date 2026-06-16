"""Create and inspect a World — a runnable entry point for the foundation.

No training, no LLM, no nets: uses the deterministic match simulator so you can
see the whole pipeline (new world -> Season-0 bootstrap -> regular seasons ->
economy/Elo/calendar). Persists to data/world/<world_id>/ (gitignored).

    python scripts/world_demo.py --world-id world-1 --seasons 3 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regionals.franchises import franchise_specs
from src.world import calendar as cal
from src.world import economy, season
from src.world.world import (
    lower_teams, new_world, save_world, upper_teams,
)


def _table(world, teams, title):
    print(f"\n{title}")
    rows = sorted(teams, key=lambda t: -t["elo"])
    for t in rows:
        grp = f" [{t['group']}]" if t["group"] else ""
        print(f"  {t['abbrev']:>4}{grp}  Elo {t['elo']:7.1f}  "
              f"money {t['treasury']:>8}  titles {t['titles']}  {t['name']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create/inspect a World")
    ap.add_argument("--world-id", default="world-demo")
    ap.add_argument("--teams", type=int, default=22)
    ap.add_argument("--seasons", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    specs = franchise_specs()[: args.teams]
    world = new_world(args.world_id, specs, name=args.world_id, seed=args.seed)
    print(f"Created {args.world_id}: {len(world['players'])} players, {len(world['teams'])} teams")

    season.bootstrap_world(world, seed=args.seed)
    print(f"Season-0 bootstrap: {world['bootstrap']} -> now season {world['season']}")

    for _ in range(args.seasons):
        s = world["season"]
        summary = season.run_regular_season(world)
        print(f"\n=== Season {s} ===")
        print(f"  Upper champion : {world['teams'][summary['upper_champion']]['name']}")
        print(f"  Lower champion : {world['teams'][summary['lower_champion']]['name']}")
        print(f"  Promoted       : {[world['teams'][t]['name'] for t in summary['promoted']]}")
        print(f"  Relegated      : {[world['teams'][t]['name'] for t in summary['relegated']]}")
        world["season"] = s + 1
        cal.skip_month(world, 1)  # off-season month

    _table(world, upper_teams(world), "UPPER LEAGUE")
    _table(world, lower_teams(world), "LOWER LEAGUE")

    # a sample player card with lineage + history
    salaried = sorted(world["players"].values(), key=lambda p: -p["salary"])[:1]
    if salaried:
        p = salaried[0]
        print(f"\nTop-paid player: {p['nickname']} ({p['faction']}/{p['archetype']}) "
              f"SoloQ {p['soloq_rating']:.0f}  salary {p['salary']:.1f}  "
              f"lineage={p['lineage']['origin']}  trophies={len(p['trophies'])}")

    print(f"\nTotal calendar events: {len(world['calendar'])}")
    print("Recent results:")
    for line in cal.results_table(world)[-5:]:
        print("  " + line)

    if not args.no_save:
        path = save_world(world)
        print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
