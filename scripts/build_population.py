"""Build + validate the league population roster (dry-run, no training).

Generates faction-locked descendants from the 18 ancestors across the four diversity
axes, asserts every genotype is unique (and distinct from the ancestors), prints a
summary, and writes the roster JSON. Does NOT train or touch the running base-18.

    python scripts/build_population.py                       # 90 nets, dry-run + summary
    python scripts/build_population.py --write               # also write the roster file
    python scripts/build_population.py --out population/roster_90_v1.json --write
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # cyrillic-safe on Windows

from src.training.population import build_default_population  # noqa: E402
from src.training.population_roster import (  # noqa: E402
    DEFAULT_ARCHETYPES, build_population, build_roster,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="population/roster_90_v1.json")
    ap.add_argument("--write", action="store_true", help="write the roster JSON to --out")
    ap.add_argument("--arch-variants", action="store_true",
                    help="enable the architecture axis (needs heterogeneous frozen-pool)")
    args = ap.parse_args()

    ancestors = build_default_population()
    descendants = build_population(ancestors, DEFAULT_ARCHETYPES, args.arch_variants)
    roster = build_roster(ancestors, DEFAULT_ARCHETYPES, args.arch_variants)  # validates

    print(f"ancestors (free players, SoloQ-only): {len(ancestors)}")
    print(f"league population (descendants):      {len(descendants)}")
    print(f"TOTAL nets:                           {len(ancestors) + len(descendants)}")
    print(f"unique genotype hashes:               "
          f"{len({d.genotype_hash() for d in descendants})} / {len(descendants)}  (vs ancestors: distinct)")

    print("\nper-faction population:", dict(Counter(d.faction for d in descendants)))
    print("per-archetype:         ", dict(Counter(d.archetype for d in descendants)))

    print("\nsample (humans_1 line -> 5 descendants):")
    print("  archetype    niche      lr        entropy  clip   order_w core_w  curriculum")
    for d in descendants:
        if d.parent_id == "humans_1":
            cur = ",".join(f"{k[0]}{v}" for k, v in d.opponent_curriculum)
            print(f"  {d.archetype:<11} {d.strategy_bias:<9} {d.learning_rate:<.6f}  "
                  f"{d.entropy_coef:<.4f}  {d.clip_coef:<.3f}  {d.order_shaping_weight:<5}  "
                  f"{d.core_progress_weight:<5}  {cur}")

    # axis spread sanity (diversity, not collapse)
    print("\naxis spread across population:")
    print(f"  entropy_coef: {min(d.entropy_coef for d in descendants):.4f}"
          f"..{max(d.entropy_coef for d in descendants):.4f}")
    print(f"  learning_rate: {min(d.learning_rate for d in descendants):.6f}"
          f"..{max(d.learning_rate for d in descendants):.6f}")
    print(f"  niches used: {sorted({d.strategy_bias for d in descendants})}")

    if args.write:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(roster, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote roster -> {out.relative_to(ROOT)}  ({out.stat().st_size} bytes)")
    else:
        print("\n(dry-run; pass --write to persist the roster JSON)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
