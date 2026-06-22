"""Re-author every team's tournament PLAYBOOK for a World via the LLM.

Why: a World bootstrapped while the LLM was unavailable (no ``ANTHROPIC_API_KEY`` in the server
process, or the ``llm_manager`` flag off) gets the HEURISTIC fallback playbook for all 22 teams —
``playbook.build_default_plan`` emits the SAME 3 hardcoded overrides for everyone (only the resulting
strategy varies slightly), so every manager looks identical. The live/advance jobs only fill MISSING
playbooks, so those heuristic plans never get upgraded. This script force-re-authors them.

Usage:
    python scripts/world_replan.py <world_id>                 # re-author ALL teams
    python scripts/world_replan.py <world_id> --only-heuristic # only those still on the heuristic

⚠️ Run while the viewer is NOT actively writing this world (finish/stop a live reveal first) to avoid
a save race. Re-authoring affects FUTURE matches (the already-played live_replay is untouched); to see
the new playbooks in action, re-run the season ("По дням") after replanning.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))            # run as `python scripts/world_replan.py` from anywhere

from src.world import feature_flags, llm_manager  # noqa: E402


def _world_path(world_id: str) -> Path:
    return ROOT / "data" / "world" / world_id / "world.json"


def _world_llm():
    """Same gate the server uses: a real client only when the flag is on AND a key/SDK is present."""
    if not feature_flags.is_enabled("llm_manager"):
        return None
    return llm_manager.get_client()


def _cond_keys(plan: dict) -> tuple:
    return tuple(sorted(o.get("note", str(o.get("when"))) for o in plan.get("overrides", [])))


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-author World playbooks via the LLM")
    ap.add_argument("world_id")
    ap.add_argument("--only-heuristic", action="store_true",
                    help="re-author only teams whose current plan is the heuristic fallback")
    args = ap.parse_args()

    path = _world_path(args.world_id)
    if not path.exists():
        raise SystemExit(f"world not found: {path}")
    world = json.loads(path.read_text(encoding="utf-8"))

    llm = _world_llm()
    print(f"LLM active: {bool(llm)}  "
          f"({'will author varied playbooks' if llm else 'None -> would FALL BACK to heuristic; set ANTHROPIC_API_KEY and enable llm_manager'})")
    if not llm:
        print("Aborting: nothing to gain re-authoring without the LLM (you'd get the same heuristic).")
        raise SystemExit(2)

    feed: list = []
    src_counts: Counter = Counter()
    base_counts: Counter = Counter()
    cond_sets: Counter = Counter()
    n = 0
    for team in world["teams"].values():
        if len(team.get("roster", [])) < 3:
            continue
        if args.only_heuristic and (team.get("tournament_plan") or {}).get("source") == "llm":
            continue
        plan = llm_manager.author_playbook(world, team, llm=llm, feed=feed)
        src_counts[plan.get("source", "?")] += 1
        base_counts[plan["base"]["strategy"]] += 1
        cond_sets[_cond_keys(plan)] += 1
        n += 1
        print(f"  {team['abbrev']:6} base={plan['base']['strategy']:10} "
              f"overrides={len(plan.get('overrides', []))} src={plan.get('source')}")

    path.write_text(json.dumps(world), encoding="utf-8")
    print(f"\nRe-authored {n} playbooks -> {path}")
    print(f"  by source       : {dict(src_counts)}")
    print(f"  base strategies : {dict(base_counts)}  ({len(base_counts)} distinct)")
    print(f"  distinct condition-sets among teams: {len(cond_sets)} (was 1 when all heuristic)")
    fbs = [e for e in feed if e.get("type") == "fallback"]
    if fbs:
        print(f"  ⚠️ {len(fbs)} LLM calls FELL BACK to heuristic (errors) — first: {fbs[0].get('msg')}")


if __name__ == "__main__":
    main()
