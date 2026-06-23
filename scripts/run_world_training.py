"""Run a World's pending team trainings (skill_up) + rookie forks for REAL — a manual kick for a
world whose off-season jobs errored, and the reference path the server auto-launch uses. Real PPO; slow.

  python scripts/run_world_training.py <world_id> --skill-nets a,b,c --rookies p108,p109
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.world import backends                          # noqa: E402
from src.world.offseason import reconcile_trained_card  # noqa: E402

RUN_DIR = "training_runs/league_population_v1"
MANIFEST = "population/frozen_population_v1.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("world_id")
    ap.add_argument("--skill-steps", type=int, default=800_000)
    ap.add_argument("--rookie-steps", type=int, default=1_300_000)
    ap.add_argument("--workers", type=int, default=0, help="0 = use the whole machine")
    ap.add_argument("--skill-nets", default="", help="comma net_ids to skill-train")
    ap.add_argument("--rookies", default="", help="comma rookie player_ids to fork")
    args = ap.parse_args()

    wp = ROOT / "data" / "world" / args.world_id / "world.json"
    world = json.loads(wp.read_text(encoding="utf-8"))
    be = backends.RealTrainingBackend(run_dir=RUN_DIR, require_flag=False)
    skills_log = []

    for nid in [n.strip() for n in args.skill_nets.split(",") if n.strip()]:
        print(f"[skill] training {nid} ({args.skill_steps:,} steps) ...", flush=True)
        try:
            r = be.continue_ppo({"target": {"net_id": nid}, "total_steps": args.skill_steps,
                                 "op": "skill_up"}, None)
            print("   ok:", r, flush=True)
            skills_log.extend(reconcile_trained_card(world, [nid]))
            wp.write_text(json.dumps(world, ensure_ascii=False), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print("   FAIL:", repr(e), flush=True)

    for rid in [r.strip() for r in args.rookies.split(",") if r.strip()]:
        print(f"[rookie] forking unique net for {rid} ({args.rookie_steps:,} steps) ...", flush=True)
        try:
            r = backends.fork_rookie_net(RUN_DIR, world, rid, args.rookie_steps,
                                         manifest_path=MANIFEST, workers=args.workers)
            print("   ok: bound", r["bound_net"], flush=True)
            wp.write_text(json.dumps(world, ensure_ascii=False), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print("   FAIL:", repr(e), flush=True)

    # record the skill deltas in the off-season report so «Характеристики» shows them
    if skills_log:
        world.setdefault("last_offseason", {}).setdefault("skills", []).extend(skills_log)
    wp.write_text(json.dumps(world, ensure_ascii=False), encoding="utf-8")
    print("saved world.", flush=True)


if __name__ == "__main__":
    main()
