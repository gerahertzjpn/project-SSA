"""Diversity-aware culling for the league population.

Removes the WEAKER of each near-duplicate "brother" pair + the lowest-fitness dead-weight,
and respawns FRESH unique players (fork from an ancestor of the SAME faction — keeps 30/30/30
— with a mutated, exploration-boosted recipe) so culling serves diversity instead of
collapsing it. The 18 ancestors are never touched. Writes the updated roster (survivors +
respawns) to <run-dir>/roster.json; the trainer resumes survivors and fork-inits respawns.

    python scripts/cull_population.py --run-dir training_runs/league_population_v1 --dry-run
    python scripts/cull_population.py --run-dir training_runs/league_population_v1 --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.training.lineage import _collect_episodes_for_net  # noqa: E402
from src.training.policy import MaskedActorCritic  # noqa: E402
from src.training.population import build_default_population, write_frozen_pool  # noqa: E402
from src.training.population_roster import (  # noqa: E402
    DEFAULT_ARCHETYPES, build_population, load_roster, save_roster, respawn_descendant,
)
from scripts.track_diversity import build_state_bank, fingerprint, BROTHER_THRESHOLD  # noqa: E402

FACTIONS = ("humans", "infected", "robots")
FIXED_ALLY = {"humans": "humans_1__skirmisher", "infected": "infected_1__skirmisher",
              "robots": "robots_1__skirmisher"}
OPP_ANCESTORS = ["humans_1", "infected_1", "robots_1"]
CV = {"order_shaping_weight": 0.0, "opponent_deterministic": False, "max_ticks": 300,
      "core_progress_weight": 0.5, "sample_orders": True, "no_order_prob": 1.0}


def _load(run_dir, nid, specs):
    return torch.load(run_dir / "nets" / nid / "latest.pt", map_location="cpu",
                      weights_only=False)["policy_state"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="training_runs/league_population_v1")
    ap.add_argument("--cull-budget", type=int, default=12)
    ap.add_argument("--fitness-episodes", type=int, default=8)
    ap.add_argument("--bank", type=int, default=200)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    run_dir = ROOT / args.run_dir
    rng = np.random.default_rng(args.seed)

    roster_path = run_dir / "roster.json"
    descendants = load_roster(roster_path) if roster_path.exists() else build_population()
    ancestors = build_default_population()
    specs = {d.net_id: d for d in descendants}
    specs.update({a.net_id: a for a in ancestors})
    anc_recipe = {a.faction: {"learning_rate": a.learning_rate, "entropy_coef": a.entropy_coef,
                              "clip_coef": a.clip_coef, "gae_lambda": a.gae_lambda} for a in ancestors}
    anc_lines = {f: [a.net_id for a in ancestors if a.faction == f] for f in FACTIONS}

    # states
    states = {d.net_id: _load(run_dir, d.net_id, specs) for d in descendants}
    anc_states = {a.net_id: torch.load(ROOT / "training_runs/population_v1/nets" / a.net_id / "latest.pt",
                                       map_location="cpu", weights_only=False)["policy_state"]
                  for a in ancestors}
    states.update(anc_states)

    # --- fitness: per-net winrate vs ancestors (fixed allies for comparability) ---
    pool = run_dir / "_cull_anc_pool.pt"
    write_frozen_pool(pool, {a.net_id: a for a in ancestors}, anc_states, update=0)
    print(f"fitness eval ({args.fitness_episodes} eps/net vs ancestors)...")
    fitness = {}
    for i, d in enumerate(sorted(descendants, key=lambda x: x.net_id)):
        allies = [FIXED_ALLY[f] for f in FACTIONS if f != d.faction]
        net = MaskedActorCritic(hidden_size=d.hidden_size)
        net.load_state_dict(states[d.net_id]); net.eval()
        all_st = {d.net_id: states[d.net_id], **{a: states[a] for a in allies}}
        _, summ = _collect_episodes_for_net(net, specs[d.net_id], allies, OPP_ANCESTORS,
                                            args.fitness_episodes, all_st, specs, str(pool), CV,
                                            episode_seed_base=args.seed + i * 101)  # deterministic
        fitness[d.net_id] = round(sum(s["win"] for s in summ) / max(1, len(summ)), 3)
    pool.unlink(missing_ok=True)

    # --- diversity: brother pairs within faction ---
    print(f"diversity eval (bank {args.bank}/faction)...")
    bank = build_state_bank(specs, args.bank, args.seed + 1)
    by_fac = {f: [d for d in descendants if d.faction == f] for f in FACTIONS}
    brothers = []   # (agreement, weaker_id, stronger_id)
    for f in FACTIONS:
        fps = {}
        for d in by_fac[f]:
            net = MaskedActorCritic(hidden_size=d.hidden_size)
            net.load_state_dict(states[d.net_id]); net.eval()
            fps[d.net_id] = fingerprint(net, bank[f])
        for a, b in combinations(fps, 2):
            ag = float(np.mean(np.all(fps[a] == fps[b], axis=1)))
            if ag >= BROTHER_THRESHOLD:
                weak, strong = (a, b) if fitness[a] <= fitness[b] else (b, a)
                brothers.append((round(ag, 3), weak, strong))
    brothers.sort(reverse=True)

    # --- cull selection: weaker-brothers first, then lowest-fitness dead-weight ---
    cull = []
    for ag, weak, strong in brothers:
        if weak not in cull and strong not in cull and len(cull) < args.cull_budget:
            cull.append(weak)
    if len(cull) < args.cull_budget:
        ranked = sorted((d.net_id for d in descendants if d.net_id not in cull),
                        key=lambda n: fitness[n])
        for nid in ranked:
            if len(cull) >= args.cull_budget:
                break
            cull.append(nid)

    # --- respawn: same faction, fork from a random ancestor line, fresh unique genotype ---
    existing = {d.genotype_hash() for d in descendants if d.net_id not in cull}
    survivors = [d for d in descendants if d.net_id not in cull]
    respawns = []
    for k, culled_id in enumerate(cull):
        fac = specs[culled_id].faction
        parent = str(rng.choice(anc_lines[fac]))
        base = DEFAULT_ARCHETYPES[int(rng.integers(len(DEFAULT_ARCHETYPES)))].name
        r = respawn_descendant(fac, parent, base, gen=2, k=k, rng=rng,
                               existing_hashes=existing, parent_recipe=anc_recipe[fac])
        existing.add(r.genotype_hash())
        respawns.append(r)

    new_roster = survivors + respawns
    assert len({d.genotype_hash() for d in new_roster}) == len(new_roster), "respawn genotype collision"
    assert len({d.net_id for d in new_roster}) == len(new_roster), "respawn net_id collision"
    by_fac_new = {f: sum(1 for d in new_roster if d.faction == f) for f in FACTIONS}

    print(f"\n=== CULL PLAN (budget {args.cull_budget}) ===")
    print(f"  brothers found: {len(brothers)}  | fitness range "
          f"{min(fitness.values()):.2f}..{max(fitness.values()):.2f}")
    print("  CULL:")
    for nid in cull:
        why = next((f"brother {ag} of {s}" for ag, w, s in brothers if w == nid), "dead-weight")
        print(f"    - {nid:<26} fit {fitness[nid]:.2f}  ({why})")
    print("  RESPAWN (fresh unique, same faction):")
    for r in respawns:
        print(f"    + {r.net_id:<22} fork<-{r.parent_id} arch {r.archetype} "
              f"ent {r.entropy_coef:.4f}")
    print(f"  new population: {len(new_roster)}  factions {by_fac_new}")

    if args.apply:
        save_roster(new_roster, roster_path)
        for nid in cull:
            d = run_dir / "nets" / nid
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        print(f"\nAPPLIED: wrote {roster_path.name} ({len(new_roster)} nets), removed {len(cull)} culled checkpoints.")
        print("Respawns will fork-init from ancestors on the next training launch (no checkpoint yet).")
    else:
        print("\n(dry-run; pass --apply to write roster + remove culled checkpoints)")


if __name__ == "__main__":
    main()
