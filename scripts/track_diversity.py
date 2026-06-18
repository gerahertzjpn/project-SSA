"""Diversity tracker — are the 90 population nets genuinely distinct, or near-duplicate
"brothers"? Behavioral similarity, the signal that would later drive diversity-aware culling.

Method: build a fixed bank of game states (from ancestor self-play), then for each net take
its argmax action on every bank state -> a behavioral fingerprint. Two nets that pick the
SAME action on (say) >85% of states play identically = brothers. Compared WITHIN each
faction (cross-faction nets differ trivially by unit roster; the real concern is same-faction
look-alikes). Reports per-faction similarity, the most-similar pairs, brother count, and an
overall diversity score; appends diversity_progress.jsonl.

    python scripts/track_diversity.py --run-dir training_runs/league_population_v1 --bank 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.training.policy import MaskedActorCritic  # noqa: E402
from src.training.population import (  # noqa: E402
    CONTROLLED_SLOTS, OPPONENT_SLOTS, build_default_population, write_frozen_pool,
)
from src.training.population_roster import build_population  # noqa: E402
from src.training.team_env import TeamTrainingEnv, TeamTrainingEnvConfig  # noqa: E402

FACTIONS = ("humans", "infected", "robots")
BROTHER_THRESHOLD = 0.85    # argmax-agreement above this = behavioral near-duplicate


def _ancestor_state(nid):
    return torch.load(ROOT / "training_runs/population_v1/nets" / nid / "latest.pt",
                      map_location="cpu", weights_only=False)["policy_state"]


def build_state_bank(specs, per_faction, seed):
    """Collect (obs, mask) states per faction from ancestor self-play (stable reference)."""
    anc = build_default_population()
    states = {a.net_id: _ancestor_state(a.net_id) for a in anc}
    anc_by_id = {a.net_id: a for a in anc}
    pool = ROOT / "training_runs/population_v1" / "_diversity_pool.pt"
    write_frozen_pool(pool, anc_by_id, states, update=0)
    nets = {}
    for a in anc:
        n = MaskedActorCritic(hidden_size=a.hidden_size); n.load_state_dict(states[a.net_id]); n.eval()
        nets[a.net_id] = n
    bank = {f: [] for f in FACTIONS}
    ctrl = ["humans_1", "infected_1", "robots_1"]
    opp = ["humans_2", "infected_2", "robots_2"]
    slot_fac = {s: specs[c].faction for s, c in zip(CONTROLLED_SLOTS, ctrl)}
    pf = {**slot_fac, **{s: specs[o].faction for s, o in zip(OPPONENT_SLOTS, opp)}}
    e = 0
    while min(len(v) for v in bank.values()) < per_faction and e < 200:
        cfg = TeamTrainingEnvConfig(
            controlled_players=CONTROLLED_SLOTS, seed=seed + e * 13, opponent_bot="policy",
            opponent_policy_checkpoint=str(pool), opponent_policy_deterministic=False,
            opponent_policy_state_ids={s: o for s, o in zip(OPPONENT_SLOTS, opp)},
            player_factions=pf, max_ticks=300, record_replay=False, record_trace=False,
            sample_orders=True, order_shaping_weight=0.0, no_order_prob=0.5)
        env = TeamTrainingEnv(cfg)
        obs, _ = env.reset(seed=seed + e * 13, player_factions=pf)
        done = False
        while not done:
            masks = env.action_masks()
            actions = {}
            for s in CONTROLLED_SLOTS:
                fac = slot_fac[s]
                if len(bank[fac]) < per_faction:
                    bank[fac].append((np.asarray(obs[s], dtype=np.float32),
                                      np.asarray(masks[s], dtype=np.bool_)))
                actions[s] = nets[ctrl[CONTROLLED_SLOTS.index(s)]].act(
                    obs[s], masks[s], deterministic=False).action
            r = env.step(actions)
            obs = r.observations
            done = bool(r.terminal or r.truncated)
        e += 1
    pool.unlink(missing_ok=True)
    return {f: v[:per_faction] for f, v in bank.items()}


def fingerprint(net, bank_states):
    """Argmax action per bank state -> (N, action_dim) array (action is factorized)."""
    acts = [np.asarray(net.act(o, m, deterministic=True).action).ravel() for o, m in bank_states]
    return np.asarray(acts, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="training_runs/league_population_v1")
    ap.add_argument("--bank", type=int, default=300, help="states per faction")
    ap.add_argument("--seed", type=int, default=77)
    args = ap.parse_args()
    run_dir = ROOT / args.run_dir

    descendants = build_population()
    specs = {d.net_id: d for d in descendants}
    specs.update({a.net_id: a for a in build_default_population()})
    print(f"building state bank ({args.bank}/faction)...")
    bank = build_state_bank(specs, args.bank, args.seed)

    by_fac = {f: [d for d in descendants if d.faction == f] for f in FACTIONS}
    rows = {}
    all_pairs = []
    overall = []
    for f in FACTIONS:
        members = by_fac[f]
        fps = {}
        for d in members:
            ck = run_dir / "nets" / d.net_id / "latest.pt"
            if not ck.exists():
                continue
            net = MaskedActorCritic(hidden_size=d.hidden_size)
            net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["policy_state"])
            net.eval()
            fps[d.net_id] = (np.asarray(fingerprint(net, bank[f])), specs[d.net_id])
        pairs = []
        for a, b in combinations(fps, 2):
            agree = float(np.mean(np.all(fps[a][0] == fps[b][0], axis=1)))  # full-action match
            pairs.append((agree, a, b))
            overall.append(agree)
        pairs.sort(reverse=True)
        all_pairs.extend(pairs)
        brothers = [(round(ag, 3), a, b) for ag, a, b in pairs if ag >= BROTHER_THRESHOLD]
        rows[f] = {"n": len(fps), "mean_agreement": round(float(np.mean([p[0] for p in pairs])), 3),
                   "max_agreement": round(pairs[0][0], 3) if pairs else 0.0,
                   "brother_pairs": len(brothers),
                   "top": [(round(ag, 3), a, b) for ag, a, b in pairs[:3]]}

    diversity = round(1 - float(np.mean(overall)), 3) if overall else 0.0
    brother_total = sum(r["brother_pairs"] for r in rows.values())
    record = {"ts": int(time.time()), "diversity_score": diversity,
              "brother_pairs_total": brother_total, "threshold": BROTHER_THRESHOLD,
              "by_faction": rows}
    with open(run_dir / "diversity_progress.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    print("\n=== DIVERSITY (behavioral, within-faction argmax agreement) ===")
    print(f"  OVERALL diversity score: {diversity}  (1 = all distinct; brothers total: {brother_total})")
    for f in FACTIONS:
        r = rows[f]
        print(f"  {f:<9} n={r['n']} mean_agree {r['mean_agreement']} max {r['max_agreement']} "
              f"brothers(>={BROTHER_THRESHOLD}): {r['brother_pairs']}")
        for ag, a, b in r["top"]:
            tag = "  <-- BROTHER" if ag >= BROTHER_THRESHOLD else ""
            print(f"      {ag:.3f}  {a}  ~  {b}{tag}")


if __name__ == "__main__":
    main()
