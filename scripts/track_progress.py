"""Progress / readiness tracker for the base-18 run (population_v1).

Two evals, both reusing the exact training machinery (free-play, no order):

1) SKILL vs PAST SELF (plateau signal): play the CURRENT 18 nets against a frozen
   pool of their OWN selves ~LOOKBACK updates ago. When the per-faction improvement
   over past-self -> ~0, skill has plateaued. (impr = wr_current - wr_past control.)

2) FACTION BALANCE (head-to-head): PURE-faction teams, CURRENT weights both sides —
   3 humans vs 3 robots, 3 humans vs 3 infected, 3 robots vs 3 infected (both
   directions, rotating through all 6 nets per faction). A faction's strength = its
   mean win rate vs the other two. If one faction dominates -> imbalance (the base18
   humans-OP concern). This is the REAL balance signal (the old per-faction-vs-past
   number only measured balanced *progress*, not balanced strength).

Plus diversity (entropy spread + collapse) and stability (NaN/kl) from metrics.jsonl.
Appends one row to training_runs/<run>/progress.jsonl and prints the checklist.

    python scripts/track_progress.py --run-dir training_runs/population_v1 \
        --lookback 1500 --episodes 16 --balance-episodes 10
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.training.lineage import _collect_episodes_for_net  # noqa: E402
from src.training.policy import MaskedActorCritic  # noqa: E402
from src.training.population import build_default_population, write_frozen_pool  # noqa: E402

FACTIONS = ("humans", "infected", "robots")

# readiness thresholds (owner-tunable)
PLATEAU_IMPR = 0.03       # |per-faction improvement vs past self| below this = plateaued
BALANCE_SPREAD = 0.12     # max-min faction head-to-head strength; below this = balanced
DIV_SPREAD_MIN = 2.0      # min entropy spread across the 18 nets
DIV_COLLAPSE = 0.15       # a net below this (avg entropy) counts as collapsed
KL_MAX = 0.05             # mean approx_kl ceiling


def _load_state(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)["policy_state"]


def _metric_rows(run_dir: Path):
    return [json.loads(l) for l in open(run_dir / "metrics.jsonl", encoding="utf-8")
            if l.strip() and '"kind": "population_update"' in l]


def _nearest_milestone(net_dir: Path, target: int) -> int | None:
    ups = sorted(int(p.stem.split("_")[1]) for p in net_dir.glob("update_*.pt"))
    le = [u for u in ups if u <= target]
    return le[-1] if le else (ups[0] if ups else None)


def _team_winrate(controlled_states, ctrl_ids, opp_ids, all_states, specs, pool_path, cv, eps, seed):
    """Win rate of the controlled team (ctrl_ids, weights from all_states) vs opp_ids
    (loaded from the frozen pool at pool_path). Reuses _collect_episodes_for_net with
    the lead net + 2 allies = the 3 controlled players."""
    lead = ctrl_ids[0]
    net = MaskedActorCritic(hidden_size=specs[lead].hidden_size)
    net.load_state_dict(controlled_states[lead]); net.eval()
    _, summ = _collect_episodes_for_net(net, specs[lead], list(ctrl_ids[1:]), list(opp_ids),
                                        eps, all_states, specs, str(pool_path), cv,
                                        episode_seed_base=seed)
    return sum(s["win"] for s in summ), len(summ)


def faction_balance(net_ids, specs, current_states, pool_path, cv, k, seed):
    """Head-to-head pure-faction strength (current weights both sides)."""
    by_fac = {f: [n for n in net_ids if specs[n].faction == f] for f in FACTIONS}
    matrix = {}
    for ai, A in enumerate(FACTIONS):
        for bi, B in enumerate(FACTIONS):
            if A == B:
                continue
            wins = tot = 0
            for half, off in enumerate((0, 3)):           # two sub-teams cover all 6 nets
                a3, b3 = by_fac[A][off:off + 3], by_fac[B][off:off + 3]
                if len(a3) < 3 or len(b3) < 3:
                    continue
                w, n = _team_winrate(current_states, a3, b3, current_states, specs, pool_path,
                                     cv, k, seed + (ai * 100 + bi * 10 + half) * 1009)
                wins += w; tot += n
            matrix[f"{A}_vs_{B}"] = round(wins / (tot or 1), 3)
    strength = {f: round(statistics.mean(matrix[f"{f}_vs_{o}"] for o in FACTIONS if o != f), 3)
                for f in FACTIONS}
    spread = round(max(strength.values()) - min(strength.values()), 3)
    return strength, matrix, spread


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="training_runs/population_v1")
    ap.add_argument("--lookback", type=int, default=1500, help="updates back for the 'past self'")
    ap.add_argument("--episodes", type=int, default=16, help="episodes/net for the plateau eval")
    ap.add_argument("--balance-episodes", type=int, default=10, help="episodes per faction sub-matchup")
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    run_dir = ROOT / args.run_dir
    specs = {s.net_id: s for s in build_default_population()}
    net_ids = list(specs)
    rows = _metric_rows(run_dir)
    cur_update, total_steps = rows[-1]["update"], rows[-1]["total_steps"]
    past_update = _nearest_milestone(run_dir / "nets" / net_ids[0], max(0, cur_update - args.lookback))

    current_states = {nid: _load_state(run_dir / "nets" / nid / "latest.pt") for nid in net_ids}
    past_states = {nid: _load_state(run_dir / "nets" / nid / f"update_{past_update}.pt") for nid in net_ids}
    cv = {"order_shaping_weight": 0.0, "opponent_deterministic": False, "max_ticks": 300,
          "core_progress_weight": 0.5, "sample_orders": True, "no_order_prob": 1.0}

    print(f"current update {cur_update} (~{total_steps/1e6:.1f}M steps) vs past @update {past_update}")

    # --- (1) SKILL vs PAST SELF ------------------------------------------------
    past_pool = run_dir / "_progress_past_pool.pt"
    write_frozen_pool(past_pool, specs, past_states, update=past_update)
    skill = []
    for i, nid in enumerate(net_ids):
        others = [n for n in net_ids if n != nid]
        ally = [others[i % len(others)], others[(i + 6) % len(others)]]
        opp = [others[(i + 1) % len(others)], others[(i + 7) % len(others)], others[(i + 12) % len(others)]]
        wc, nc = _team_winrate(current_states, [nid] + ally, opp, past_states, specs, past_pool, cv, args.episodes, args.seed + i)
        wp, nps = _team_winrate(past_states, [nid] + ally, opp, past_states, specs, past_pool, cv, args.episodes, args.seed + i)
        skill.append({"net": nid, "faction": specs[nid].faction,
                      "wr_cur": round(wc / nc, 3), "impr": round(wc / nc - wp / nps, 3)})
    past_pool.unlink(missing_ok=True)
    per_fac_impr = {f: round(statistics.mean(r["impr"] for r in skill if r["faction"] == f), 3) for f in FACTIONS}
    overall_impr = round(statistics.mean(r["impr"] for r in skill), 3)
    max_fac_impr = max(abs(v) for v in per_fac_impr.values())

    # --- (2) FACTION BALANCE (head-to-head, current weights) -------------------
    cur_pool = run_dir / "_progress_cur_pool.pt"
    write_frozen_pool(cur_pool, specs, current_states, update=cur_update)
    fac_strength, fac_matrix, fac_spread = faction_balance(net_ids, specs, current_states, cur_pool, cv, args.balance_episodes, args.seed + 99)
    cur_pool.unlink(missing_ok=True)

    # --- diversity + stability + episode plateau (from metrics) ---------------
    recent = rows[-40:]
    ent = {}
    for r in recent:
        for nid, d in (r.get("loss_by_net") or {}).items():
            if isinstance(d, dict) and "entropy" in d:
                ent.setdefault(nid, []).append(d["entropy"])
    avg_ent = {nid: statistics.mean(v) for nid, v in ent.items() if v}
    spread = round((max(avg_ent.values()) - min(avg_ent.values())) if avg_ent else 0.0, 3)
    collapsed = [n for n, e in avg_ent.items() if e < DIV_COLLAPSE]
    kl_mean = round(statistics.mean(d["approx_kl"] for r in recent for d in (r.get("loss_by_net") or {}).values()
                                    if isinstance(d, dict) and "approx_kl" in d), 4)
    nan = any((d.get("entropy") != d.get("entropy")) for r in recent
              for d in (r.get("loss_by_net") or {}).values() if isinstance(d, dict))
    es = [r["mean_episode_steps"] for r in rows[-400:]]
    ep_slope = round((statistics.mean(es[-100:]) - statistics.mean(es[:100])), 1) if len(es) >= 200 else None

    # decisiveness ("sane decisions" proxy): low draw rate = games actually resolve,
    # short+falling episode length = purposeful play (not dithering to max_ticks).
    draw_rate = round(statistics.mean(r["controlled_draw_rate"] for r in recent), 3)
    mean_ep = round(statistics.mean(r["mean_episode_steps"] for r in recent), 1)

    # balance is INFO only (owner decision: the league handles faction balance via
    # economy/draft + patch-time training certificates) — NOT a readiness blocker.
    gates = {
        "skill_plateau": max_fac_impr < PLATEAU_IMPR,          # every faction stopped improving
        "decisive": draw_rate < 0.1,                           # games resolve (sane, not stalling)
        "diversity": spread >= DIV_SPREAD_MIN and not collapsed,
        "stability": (not nan) and kl_mean < KL_MAX,
        "episodes_plateau": (ep_slope is not None and abs(ep_slope) < 5),
    }
    ready = all(gates.values())

    record = {"ts": int(time.time()), "update": cur_update, "total_steps": total_steps,
              "past_update": past_update, "overall_impr": overall_impr, "per_faction_impr": per_fac_impr,
              "faction_strength": fac_strength, "faction_matrix": fac_matrix, "faction_spread": fac_spread,
              "entropy_spread": spread, "collapsed": collapsed, "kl_mean": kl_mean,
              "ep_slope_400": ep_slope, "per_net": skill, "gates": gates, "ready": ready}
    with open(run_dir / "progress.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    print("\n=== READINESS DASHBOARD ===")
    print(f"  SKILL vs past self (-{args.lookback}u): overall {overall_impr:+.3f}  per-faction {per_fac_impr}  (~0 = plateaued)")
    print(f"  FACTION BALANCE (head-to-head, current): strength {fac_strength}  spread {fac_spread} (gate <{BALANCE_SPREAD})")
    print(f"    matrix (row beats col): {fac_matrix}")
    print(f"  DIVERSITY: entropy spread {spread}  collapsed {collapsed or 'none'}")
    print(f"  STABILITY: kl {kl_mean}  NaN {nan}   EPISODES slope/400u {ep_slope}")
    print("  --- gates ---")
    for g, ok in gates.items():
        print(f"    [{'x' if ok else ' '}] {g}")
    print(f"  >>> READY FOR POPULATION: {'YES' if ready else 'NO'}")


if __name__ == "__main__":
    main()
