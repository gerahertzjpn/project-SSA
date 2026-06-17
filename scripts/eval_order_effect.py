"""Order-effectiveness eval: do strategy orders actually change outcomes?

Picks a WEAK team and a STRONG team (by net ids), measures the baseline win rate
with NO orders, then re-measures with the weak team given each strategy order
(strong team always un-ordered). Answers: (1) do orders shift behaviour/results,
(2) can an order boost a weak team to beat a stronger one. Game length (ticks) is
a behavioural proxy (aggressive/all-in -> shorter, turtle -> longer).

    python scripts/eval_order_effect.py --n 24
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.runner import run_match  # noqa: E402

NET = "policy:training_runs/base18_v1/nets/{nid}/latest.pt"
ROLES = {"A1": "carry_push", "A2": "anchor_defend", "A3": "support_flex"}

# Strategy conditions applied to the WEAK team (A). Strong team (B) stays un-ordered.
CONDITIONS = {
    "no_order": None,
    "aggressive/core_rush": {"tempo": "aggressive", "win_path": "core_rush"},
    "all_in/core_rush": {"tempo": "all_in", "win_path": "core_rush"},
    "turtle/economy": {"tempo": "turtle", "win_path": "economy"},
    "balanced/baron_lord": {"tempo": "balanced", "win_path": "baron_lord"},
}


def build(weak, strong):
    pm = {"A1": NET.format(nid=weak[0]), "A2": NET.format(nid=weak[1]), "A3": NET.format(nid=weak[2]),
          "B1": NET.format(nid=strong[0]), "B2": NET.format(nid=strong[1]), "B3": NET.format(nid=strong[2])}
    fac = lambda nid: nid.split("_")[0]  # noqa: E731
    pf = {"A1": fac(weak[0]), "A2": fac(weak[1]), "A3": fac(weak[2]),
          "B1": fac(strong[0]), "B2": fac(strong[1]), "B3": fac(strong[2])}
    return pm, pf


def orders_for(cond):
    if cond is None:
        return {p: {"absent": True} for p in ("A1", "A2", "A3", "B1", "B2", "B3")}
    a = {p: {"tempo": cond["tempo"], "win_path": cond["win_path"], "role": ROLES[p]}
         for p in ("A1", "A2", "A3")}
    b = {p: {"absent": True} for p in ("B1", "B2", "B3")}
    return {**a, **b}


def run_condition(pm, pf, cond, n, seed0):
    orders = orders_for(cond)
    wins = 0
    ticks = []
    for i in range(n):
        r = run_match(seed=seed0 + i, bot_type="mixed", mode="training",
                      player_models=pm, player_factions=pf, orders=orders)
        if r.winner == "TEAM_A":
            wins += 1
        ticks.append(r.total_ticks)
    return wins / n, statistics.mean(ticks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weak", default="robots_2,robots_4,robots_6")
    ap.add_argument("--strong", default="humans_1,humans_3,humans_4")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=4000)
    args = ap.parse_args()
    weak = args.weak.split(",")
    strong = args.strong.split(",")
    pm, pf = build(weak, strong)

    print(f"WEAK (A) = {weak}")
    print(f"STRONG (B) = {strong}")
    print(f"n={args.n} matches per condition. A-winrate = weak team's win rate.\n")
    base = None
    print(f"{'condition':22} {'A-winrate':>10} {'avg ticks':>10}")
    for name, cond in CONDITIONS.items():
        wr, tk = run_condition(pm, pf, cond, args.n, args.seed)
        if name == "no_order":
            base = wr
        delta = "" if base is None else f"  ({wr - base:+.2f} vs no-order)"
        print(f"{name:22} {wr:>10.2f} {tk:>10.0f}{delta}")
    print("\nReading: if no_order A-winrate < 0.5, the weak team is genuinely weaker; "
          "if an order's winrate climbs above ~0.5, the order boosted it to beat the stronger team.")


if __name__ == "__main__":
    main()
