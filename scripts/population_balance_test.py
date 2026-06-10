"""Large-scale balance audit of the trained 18-net population.

Answers four questions the user asked:
  1. Is there a clear favourite faction?           -> Experiment A (pure-faction H2H)
  2. How actively are mechanics used?              -> action-stream + final-state telemetry
     (buildings / units / economy / lord)
  3. If imbalanced, what is the cause?             -> per-mechanic faction breakdown
  4. Is any single net actually "imba"?            -> Experiment B (per-net win rate)

The 18 nets are faction-locked, so a net always plays its own faction in
whatever slot it occupies. Position is randomised so side-bias is averaged out.

Headless (no trace/replay). Reads checkpoints from training_runs/population_18_v1.

    python scripts/population_balance_test.py --smoke
    python scripts/population_balance_test.py --matches-per-pair 60 --mixed-matches 600
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.bots.policy_bot import PolicyBot
from src.core.constants import (
    FACTION_TYPES,
    PLAYER_IDS,
    TEAM_A_PLAYERS,
    TEAM_B_PLAYERS,
    UNIT_COST,
    UNIT_FACTION,
    BUILDING_COST,
    SPECIAL_DEST_ALLY_LORD,
    SPECIAL_DEST_ENEMY_LORD,
    ActionType,
)
from src.core.engine import Action, GameEngine
from src.core.state import make_initial_state

RUN_DIR = Path("training_runs/population_18_v1")
NET_IDS = [f"{fac}_{i}" for fac in FACTION_TYPES for i in range(1, 7)]
UNIT_COST_BY_NAME = {u.value: c for u, c in UNIT_COST.items()}
UNIT_FACTION_BY_NAME = {u.value: f for u, f in UNIT_FACTION.items()}
BUILDING_COST_BY_NAME = {b.value: c for b, c in BUILDING_COST.items()}


def net_faction(net_id: str) -> str:
    return net_id.rsplit("_", 1)[0]


def checkpoint_path(net_id: str) -> Path:
    return RUN_DIR / "nets" / net_id / "latest.pt"


# ── recording wrapper ────────────────────────────────────────────────────────
class RecordingBot:
    """Wraps a PolicyBot and tallies the kinds of actions it emits."""

    def __init__(self, inner: PolicyBot, net_id: str):
        self.inner = inner
        self.net_id = net_id
        self.faction = net_faction(net_id)
        self.player_id = inner.player_id
        self.team = inner.team
        self.stats = {
            "decisions": 0,
            "hold": 0,
            "produce": 0,
            "build": 0,
            "move": 0,
            "move_lord_ally": 0,
            "move_lord_enemy": 0,
            "multi": 0,
            "units_produced": Counter(),
            "buildings_built": Counter(),
            "resource_on_units": 0,
            "resource_on_buildings": 0,
        }

    def act(self, obs: dict) -> Action:
        action = self.inner.act(obs)
        self.stats["decisions"] += 1
        self._record(action)
        return action

    def _record(self, action: Action) -> None:
        at = action.action_type
        if at == ActionType.HOLD:
            self.stats["hold"] += 1
        elif at == ActionType.PRODUCE:
            self.stats["produce"] += 1
            if action.unit_type is not None:
                name = action.unit_type.value
                self.stats["units_produced"][name] += 1
                self.stats["resource_on_units"] += UNIT_COST_BY_NAME.get(name, 0)
        elif at == ActionType.BUILD:
            self.stats["build"] += 1
            if action.building_type is not None:
                name = action.building_type.value
                self.stats["buildings_built"][name] += 1
                self.stats["resource_on_buildings"] += BUILDING_COST_BY_NAME.get(name, 0)
        elif at == ActionType.MOVE:
            self.stats["move"] += 1
            if action.destination == SPECIAL_DEST_ALLY_LORD:
                self.stats["move_lord_ally"] += 1
            elif action.destination == SPECIAL_DEST_ENEMY_LORD:
                self.stats["move_lord_enemy"] += 1
        elif at == ActionType.MULTI:
            self.stats["multi"] += 1
            for sub in (action.sub_actions or []):
                self._record(sub)


_BOT_CACHE: dict[tuple[str, str], PolicyBot] = {}


def make_recording_bot(net_id: str, player_id: str, team: str) -> RecordingBot:
    key = (net_id, player_id)
    inner = _BOT_CACHE.get(key)
    if inner is None:
        inner = PolicyBot(
            player_id,
            team,
            checkpoint_path=checkpoint_path(net_id),
            deterministic=True,
        )
        _BOT_CACHE[key] = inner
    return RecordingBot(inner, net_id)


# ── one match ────────────────────────────────────────────────────────────────
def run_match(assignment: dict[str, str], seed: int) -> dict[str, Any]:
    """assignment: player_id -> net_id. Faction follows the net into its slot."""
    player_factions = {pid: net_faction(net_id) for pid, net_id in assignment.items()}
    state = make_initial_state(seed=seed, player_factions=player_factions)
    bots = {
        pid: make_recording_bot(net_id, pid, "TEAM_A" if pid in TEAM_A_PLAYERS else "TEAM_B")
        for pid, net_id in assignment.items()
    }
    engine = GameEngine(state, bots, record_trace=False, record_replay=False)
    winner = engine.run()

    baron = state.baron
    lord_owner = baron.allied_lord_owner
    return {
        "winner": winner,
        "winner_type": state.winner_type or "unknown",
        "ticks": state.tick,
        "assignment": dict(assignment),
        "bot_stats": {pid: bots[pid].stats for pid in assignment},
        "final_resources": {pid: round(float(state.players[pid].resources), 2) for pid in assignment},
        "lord_owner": lord_owner,
        "lord_owner_faction": (player_factions.get(lord_owner) if lord_owner in player_factions else None),
        "lord_expended": bool(baron.allied_lord_expended),
        "buildings_standing": _standing_buildings(state, player_factions),
    }


def _standing_buildings(state, player_factions: dict[str, str]) -> dict[str, int]:
    """Buildings still alive at match end, counted per owning faction."""
    out: Counter = Counter()
    for node in state.nodes.values():
        for b in node.buildings:
            fac = player_factions.get(b.owner)
            if fac is not None and b.hp > 0:
                out[fac] += 1
    return dict(out)


# ── aggregation helpers ──────────────────────────────────────────────────────
def merge_bot_stats(target: dict, stats: dict) -> None:
    for key in ("decisions", "hold", "produce", "build", "move",
                "move_lord_ally", "move_lord_enemy", "multi",
                "resource_on_units", "resource_on_buildings"):
        target[key] += stats[key]
    for name, n in stats["units_produced"].items():
        target["units_produced"][name] += n
    for name, n in stats["buildings_built"].items():
        target["buildings_built"][name] += n


def new_accumulator() -> dict:
    return {
        "decisions": 0, "hold": 0, "produce": 0, "build": 0, "move": 0,
        "move_lord_ally": 0, "move_lord_enemy": 0, "multi": 0,
        "resource_on_units": 0, "resource_on_buildings": 0,
        "units_produced": Counter(), "buildings_built": Counter(),
    }


def acc_to_serializable(acc: dict) -> dict:
    out = dict(acc)
    out["units_produced"] = dict(acc["units_produced"].most_common())
    out["buildings_built"] = dict(acc["buildings_built"].most_common())
    return out


# ── Experiment A: pure-faction head-to-head ──────────────────────────────────
def experiment_a(matches_per_pair: int, base_seed: int, rng: random.Random) -> dict[str, Any]:
    """3 nets of faction X (Team A) vs 3 nets of faction Y (Team B)."""
    faction_wins: Counter = Counter()
    faction_games: Counter = Counter()
    pair_rows: dict[str, dict] = {}
    faction_acc = {f: new_accumulator() for f in FACTION_TYPES}
    winner_types: Counter = Counter()

    nets_by_fac = {f: [n for n in NET_IDS if net_faction(n) == f] for f in FACTION_TYPES}

    pairs = [(x, y) for x in FACTION_TYPES for y in FACTION_TYPES]  # incl. mirrors
    seed = base_seed
    for (fx, fy) in pairs:
        a_wins = b_wins = draws = 0
        for _ in range(matches_per_pair):
            a_nets = rng.sample(nets_by_fac[fx], 3)
            b_nets = rng.sample(nets_by_fac[fy], 3)
            assignment = {
                "A1": a_nets[0], "A2": a_nets[1], "A3": a_nets[2],
                "B1": b_nets[0], "B2": b_nets[1], "B3": b_nets[2],
            }
            r = run_match(assignment, seed)
            seed += 1
            winner_types[r["winner_type"]] += 1
            if r["winner"] == "TEAM_A":
                a_wins += 1
            elif r["winner"] == "TEAM_B":
                b_wins += 1
            else:
                draws += 1
            # faction win attribution (skip mirrors for the global tally)
            if fx != fy:
                faction_games[fx] += 1
                faction_games[fy] += 1
                if r["winner"] == "TEAM_A":
                    faction_wins[fx] += 1
                elif r["winner"] == "TEAM_B":
                    faction_wins[fy] += 1
            # mechanics by faction
            for pid, st in r["bot_stats"].items():
                merge_bot_stats(faction_acc[net_faction(assignment[pid])], st)

        pair_rows[f"{fx}_A_vs_{fy}_B"] = {
            "games": matches_per_pair, "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
            "a_win_rate": round(a_wins / matches_per_pair, 3),
        }

    faction_winrate = {
        f: round(faction_wins[f] / faction_games[f], 3) if faction_games[f] else None
        for f in FACTION_TYPES
    }
    return {
        "matches_per_pair": matches_per_pair,
        "pair_results": pair_rows,
        "faction_winrate_nonmirror": faction_winrate,
        "faction_wins": dict(faction_wins),
        "faction_games": dict(faction_games),
        "winner_types": dict(winner_types),
        "faction_mechanics": {f: acc_to_serializable(faction_acc[f]) for f in FACTION_TYPES},
    }


# ── Experiment B: mixed teams, per-net win rate ──────────────────────────────
def experiment_b(mixed_matches: int, base_seed: int, rng: random.Random) -> dict[str, Any]:
    """Random mixed 3v3 from all 18 nets; attribute team result to each net."""
    net_wins: Counter = Counter()
    net_games: Counter = Counter()
    net_acc = {n: new_accumulator() for n in NET_IDS}
    lord_owner_faction: Counter = Counter()
    lord_used_games = 0
    winner_types: Counter = Counter()
    draws = 0

    seed = base_seed
    for _ in range(mixed_matches):
        six = rng.sample(NET_IDS, 6)
        assignment = dict(zip(PLAYER_IDS, six))
        r = run_match(assignment, seed)
        seed += 1
        winner_types[r["winner_type"]] += 1
        win_team = r["winner"]
        if win_team == "DRAW":
            draws += 1
        for pid, net_id in assignment.items():
            team = "TEAM_A" if pid in TEAM_A_PLAYERS else "TEAM_B"
            net_games[net_id] += 1
            if win_team == team:
                net_wins[net_id] += 1
            merge_bot_stats(net_acc[net_id], r["bot_stats"][pid])
        if r["lord_owner_faction"]:
            lord_owner_faction[r["lord_owner_faction"]] += 1
        if r["lord_owner"] is not None:
            lord_used_games += 1

    net_winrate = {
        n: round(net_wins[n] / net_games[n], 3) if net_games[n] else None
        for n in NET_IDS
    }
    # per-faction average net win rate
    fac_winrate = {}
    for f in FACTION_TYPES:
        vals = [net_winrate[n] for n in NET_IDS if net_faction(n) == f and net_winrate[n] is not None]
        fac_winrate[f] = round(sum(vals) / len(vals), 3) if vals else None

    return {
        "mixed_matches": mixed_matches,
        "draws": draws,
        "winner_types": dict(winner_types),
        "net_winrate": net_winrate,
        "net_games": dict(net_games),
        "faction_avg_net_winrate": fac_winrate,
        "lord_owner_faction": dict(lord_owner_faction),
        "lord_used_games": lord_used_games,
        "net_mechanics": {n: acc_to_serializable(net_acc[n]) for n in NET_IDS},
    }


# ── reporting ────────────────────────────────────────────────────────────────
def mechanics_rates(acc: dict) -> dict[str, Any]:
    d = max(1, acc["decisions"])
    return {
        "decisions": acc["decisions"],
        "produce_per_kdec": round(1000 * acc["produce"] / d, 1),
        "build_per_kdec": round(1000 * acc["build"] / d, 1),
        "move_per_kdec": round(1000 * acc["move"] / d, 1),
        "hold_frac": round(acc["hold"] / d, 3),
        "lord_ally_calls": acc["move_lord_ally"],
        "lord_enemy_calls": acc["move_lord_enemy"],
        "resource_units": acc["resource_on_units"],
        "resource_buildings": acc["resource_on_buildings"],
        "build_spend_frac": round(
            acc["resource_on_buildings"] / max(1, acc["resource_on_units"] + acc["resource_on_buildings"]), 3),
        "top_units": dict(Counter(acc["units_produced"]).most_common(4)),
        "buildings": dict(Counter(acc["buildings_built"]).most_common()),
    }


def print_report(report: dict[str, Any]) -> None:
    a = report["experiment_a_faction_h2h"]
    b = report["experiment_b_per_net"]
    print("\n" + "=" * 70)
    print("  POPULATION-18 BALANCE TEST")
    print("=" * 70)
    print(f"commit-free headless audit | elapsed {report['meta']['elapsed_s']}s | "
          f"total matches {report['meta']['total_matches']}")

    print("\n--- Q1. FACTION FAVOURITE (Experiment A, pure-faction H2H, non-mirror) ---")
    for f in FACTION_TYPES:
        wr = a["faction_winrate_nonmirror"][f]
        print(f"  {f:9s} win-rate = {wr}  ({a['faction_wins'].get(f,0)}/{a['faction_games'].get(f,0)})")
    print("  pairwise (Team-A side win-rate):")
    for name, row in a["pair_results"].items():
        print(f"    {name:28s} A_win={row['a_win_rate']:.3f}  "
              f"({row['a_wins']}-{row['b_wins']}-{row['draws']})")
    print(f"  winner_types={a['winner_types']}")

    print("\n--- Q2/Q3. MECHANICS USAGE BY FACTION (per 1000 decisions) ---")
    for f in FACTION_TYPES:
        r = mechanics_rates(a["faction_mechanics"][f])
        print(f"  [{f}] produce={r['produce_per_kdec']} build={r['build_per_kdec']} "
              f"move={r['move_per_kdec']} hold={r['hold_frac']} "
              f"lordA/E={r['lord_ally_calls']}/{r['lord_enemy_calls']} "
              f"bld_spend%={r['build_spend_frac']}")
        print(f"        top_units={r['top_units']}")
        print(f"        buildings={r['buildings']}")

    print("\n--- Q4. PER-NET WIN RATE (Experiment B, mixed teams) -> 'imba' check ---")
    ranked = sorted(b["net_winrate"].items(), key=lambda kv: (kv[1] is not None, kv[1]), reverse=True)
    for net_id, wr in ranked:
        bar = "#" * int(round((wr or 0) * 40))
        print(f"  {net_id:11s} {wr:.3f} ({b['net_games'].get(net_id,0):4d}g) {bar}")
    print(f"  faction avg net win-rate = {b['faction_avg_net_winrate']}")
    print(f"  lord used in {b['lord_used_games']}/{b['mixed_matches']} mixed games; "
          f"owner faction={b['lord_owner_faction']}")
    print(f"  winner_types={b['winner_types']}  draws={b['draws']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Population-18 balance audit")
    parser.add_argument("--matches-per-pair", type=int, default=40,
                        help="Experiment A matches per ordered faction pair (9 pairs)")
    parser.add_argument("--mixed-matches", type=int, default=400,
                        help="Experiment B random mixed 3v3 matches")
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--output", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.matches_per_pair = 3
        args.mixed_matches = 12

    rng = random.Random(args.seed)
    t0 = time.time()
    exp_a = experiment_a(args.matches_per_pair, args.seed, rng)
    exp_b = experiment_b(args.mixed_matches, args.seed + 500_000, rng)
    elapsed = round(time.time() - t0, 1)

    total = args.matches_per_pair * len(FACTION_TYPES) ** 2 + args.mixed_matches
    report = {
        "meta": {"seed": args.seed, "elapsed_s": elapsed, "total_matches": total,
                 "matches_per_pair": args.matches_per_pair, "mixed_matches": args.mixed_matches},
        "experiment_a_faction_h2h": exp_a,
        "experiment_b_per_net": exp_b,
    }
    print_report(report)

    out = Path(args.output) if args.output else Path("analysis") / f"population_balance_seed_{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved JSON: {out}  (elapsed {elapsed}s)")


if __name__ == "__main__":
    main()
