from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.analyze_random_lord_stats import aggregate as aggregate_lord_stats
from scripts.analyze_random_lord_stats import analyze_match
from src.core.combat import (
    build_baron_unit,
    build_core_unit,
    build_neutral_units,
    build_player_units,
    count_survivors_by_player,
    has_hostile_units,
    tick_battle_units,
)
from src.core.constants import (
    BARON_HP,
    CORE_HP,
    MAX_COMBAT_TICKS,
    NEUTRAL_GUARD_HP,
    PLAYER_IDS,
    TEAM_A_PLAYERS,
    TEAM_B_PLAYERS,
    UNIT_COST,
    UnitType,
)
from src.core.runner import current_code_commit, make_bots, rules_meta
from src.core.state import get_team, make_initial_state
from src.core.units import UNIT_STATS
from src.core import engine as engine_module
from src.core.engine import GameEngine


UNIT_TYPES = (UnitType.VANGUARD, UnitType.RANGER, UnitType.SKIRMISHER)
UNIT_KEYS = tuple(unit.value for unit in UNIT_TYPES)
SPECIAL_TYPES = {"NEUTRAL", "BARON", "CORE"}


def unit_cost(units: dict[str, int]) -> int:
    total = 0
    for unit_type, count in (units or {}).items():
        if int(count or 0) <= 0:
            continue
        key = UnitType.RANGER if unit_type == "ARCHER" else UnitType(unit_type)
        if key not in UNIT_COST:
            continue
        total += int(count) * int(UNIT_COST[key])
    return total


def normalize_units(units: dict[str | UnitType, int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for unit_type, count in (units or {}).items():
        value = unit_type.value if isinstance(unit_type, UnitType) else str(unit_type)
        if value == "ARCHER":
            value = UnitType.RANGER.value
        if int(count or 0) > 0:
            result[value] = result.get(value, 0) + int(count)
    return dict(sorted(result.items()))


def sub_units(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    result = {}
    for unit_type in UNIT_KEYS:
        value = int(left.get(unit_type, 0) or 0) - int(right.get(unit_type, 0) or 0)
        if value:
            result[unit_type] = value
    return result


def numeric_stats(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 3),
        "median": round(statistics.median(ordered), 3),
        "mean": round(statistics.mean(ordered), 3),
        "max": round(ordered[-1], 3),
    }


def counter_dict(counter: Counter) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: str(item[0])))


def resolution_side(unit) -> str:
    if unit.unit_type in SPECIAL_TYPES:
        return unit.unit_type
    return get_team(unit.team)


def force_timeout_resolution(units: list) -> None:
    hp_by_side: dict[str, float] = {}
    for unit in units:
        if unit.is_alive():
            side = resolution_side(unit)
            hp_by_side[side] = hp_by_side.get(side, 0.0) + float(unit.hp)
    if not hp_by_side:
        return
    winner_side = max(hp_by_side, key=hp_by_side.get)
    for unit in units:
        if unit.is_alive() and resolution_side(unit) != winner_side:
            unit.hp = 0.0


def simulate_battle(
    attacker_units: dict[str | UnitType, int],
    *,
    defender_units: dict[str | UnitType, int] | None = None,
    objective: str = "pvp",
    neutral_hp: int = 0,
    baron_hp: float = BARON_HP,
    core_hp: float = CORE_HP,
) -> dict[str, Any]:
    attacker_units = normalize_units(attacker_units)
    defender_units = normalize_units(defender_units or {})
    units = build_player_units(attacker_units, "A1", "attacker")

    if objective == "pvp":
        units.extend(build_player_units(defender_units, "B1", "defender"))
    elif objective == "neutral":
        units.extend(build_neutral_units(int(neutral_hp)))
    elif objective == "baron":
        units.append(build_baron_unit(baron_hp))
    elif objective == "core":
        units.extend(build_player_units(defender_units, "B1", "defender"))
        units.append(build_core_unit("B1", hp=float(core_hp)))
    else:
        raise ValueError(f"Unknown objective: {objective}")

    neutral_aggro_teams = {"TEAM_A"} if objective == "neutral" else set()
    ticks = 0
    for _ in range(MAX_COMBAT_TICKS):
        if not has_hostile_units(units, neutral_aggro_teams):
            break
        tick_battle_units(
            units,
            record_trace=False,
            neutral_aggro_teams=neutral_aggro_teams,
        )
        ticks += 1

    timed_out = has_hostile_units(units, neutral_aggro_teams)
    if timed_out:
        force_timeout_resolution(units)

    survivors_by_player = count_survivors_by_player(units)
    attacker_survivors = normalize_units(survivors_by_player.get("A1", {}))
    defender_survivors = normalize_units(survivors_by_player.get("B1", {}))
    special_hp = {
        special: round(
            sum(max(0.0, float(unit.hp)) for unit in units if unit.unit_type == special and unit.is_alive()),
            3,
        )
        for special in SPECIAL_TYPES
    }
    special_alive = {key: value > 0 for key, value in special_hp.items()}

    if objective == "neutral":
        success = bool(attacker_survivors) and not special_alive["NEUTRAL"]
        winner = "A1" if success else "NEUTRAL"
    elif objective == "baron":
        success = bool(attacker_survivors) and not special_alive["BARON"]
        winner = "A1" if success else "BARON"
    elif objective == "core":
        success = not special_alive["CORE"]
        winner = "A1" if success and attacker_survivors else "B1"
    else:
        success = bool(attacker_survivors) and not defender_survivors
        if attacker_survivors and not defender_survivors:
            winner = "A1"
        elif defender_survivors and not attacker_survivors:
            winner = "B1"
        elif attacker_survivors and defender_survivors:
            winner = "UNRESOLVED"
        else:
            winner = "NONE"

    return {
        "objective": objective,
        "winner": winner,
        "success": bool(success),
        "ticks": ticks,
        "timed_out": bool(timed_out),
        "attacker": attacker_units,
        "attacker_cost": unit_cost(attacker_units),
        "attacker_survivors": attacker_survivors,
        "attacker_losses": sub_units(attacker_units, attacker_survivors),
        "attacker_loss_cost": unit_cost(sub_units(attacker_units, attacker_survivors)),
        "defender": defender_units,
        "defender_survivors": defender_survivors,
        "defender_losses": sub_units(defender_units, defender_survivors),
        "special_hp": special_hp,
    }


def pure_army(unit_type: UnitType, budget: int) -> dict[str, int]:
    count = budget // int(UNIT_COST[unit_type])
    return {unit_type.value: count} if count > 0 else {}


def rps_matrix() -> list[dict[str, Any]]:
    rows = []
    for budget in (12, 24, 36, 48, 60):
        for attacker in UNIT_TYPES:
            for defender in UNIT_TYPES:
                result = simulate_battle(
                    pure_army(attacker, budget),
                    defender_units=pure_army(defender, budget),
                    objective="pvp",
                )
                rows.append({
                    "budget": budget,
                    "attacker_type": attacker.value,
                    "defender_type": defender.value,
                    "attacker_count": result["attacker"].get(attacker.value, 0),
                    "defender_count": result["defender"].get(defender.value, 0),
                    "winner": result["winner"],
                    "ticks": result["ticks"],
                    "attacker_survivors": result["attacker_survivors"],
                    "defender_survivors": result["defender_survivors"],
                    "attacker_loss_cost": result["attacker_loss_cost"],
                    "defender_loss_cost": unit_cost(result["defender_losses"]),
                })
    return rows


def unit_compositions_exact_cost(cost: int) -> list[dict[str, int]]:
    result = []
    for vanguard in range(cost // int(UNIT_COST[UnitType.VANGUARD]) + 1):
        for ranger in range(cost // int(UNIT_COST[UnitType.RANGER]) + 1):
            remaining = cost - vanguard * int(UNIT_COST[UnitType.VANGUARD]) - ranger * int(UNIT_COST[UnitType.RANGER])
            if remaining < 0:
                continue
            skirmisher_cost = int(UNIT_COST[UnitType.SKIRMISHER])
            if remaining % skirmisher_cost != 0:
                continue
            skirmisher = remaining // skirmisher_cost
            units = normalize_units({
                UnitType.VANGUARD.value: vanguard,
                UnitType.RANGER.value: ranger,
                UnitType.SKIRMISHER.value: skirmisher,
            })
            if units:
                result.append(units)
    result.sort(key=lambda item: (sum(item.values()), item.get(UnitType.RANGER.value, 0)))
    return result


def find_minimal_successes(
    *,
    objective: str,
    max_cost: int,
    limit: int = 8,
    defender_units: dict[str, int] | None = None,
    neutral_hp: int = 0,
    baron_hp: float = BARON_HP,
    core_hp: float = CORE_HP,
) -> list[dict[str, Any]]:
    successes = []
    for cost in range(1, max_cost + 1):
        for units in unit_compositions_exact_cost(cost):
            result = simulate_battle(
                units,
                defender_units=defender_units,
                objective=objective,
                neutral_hp=neutral_hp,
                baron_hp=baron_hp,
                core_hp=core_hp,
            )
            if result["success"]:
                successes.append(result)
                if len(successes) >= limit:
                    return successes
        if successes:
            return successes
    return successes


def neutral_clear_table() -> list[dict[str, Any]]:
    rows = []
    for strength in range(1, 8):
        income = neutral_income_for_strength(strength)
        successes = find_minimal_successes(objective="neutral", neutral_hp=strength, max_cost=90, limit=6)
        for result in successes:
            rows.append({
                "neutral_strength": strength,
                "income": income,
                "army": result["attacker"],
                "army_cost": result["attacker_cost"],
                "survivors": result["attacker_survivors"],
                "losses": result["attacker_losses"],
                "loss_cost": result["attacker_loss_cost"],
                "loss_payback_ticks": round(result["attacker_loss_cost"] / max(1, income), 3),
                "full_force_payback_ticks": round(result["attacker_cost"] / max(1, income), 3),
                "ticks": result["ticks"],
            })
    return rows


def neutral_income_for_strength(strength: int) -> int:
    return max(1, (int(strength) + 1) // 2)


def raid_sequence(army: dict[str, int], *, target: str, max_raids: int = 8) -> dict[str, Any]:
    hp = float(BARON_HP if target == "baron" else CORE_HP)
    history = []
    total_loss_cost = 0
    for raid in range(1, max_raids + 1):
        result = simulate_battle(
            army,
            objective=target,
            baron_hp=hp,
            core_hp=hp,
        )
        total_loss_cost += result["attacker_loss_cost"]
        hp_key = "BARON" if target == "baron" else "CORE"
        hp = float(result["special_hp"][hp_key])
        history.append({
            "raid": raid,
            "success": result["success"],
            "remaining_hp": round(hp, 3),
            "losses": result["attacker_losses"],
            "survivors": result["attacker_survivors"],
            "loss_cost": result["attacker_loss_cost"],
            "ticks": result["ticks"],
        })
        if result["success"]:
            break
    return {
        "target": target,
        "army": normalize_units(army),
        "army_cost": unit_cost(normalize_units(army)),
        "raids_until_success": history[-1]["raid"] if history and history[-1]["success"] else None,
        "total_loss_cost": total_loss_cost,
        "history": history,
    }


def lord_and_core_tables() -> dict[str, Any]:
    sample_raids = [
        {UnitType.VANGUARD.value: 8},
        {UnitType.RANGER.value: 8},
        {UnitType.SKIRMISHER.value: 8},
        {UnitType.VANGUARD.value: 5, UnitType.RANGER.value: 5},
        {UnitType.VANGUARD.value: 6, UnitType.RANGER.value: 6, UnitType.SKIRMISHER.value: 3},
        {UnitType.VANGUARD.value: 10, UnitType.RANGER.value: 10},
        {UnitType.VANGUARD.value: 12, UnitType.SKIRMISHER.value: 12},
    ]
    core_defenses = {
        "empty": {},
        "5_vanguard": {UnitType.VANGUARD.value: 5},
        "5_ranger": {UnitType.RANGER.value: 5},
        "5_skirmisher": {UnitType.SKIRMISHER.value: 5},
        "mixed_3_each": {
            UnitType.VANGUARD.value: 3,
            UnitType.RANGER.value: 3,
            UnitType.SKIRMISHER.value: 3,
        },
    }
    return {
        "minimal_lord_kill_armies": find_minimal_successes(objective="baron", max_cost=240, limit=12),
        "lord_failed_raid_sequences": [raid_sequence(army, target="baron") for army in sample_raids],
        "minimal_core_kill_armies": {
            label: find_minimal_successes(
                objective="core",
                defender_units=defense,
                max_cost=220,
                limit=8,
            )
            for label, defense in core_defenses.items()
        },
        "core_raid_sequences": [raid_sequence(army, target="core") for army in sample_raids],
    }


def run_match_with_counters(request: dict[str, Any]) -> dict[str, Any]:
    seed = int(request["seed"])
    bot_type = str(request["bot_type"])
    state = make_initial_state(seed=seed)
    production_counts: Counter[str] = Counter()

    original_apply_production = engine_module.apply_production

    def counted_apply_production(state_arg, player_id, unit_type):
        applied = original_apply_production(state_arg, player_id, unit_type)
        if applied:
            production_counts[f"{player_id}:{unit_type.value}"] += 1
            production_counts[f"{get_team(player_id)}:{unit_type.value}"] += 1
            production_counts[f"ALL:{unit_type.value}"] += 1
        return applied

    engine_module.apply_production = counted_apply_production
    try:
        engine = GameEngine(
            state,
            make_bots(bot_type, seed_offset=seed * 10),
            record_trace=True,
            record_replay=True,
        )
        winner = engine.run()
    finally:
        engine_module.apply_production = original_apply_production

    match = analyze_match(seed, winner, state, engine.replay_ticks)
    match["bot_type"] = bot_type
    match["production_counts"] = counter_dict(production_counts)
    match["final_resources"] = {
        player_id: round(float(player.resources), 3)
        for player_id, player in sorted(state.players.items())
    }
    match["final_nodes_by_owner"] = Counter(node.owner for node in state.nodes.values())
    match["final_cp_income_by_owner"] = Counter()
    match["final_units_by_owner_type"] = Counter()
    for node_id, node in state.nodes.items():
        if node.neutral_income > 0 and node.neutral_hp <= 0 and node.owner in state.players:
            match["final_cp_income_by_owner"][node.owner] += int(node.neutral_income)
        for unit_type, count in (node.units or {}).items():
            match["final_units_by_owner_type"][f"{node.owner}:{unit_type}"] += int(count or 0)
        for visitor_id, units in (node.visiting_units or {}).items():
            for unit_type, count in (units or {}).items():
                match["final_units_by_owner_type"][f"{visitor_id}:{unit_type}"] += int(count or 0)
    match["final_nodes_by_owner"] = counter_dict(match["final_nodes_by_owner"])
    match["final_cp_income_by_owner"] = counter_dict(match["final_cp_income_by_owner"])
    match["final_units_by_owner_type"] = counter_dict(match["final_units_by_owner_type"])
    return match


def chunked(items: list[Any], workers: int) -> list[list[Any]]:
    workers = max(1, min(workers, len(items)))
    return [items[index::workers] for index in range(workers)]


def run_match_chunk(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run_match_with_counters(request) for request in requests]


def run_match_stats(bot_type: str, matches: int, seed: int, workers: int) -> dict[str, Any]:
    requests = [
        {"bot_type": bot_type, "seed": int(seed) + index}
        for index in range(int(matches))
    ]
    if workers <= 1:
        results = run_match_chunk(requests)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(int(workers), len(requests))) as pool:
            nested = pool.map(run_match_chunk, chunked(requests, workers))
        results = [item for chunk in nested for item in chunk]
        results.sort(key=lambda item: item["seed"])

    lord_summary = aggregate_lord_stats(results)
    production = Counter()
    final_nodes = Counter()
    final_income = Counter()
    final_units = Counter()
    final_resources = []
    for match in results:
        production.update(match["production_counts"])
        final_nodes.update(match["final_nodes_by_owner"])
        final_income.update(match["final_cp_income_by_owner"])
        final_units.update(match["final_units_by_owner_type"])
        final_resources.extend(match["final_resources"].values())

    return {
        "bot_type": bot_type,
        "matches": int(matches),
        "summary": lord_summary,
        "production_counts": counter_dict(production),
        "production_per_match": {
            key: round(value / max(1, int(matches)), 3)
            for key, value in sorted(production.items())
        },
        "final_nodes_by_owner": counter_dict(final_nodes),
        "final_cp_income_by_owner": counter_dict(final_income),
        "final_units_by_owner_type": counter_dict(final_units),
        "final_resources": numeric_stats(final_resources),
        "matches_detail": results,
    }


def constants_snapshot() -> dict[str, Any]:
    return {
        "unit_cost": {unit.value: UNIT_COST[unit] for unit in UNIT_TYPES},
        "unit_stats": {
            unit.value: dict(UNIT_STATS[unit])
            for unit in (UnitType.VANGUARD, UnitType.RANGER, UnitType.SKIRMISHER, UnitType.LORD)
        },
        "core_hp": CORE_HP,
        "baron_hp": BARON_HP,
        "neutral_guard_hp": NEUTRAL_GUARD_HP,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance audit for combat/economy before training")
    parser.add_argument("--matches", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--bot-types", default="random,scripted")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    started_at = time.perf_counter()
    bot_types = [item.strip() for item in str(args.bot_types).split(",") if item.strip()]
    match_stats = {
        bot_type: run_match_stats(bot_type, int(args.matches), int(args.seed), int(args.workers))
        for bot_type in bot_types
    }
    payload = {
        "kind": "balance_audit",
        "config": {
            "matches_per_bot_type": int(args.matches),
            "seed": int(args.seed),
            "workers": int(args.workers),
            "bot_types": bot_types,
            "code_commit": current_code_commit(),
            "rules": rules_meta(),
        },
        "constants": constants_snapshot(),
        "lab": {
            "rps_equal_budget": rps_matrix(),
            "neutral_clear": neutral_clear_table(),
            "lord_and_core": lord_and_core_tables(),
        },
        "match_stats": match_stats,
        "elapsed_sec": round(time.perf_counter() - started_at, 3),
    }

    out = Path(args.out or f"analysis/balance_audit_{int(args.seed)}_{int(args.matches)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)

    compact = {
        "kind": payload["kind"],
        "out": str(out),
        "elapsed_sec": payload["elapsed_sec"],
        "config": payload["config"],
        "constants": payload["constants"],
        "match_summary": {
            bot_type: {
                "winner_counts": stats["summary"]["winner_counts"],
                "winner_type_counts": stats["summary"]["winner_type_counts"],
                "ticks": stats["summary"]["ticks"],
                "lord": stats["summary"]["lord"],
                "production_per_match": stats["production_per_match"],
            }
            for bot_type, stats in match_stats.items()
        },
        "lab_excerpt": {
            "minimal_lord_kill_armies": payload["lab"]["lord_and_core"]["minimal_lord_kill_armies"][:5],
            "minimal_core_kill_empty": payload["lab"]["lord_and_core"]["minimal_core_kill_armies"]["empty"][:5],
            "neutral_clear_first_options": payload["lab"]["neutral_clear"][:14],
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
