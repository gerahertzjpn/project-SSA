from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.analyze_random_lord_stats import aggregate as aggregate_lord_matches
from scripts.analyze_random_lord_stats import run_one as run_lord_trace_match
from src.core.combat import build_baron_unit, build_building_unit, build_core_unit, build_neutral_units, build_player_units
from src.core.combat2d import (
    COMBAT_RANGE_TO_MAP_UNITS,
    tick_battle_units_2d,
    unit_range_map,
)
from src.core.constants import (
    BARON_ATTACK_RATE,
    BARON_ATK,
    BARON_HP,
    BARON_SECONDARY_ATTACK_RATE,
    BARON_SECONDARY_ATK,
    BUILDING_BUILD_TICKS,
    BUILDING_COST,
    BUILDING_STATS,
    CORE_ATK,
    CORE_ATTACK_RATE,
    CORE_ATTACK_RANGE,
    CORE_HP,
    DEATH_TIME_START_TICK,
    PLAYER_IDS,
    RESOURCE_PER_TICK,
    TEAM_A_PLAYERS,
    TEAM_B_PLAYERS,
    TICK_LIMIT,
    UNIT_COST,
    BuildingType,
    UnitType,
    death_time_core_damage_for_tick,
)
from src.core.map import ADJACENCY
from src.core.map2d import (
    CORE_COMBAT_RADIUS,
    CORE_THREAT_RADIUS,
    LORD_THREAT_RADIUS,
    MOVEMENT_MAP_UNITS_PER_TICK,
)
from src.core.runner import current_code_commit, make_bots, rules_meta, run_match
from src.core.state import make_initial_state
from src.core.engine import GameEngine
from src.core.units import NEUTRAL_UNIT_BASE, UNIT_STATS


UNIT_TYPES = tuple(UNIT_COST.keys())
UNIT_LABELS = {unit: unit.value for unit in UNIT_TYPES}


def main() -> None:
    parser = argparse.ArgumentParser(description="Current 2D balance audit suite.")
    parser.add_argument("--matches", type=int, default=60, help="Headless random-vs-random matches.")
    parser.add_argument("--lord-matches", type=int, default=20, help="Trace matches for detailed Lord stats.")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    report = build_report(args.seed, args.matches, args.lord_matches)
    print_human_report(report)

    output = Path(args.output) if args.output else Path("analysis") / f"current_balance_suite_seed_{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved JSON: {output}")


def build_report(seed: int, matches: int, lord_matches: int) -> dict[str, Any]:
    return {
        "meta": {
            "seed": seed,
            "commit": current_code_commit(),
            "rules": rules_meta(),
        },
        "constants": constants_snapshot(),
        "micro": {
            "equal_budget_50": equal_budget_matrix(50),
            "budget_curve": budget_curve([20, 30, 40, 50, 60]),
            "objectives": objective_tests([20, 30, 40, 50, 60]),
            "building_objectives": building_objective_tests([30, 50, 80, 120]),
        },
        "macro_random": random_match_stats(seed, matches),
        "macro_factions": faction_match_stats(seed + 20_000, max(1, matches // 2)),
        "lord_random": lord_stats(seed + 10_000, lord_matches),
    }


def constants_snapshot() -> dict[str, Any]:
    death_time_damage = [death_time_core_damage_for_tick(tick) for tick in range(DEATH_TIME_START_TICK, TICK_LIMIT)]
    return {
        "tick_limit": TICK_LIMIT,
        "death_time_start": DEATH_TIME_START_TICK,
        "death_time_damage": {
            "ticks": len(death_time_damage),
            "total": sum(death_time_damage),
            "min_tick_damage": min(death_time_damage) if death_time_damage else 0,
            "max_tick_damage": max(death_time_damage) if death_time_damage else 0,
        },
        "economy": {
            "resource_per_tick": RESOURCE_PER_TICK,
            "unit_cost": {UNIT_LABELS[unit]: int(UNIT_COST[unit]) for unit in UNIT_TYPES},
        },
        "buildings": {
            building.value: {
                **dict(BUILDING_STATS[building]),
                "cost": int(BUILDING_COST[building]),
                "build_ticks": int(BUILDING_BUILD_TICKS[building]),
            }
            for building in BuildingType
        },
        "units": {
            UNIT_LABELS[unit]: {
                **dict(UNIT_STATS[unit]),
                "cost": int(UNIT_COST[unit]),
                "range_map_units": unit_stat_range_map(unit),
                "move_map_units_per_tick": MOVEMENT_MAP_UNITS_PER_TICK * float(UNIT_STATS[unit]["speed"]),
            }
            for unit in UNIT_TYPES
        },
        "neutral": dict(NEUTRAL_UNIT_BASE),
        "lord": {
            "hp": BARON_HP,
            "atk": BARON_ATK,
            "attack_rate": BARON_ATTACK_RATE,
            "secondary_atk": BARON_SECONDARY_ATK,
            "secondary_attack_rate": BARON_SECONDARY_ATTACK_RATE,
            "range": UNIT_STATS[UnitType.LORD]["range"],
            "range_map_units": unit_stat_range_map(UnitType.LORD),
            "move_map_units_per_tick": MOVEMENT_MAP_UNITS_PER_TICK * float(UNIT_STATS[UnitType.LORD]["speed"]),
        },
        "core": {
            "hp": CORE_HP,
            "atk": CORE_ATK,
            "attack_rate": CORE_ATTACK_RATE,
            "range": CORE_ATTACK_RANGE,
            "range_map_units": CORE_ATTACK_RANGE * COMBAT_RANGE_TO_MAP_UNITS,
        },
        "event_radii": {
            "core_combat_radius": CORE_COMBAT_RADIUS,
            "core_threat_radius": CORE_THREAT_RADIUS,
            "lord_threat_radius": LORD_THREAT_RADIUS,
        },
        "map": {
            "movement_map_units_per_tick": MOVEMENT_MAP_UNITS_PER_TICK,
            "combat_range_to_map_units": COMBAT_RANGE_TO_MAP_UNITS,
            "adjacency_degree": {node: len(neighbors) for node, neighbors in ADJACENCY.items()},
        },
    }


def unit_stat_range_map(unit_type: UnitType) -> float:
    unit = build_player_units({unit_type.value: 1}, "A1", "attacker")[0]
    return round(unit_range_map(unit), 3)


def equal_budget_matrix(budget: int) -> list[dict[str, Any]]:
    rows = []
    for left in UNIT_TYPES:
        for right in UNIT_TYPES:
            if left == right:
                continue
            rows.append(run_unit_duel(left, right, budget))
    return rows


def budget_curve(budgets: list[int]) -> list[dict[str, Any]]:
    rows = []
    for budget in budgets:
        for left, right in (
            (UnitType.SKIRMISHER, UnitType.RANGER),
            (UnitType.VANGUARD, UnitType.SKIRMISHER),
            (UnitType.RANGER, UnitType.VANGUARD),
            (UnitType.RAIDER, UnitType.VANGUARD),
            (UnitType.RAIDER, UnitType.RANGER),
            (UnitType.RAIDER, UnitType.SKIRMISHER),
            (UnitType.BMP, UnitType.RAIDER),
            (UnitType.BMP, UnitType.RANGER),
            (UnitType.BMP, UnitType.SKIRMISHER),
            (UnitType.BMP, UnitType.VANGUARD),
            (UnitType.RAVAGER, UnitType.VANGUARD),
            (UnitType.RAVAGER, UnitType.RANGER),
            (UnitType.RAVAGER, UnitType.SKIRMISHER),
            (UnitType.BEHEMOTH, UnitType.RAIDER),
            (UnitType.BEHEMOTH, UnitType.BMP),
            (UnitType.GUARDIAN_MECH, UnitType.BRUTE),
            (UnitType.GUARDIAN_MECH, UnitType.RAVAGER),
            (UnitType.SIEGE_PLATFORM, UnitType.BEHEMOTH),
            (UnitType.SIEGE_PLATFORM, UnitType.BMP),
        ):
            rows.append(run_unit_duel(left, right, budget))
    return rows


def objective_tests(budgets: list[int]) -> list[dict[str, Any]]:
    rows = []
    for budget in budgets:
        for attacker in UNIT_TYPES:
            rows.append(run_objective(attacker, budget, "neutral_6"))
            rows.append(run_objective(attacker, budget, "neutral_10"))
            rows.append(run_objective(attacker, budget, "lord"))
            rows.append(run_objective(attacker, budget, "core"))
    return rows


def building_objective_tests(budgets: list[int]) -> list[dict[str, Any]]:
    rows = []
    for budget in budgets:
        for attacker in UNIT_TYPES:
            rows.append(run_building_objective(attacker, budget, [BuildingType.TURRET], label="turret"))
            rows.append(run_building_objective(attacker, budget, [BuildingType.TURRET], include_core=True, label="turret_core"))
            rows.append(run_building_objective(attacker, budget, [BuildingType.TECH_2, BuildingType.TECH_1], include_core=True, label="two_base_buildings_core"))
    return rows


def run_building_objective(
    attacker_type: UnitType,
    budget: int,
    buildings: list[BuildingType],
    *,
    include_core: bool = False,
    label: str,
    max_ticks: int = 120,
) -> dict[str, Any]:
    attacker_count = budget // int(UNIT_COST[attacker_type])
    attackers = build_player_units({attacker_type.value: attacker_count}, "A1", "attacker")
    defenders = [
        build_building_unit(building, "B1", hp=float(BUILDING_STATS[building]["hp"]))
        for building in buildings
    ]
    if include_core:
        defenders.append(build_core_unit("B1", hp=CORE_HP))
    units = attackers + defenders
    place_line(units, "A1", "B1", left_x=170.0, right_x=440.0)
    result = simulate_units(units, neutral_aggro_teams=set(), max_ticks=max_ticks)
    return {
        "kind": "building_objective",
        "budget": budget,
        "attacker": army_label(attacker_type, attacker_count),
        "objective": label,
        **result,
    }


def run_unit_duel(left_type: UnitType, right_type: UnitType, budget: int, max_ticks: int = 80) -> dict[str, Any]:
    left_count = budget // int(UNIT_COST[left_type])
    right_count = budget // int(UNIT_COST[right_type])
    left_units = build_player_units({left_type.value: left_count}, "A1", "attacker")
    right_units = build_player_units({right_type.value: right_count}, "B1", "defender")
    units = left_units + right_units
    place_line(units, "A1", "B1")
    result = simulate_units(units, neutral_aggro_teams=set(), max_ticks=max_ticks)
    return {
        "kind": "unit_duel",
        "budget": budget,
        "left": army_label(left_type, left_count),
        "right": army_label(right_type, right_count),
        **result,
    }


def run_objective(attacker_type: UnitType, budget: int, objective: str, max_ticks: int = 100) -> dict[str, Any]:
    attacker_count = budget // int(UNIT_COST[attacker_type])
    attackers = build_player_units({attacker_type.value: attacker_count}, "A1", "attacker")
    if objective == "neutral_6":
        defenders = build_neutral_units(6)
        aggro = {"TEAM_A"}
    elif objective == "neutral_10":
        defenders = build_neutral_units(10)
        aggro = {"TEAM_A"}
    elif objective == "lord":
        defenders = [build_baron_unit(BARON_HP)]
        aggro = set()
    elif objective == "core":
        defenders = [build_core_unit("B1", hp=CORE_HP)]
        aggro = set()
    else:
        raise ValueError(objective)

    units = attackers + defenders
    place_line(units, "A1", "B1", left_x=170.0, right_x=440.0)
    result = simulate_units(units, neutral_aggro_teams=aggro, max_ticks=max_ticks)
    return {
        "kind": "objective",
        "budget": budget,
        "attacker": army_label(attacker_type, attacker_count),
        "objective": objective,
        **result,
    }


def simulate_units(units, neutral_aggro_teams: set[str], max_ticks: int) -> dict[str, Any]:
    ticks = 0
    last_attacks = 0
    while ticks < max_ticks and hostile_sides_alive(units, neutral_aggro_teams):
        frame = tick_battle_units_2d(
            units,
            record_trace=True,
            neutral_aggro_teams=neutral_aggro_teams,
            battle=None,
        )
        last_attacks = sum(len(row.get("attacks") or []) for row in frame)
        ticks += 1

    alive = summarize_alive(units)
    return {
        "winner": infer_winner(alive),
        "ticks": ticks,
        "last_tick_attacks": last_attacks,
        "alive": alive,
        "total_hp": {
            side: round(sum(unit.hp for unit in units if side_of(unit.team) == side and unit.is_alive()), 3)
            for side in ("TEAM_A", "TEAM_B", "NEUTRAL", "BARON")
        },
    }


def place_line(units, left_team: str, right_team: str, left_x: float = 220.0, right_x: float = 460.0) -> None:
    buckets: dict[str, list] = {left_team: [], right_team: []}
    for unit in units:
        if side_of(unit.team) == "TEAM_A":
            buckets[left_team].append(unit)
        else:
            buckets[right_team].append(unit)

    assign_formation(buckets[left_team], base_x=left_x, base_y=205.0, forward_x=1.0)
    assign_formation(buckets[right_team], base_x=right_x, base_y=205.0, forward_x=-1.0)


def assign_formation(units, base_x: float, base_y: float, forward_x: float) -> None:
    by_type = {}
    for unit in units:
        by_type.setdefault(unit.unit_type, []).append(unit)
    front_offsets = {
        UnitType.VANGUARD.value: 20.0,
        UnitType.SKIRMISHER.value: 11.0,
        UnitType.RAIDER.value: 13.0,
        UnitType.RANGER.value: 2.0,
        UnitType.BMP.value: 0.0,
        UnitType.SWARMLING.value: 20.0,
        UnitType.SPITTER.value: 4.0,
        UnitType.BRUTE.value: 22.0,
        UnitType.RAVAGER.value: 14.0,
        UnitType.BEHEMOTH.value: 24.0,
        UnitType.WALKER.value: 22.0,
        UnitType.LASER.value: 2.0,
        UnitType.DRONE.value: 18.0,
        UnitType.GUARDIAN_MECH.value: 18.0,
        UnitType.SIEGE_PLATFORM.value: 0.0,
        "NEUTRAL": 0.0,
        "BARON": 0.0,
        "CORE": 0.0,
        BuildingType.TURRET.value: 0.0,
        BuildingType.BOOSTER.value: 0.0,
        BuildingType.POWER.value: 0.0,
        BuildingType.LIFE.value: 0.0,
        BuildingType.ECONOMY_COLLECTOR.value: 0.0,
        BuildingType.TECH_1.value: 0.0,
        BuildingType.TECH_2.value: 0.0,
        UnitType.LORD.value: 22.0,
    }
    for unit_type in (
        tuple(unit.value for unit in UNIT_TYPES)
        + tuple(building.value for building in BuildingType)
        + (UnitType.LORD.value, "BARON", "CORE", "NEUTRAL")
    ):
        bucket = by_type.get(unit_type, [])
        visible_rows = min(9, max(1, len(bucket)))
        for local_index, unit in enumerate(bucket):
            row = local_index % 9
            col = local_index // 9
            y = base_y + (row - (visible_rows - 1) / 2.0) * 7.0
            x = base_x + forward_x * (front_offsets.get(unit.unit_type, 0.0) - col * 8.0)
            unit.x = x
            unit.y = y
            unit.forward_x = forward_x
            unit.forward_y = 0.0


def hostile_sides_alive(units, neutral_aggro_teams: set[str]) -> bool:
    alive_sides = set()
    for unit in units:
        if not unit.is_alive():
            continue
        side = side_of(unit.team)
        if side in {"NEUTRAL"} and not neutral_aggro_teams:
            continue
        alive_sides.add(side)
    return len(alive_sides) > 1


def summarize_alive(units) -> dict[str, dict[str, int]]:
    alive: dict[str, Counter] = {}
    for unit in units:
        if not unit.is_alive():
            continue
        side = side_of(unit.team)
        alive.setdefault(side, Counter())[unit.unit_type] += 1
    return {side: dict(sorted(counter.items())) for side, counter in sorted(alive.items())}


def infer_winner(alive: dict[str, dict[str, int]]) -> str:
    sides = [side for side, units in alive.items() if sum(units.values()) > 0]
    if len(sides) == 1:
        return sides[0]
    if not sides:
        return "MUTUAL_DEATH"
    return "TIMEOUT_OR_MIXED"


def side_of(team: str) -> str:
    if team in {"A1", "A2", "A3"}:
        return "TEAM_A"
    if team in {"B1", "B2", "B3"}:
        return "TEAM_B"
    return team


def army_label(unit_type: UnitType, count: int) -> dict[str, Any]:
    return {
        "type": UNIT_LABELS[unit_type],
        "count": count,
        "cost": count * int(UNIT_COST[unit_type]),
    }


def random_match_stats(seed: int, matches: int) -> dict[str, Any]:
    winners = Counter()
    winner_types = Counter()
    ticks = []
    final_core_hp = []
    for index in range(matches):
        result = run_match(seed + index, "random", mode="training")
        winners[result.winner] += 1
        winner_types[result.winner_type or "unknown"] += 1
        ticks.append(result.total_ticks)
        final_core_hp.append({
            player_id: round(float(result.state.players[player_id].core_hp), 3)
            for player_id in PLAYER_IDS
        })
    return {
        "matches": matches,
        "winners": dict(winners),
        "winner_types": dict(winner_types),
        "ticks": numeric_stats(ticks),
        "sample_final_core_hp": final_core_hp[:5],
    }


def _factions_for_setup(setup: str) -> dict[str, str]:
    if setup == "all_humans":
        return {}
    if setup == "all_infected":
        return {player_id: "infected" for player_id in PLAYER_IDS}
    if setup == "all_robots":
        return {player_id: "robots" for player_id in PLAYER_IDS}
    if setup == "humans_vs_infected":
        return {player_id: ("humans" if player_id in TEAM_A_PLAYERS else "infected") for player_id in PLAYER_IDS}
    if setup == "infected_vs_humans":
        return {player_id: ("infected" if player_id in TEAM_A_PLAYERS else "humans") for player_id in PLAYER_IDS}
    if setup == "humans_vs_robots":
        return {player_id: ("humans" if player_id in TEAM_A_PLAYERS else "robots") for player_id in PLAYER_IDS}
    if setup == "robots_vs_humans":
        return {player_id: ("robots" if player_id in TEAM_A_PLAYERS else "humans") for player_id in PLAYER_IDS}
    if setup == "infected_vs_robots":
        return {player_id: ("infected" if player_id in TEAM_A_PLAYERS else "robots") for player_id in PLAYER_IDS}
    if setup == "robots_vs_infected":
        return {player_id: ("robots" if player_id in TEAM_A_PLAYERS else "infected") for player_id in PLAYER_IDS}
    if setup == "mixed_slots":
        factions = ("humans", "infected", "robots")
        return {
            player_id: factions[index % len(factions)]
            for index, player_id in enumerate(PLAYER_IDS)
        }
    raise ValueError(setup)


def faction_match_stats(seed: int, matches: int) -> dict[str, Any]:
    setups = (
        "all_humans",
        "all_infected",
        "all_robots",
        "humans_vs_infected",
        "infected_vs_humans",
        "humans_vs_robots",
        "robots_vs_humans",
        "infected_vs_robots",
        "robots_vs_infected",
        "mixed_slots",
    )
    bot_types = ("random", "scripted")
    rows = {}
    for bot_type in bot_types:
        for setup_index, setup in enumerate(setups):
            winners = Counter()
            winner_types = Counter()
            ticks = []
            final_core_team_hp = []
            for match_index in range(matches):
                match_seed = seed + setup_index * 10_000 + (0 if bot_type == "random" else 100_000) + match_index
                state = make_initial_state(seed=match_seed, player_factions=_factions_for_setup(setup))
                engine = GameEngine(
                    state,
                    make_bots(bot_type, seed_offset=match_seed * 10),
                    record_trace=False,
                    record_replay=False,
                )
                winner = engine.run()
                winners[winner] += 1
                winner_types[state.winner_type or "unknown"] += 1
                ticks.append(state.tick)
                final_core_team_hp.append({
                    "TEAM_A": round(sum(float(state.players[player_id].core_hp) for player_id in TEAM_A_PLAYERS), 3),
                    "TEAM_B": round(sum(float(state.players[player_id].core_hp) for player_id in TEAM_B_PLAYERS), 3),
                })
            rows[f"{bot_type}__{setup}"] = {
                "matches": matches,
                "winners": dict(winners),
                "winner_types": dict(winner_types),
                "ticks": numeric_stats(ticks),
                "sample_final_core_team_hp": final_core_team_hp[:5],
            }
    return rows


def lord_stats(seed: int, matches: int) -> dict[str, Any]:
    rows = [run_lord_trace_match(seed + index) for index in range(matches)]
    return {
        "matches": matches,
        "aggregate": aggregate_lord_matches(rows),
        "sample": rows[:3],
    }


def numeric_stats(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 3),
        "mean": round(statistics.mean(ordered), 3),
        "median": round(statistics.median(ordered), 3),
        "max": round(ordered[-1], 3),
    }


def print_human_report(report: dict[str, Any]) -> None:
    constants = report["constants"]
    print("=== Current Balance Suite ===")
    print(f"commit: {report['meta']['commit']}")
    print("units:")
    for unit, data in constants["units"].items():
        print(
            f"  {unit}: hp={data['hp']} atk={data['atk']} range={data['range']} "
            f"speed={data['speed']} cost={data['cost']} map_speed={data['move_map_units_per_tick']}"
        )
    print(
        f"lord: hp={constants['lord']['hp']} "
        f"atk={constants['lord']['atk']}x{constants['lord']['attack_rate']}+"
        f"{constants['lord']['secondary_atk']}x{constants['lord']['secondary_attack_rate']} "
        f"range_map={constants['lord']['range_map_units']}"
    )
    print(f"core: hp={constants['core']['hp']} atk={constants['core']['atk']} range_map={constants['core']['range_map_units']}")
    print(f"economy: +{constants['economy']['resource_per_tick']} resources/tick")
    print("buildings:")
    for building, data in constants["buildings"].items():
        print(
            f"  {building}: hp={data['hp']} atk={data['atk']} range={data['range']} "
            f"income={data['income']} cost={data['cost']} build={data['build_ticks']}t"
        )

    print("\nunit duel budget 50:")
    for row in report["micro"]["equal_budget_50"]:
        print(f"  {row['left']['count']} {row['left']['type']} vs {row['right']['count']} {row['right']['type']}: {row['winner']} in {row['ticks']}t alive={row['alive']}")

    print("\nrandom macro:")
    macro = report["macro_random"]
    print(f"  winners={macro['winners']} winner_types={macro['winner_types']} ticks={macro['ticks']}")

    print("\nfaction macro:")
    for name, data in report["macro_factions"].items():
        print(f"  {name}: winners={data['winners']} winner_types={data['winner_types']} ticks={data['ticks']}")

    print("\nlord trace aggregate:")
    aggregate = report["lord_random"]["aggregate"]
    lord = aggregate.get("lord", {})
    print(json.dumps({
        "matches": aggregate.get("matches"),
        "winner_counts": aggregate.get("winner_counts"),
        "winner_type_counts": aggregate.get("winner_type_counts"),
        "neutral_lord_killed": lord.get("neutral_lord_killed"),
        "owner_players": lord.get("owner_players"),
        "owner_teams": lord.get("owner_teams"),
        "first_kill_tick": lord.get("first_kill_tick"),
        "core_expended": lord.get("core_expended"),
        "died_before_core": lord.get("died_before_core"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
