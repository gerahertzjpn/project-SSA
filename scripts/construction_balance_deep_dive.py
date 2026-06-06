from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import src.core.engine as engine_module
from scripts.current_balance_suite import (
    army_label,
    assign_formation,
    run_unit_duel,
    simulate_units,
)
from src.bots.random_bot import RandomBot
from src.bots.scripted_bot import ScriptedBot
from src.core.combat import build_building_unit, build_core_unit, build_player_units
from src.core.constants import (
    BUILDING_BUILD_TICKS,
    BUILDING_COST,
    BUILDING_STATS,
    CORE_HP,
    DEATH_TIME_START_TICK,
    PLAYER_IDS,
    RESOURCE_PER_TICK,
    TEAM_A_PLAYERS,
    TEAM_B_PLAYERS,
    UNIT_COST,
    unit_tech_requirements,
    ActionType,
    BuildingType,
    UnitType,
)
from src.core.engine import Action, GameEngine
from src.core.map import ADJACENCY, NODE_TYPES, NodeType
from src.core.runner import current_code_commit, rules_meta
from src.core.state import get_team, make_initial_state
from src.core.units import UNIT_STATS


BASIC_UNITS = (UnitType.VANGUARD, UnitType.RANGER, UnitType.SKIRMISHER)
ALL_UNITS = (UnitType.VANGUARD, UnitType.RANGER, UnitType.SKIRMISHER, UnitType.RAIDER, UnitType.BMP)


def numeric_stats(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 3),
        "p10": round(percentile(ordered, 0.10), 3),
        "median": round(statistics.median(ordered), 3),
        "mean": round(statistics.mean(ordered), 3),
        "p90": round(percentile(ordered, 0.90), 3),
        "max": round(ordered[-1], 3),
    }


def percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    q = max(0.0, min(1.0, float(q)))
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(len(ordered) - 1, low + 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def counter_dict(counter: Counter) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: str(item[0])))


def unit_total(units: dict) -> int:
    return sum(int(count or 0) for count in (units or {}).values())


@dataclass(frozen=True)
class StrategyConfig:
    build_plan: tuple[BuildingType, ...] = ()
    cp_turrets: bool = False
    produce_mode: str = "basic_mix"
    send_threshold: int = 7
    target_baron: bool = True
    use_multi: bool = True
    enemy_priority: bool = False


STRATEGIES: dict[str, StrategyConfig] = {
    "basic_mix": StrategyConfig(produce_mode="basic_mix", send_threshold=7, cp_turrets=False),
    "core_basic": StrategyConfig(produce_mode="basic_mix", send_threshold=7, cp_turrets=False, enemy_priority=True),
    "ranger_spam": StrategyConfig(produce_mode="ranger", send_threshold=6, cp_turrets=False),
    "skirm_flood": StrategyConfig(produce_mode="skirmisher", send_threshold=9, cp_turrets=False),
    "vanguard_push": StrategyConfig(produce_mode="vanguard", send_threshold=8, cp_turrets=False),
    "raider_eco": StrategyConfig(
        build_plan=(BuildingType.ECONOMY_COLLECTOR, BuildingType.TECH_1),
        produce_mode="raider",
        send_threshold=7,
        cp_turrets=True,
    ),
    "raider_fast": StrategyConfig(
        build_plan=(BuildingType.TECH_1,),
        produce_mode="raider",
        send_threshold=7,
        cp_turrets=False,
    ),
    "core_raider_fast": StrategyConfig(
        build_plan=(BuildingType.TECH_1,),
        produce_mode="raider",
        send_threshold=7,
        cp_turrets=False,
        enemy_priority=True,
    ),
    "bmp_rush": StrategyConfig(
        build_plan=(BuildingType.TECH_1, BuildingType.TECH_2),
        produce_mode="bmp",
        send_threshold=5,
        cp_turrets=False,
    ),
    "core_bmp_rush": StrategyConfig(
        build_plan=(BuildingType.TECH_1, BuildingType.TECH_2),
        produce_mode="bmp",
        send_threshold=5,
        cp_turrets=False,
        enemy_priority=True,
    ),
    "eco_turtle": StrategyConfig(
        build_plan=(BuildingType.ECONOMY_COLLECTOR, BuildingType.BOOSTER),
        produce_mode="ranger",
        send_threshold=8,
        cp_turrets=True,
    ),
}


class StrategyBot:
    def __init__(self, player_id: str, team: str, strategy: str):
        self.player_id = player_id
        self.team = team
        self.strategy = strategy
        self.config = STRATEGIES[strategy]

    def act(self, obs: dict) -> Action:
        steps: list[Action] = []

        move = self._move_action(obs)
        build = self._build_action(obs)
        produce = self._production_action(obs)

        if move is not None:
            steps.append(move)
        if build is not None:
            steps.append(build)
        if produce is not None:
            steps.append(produce)

        if not steps:
            return Action(ActionType.HOLD)
        if len(steps) == 1 or not self.config.use_multi:
            return steps[0]
        return Action(ActionType.MULTI, sub_actions=steps[:3])

    def _build_action(self, obs: dict) -> Optional[Action]:
        resources = float(obs.get("resources", 0.0) or 0.0)
        nodes = obs.get("nodes", {}) or {}
        core_node = obs.get("core_node")
        core = nodes.get(core_node, {}) or {}
        completed = self._completed_buildings(obs)

        core_buildings = [
            building for building in core.get("buildings", []) or []
            if building.get("owner") == self.player_id
        ]
        core_types = {building.get("building_type") for building in core_buildings}
        if len(core_buildings) < 2:
            for building_type in self.config.build_plan:
                if building_type.value in core_types:
                    continue
                if building_type == BuildingType.TECH_2 and BuildingType.TECH_1.value not in completed:
                    continue
                if resources >= BUILDING_COST[building_type]:
                    return Action(ActionType.BUILD, destination=core_node, building_type=building_type)

        if not self.config.cp_turrets:
            return None
        for node_id in sorted(nodes):
            node = nodes[node_id]
            if NODE_TYPES.get(node_id) != NodeType.CONTROL_POINT:
                continue
            if node.get("owner") != self.player_id or int(node.get("neutral_hp", 0) or 0) > 0:
                continue
            existing = [
                building for building in node.get("buildings", []) or []
                if building.get("owner") == self.player_id
            ]
            if not existing and resources >= BUILDING_COST[BuildingType.TURRET]:
                return Action(ActionType.BUILD, destination=node_id, building_type=BuildingType.TURRET)
        return None

    def _production_action(self, obs: dict) -> Optional[Action]:
        resources = float(obs.get("resources", 0.0) or 0.0)
        completed = self._completed_buildings(obs)
        mode = self.config.produce_mode

        if mode == "raider" and BuildingType.TECH_1.value not in completed:
            return self._basic_production(obs, resources, reserve=self._next_build_reserve(obs))
        if mode == "bmp" and any(
            required.value not in completed
            for required in unit_tech_requirements(UnitType.BMP)
        ):
            return self._basic_production(obs, resources, reserve=self._next_build_reserve(obs), cap_core_units=3)

        if mode == "raider":
            return self._produce_if_affordable(resources, UnitType.RAIDER, completed)
        if mode == "bmp":
            return self._produce_if_affordable(resources, UnitType.BMP, completed)
        if mode == "ranger":
            return self._produce_if_affordable(resources, UnitType.RANGER, completed)
        if mode == "skirmisher":
            return self._produce_if_affordable(resources, UnitType.SKIRMISHER, completed)
        if mode == "vanguard":
            return self._produce_if_affordable(resources, UnitType.VANGUARD, completed)
        return self._basic_production(obs, resources)

    def _basic_production(
        self,
        obs: dict,
        resources: float,
        reserve: float = 0.0,
        cap_core_units: Optional[int] = None,
    ) -> Optional[Action]:
        core_node = obs.get("core_node")
        core_units = (obs.get("nodes", {}) or {}).get(core_node, {}).get("units", {}) or {}
        if cap_core_units is not None and unit_total(core_units) >= cap_core_units:
            return None
        spendable = resources - float(reserve or 0.0)
        if spendable < min(UNIT_COST[unit] for unit in BASIC_UNITS):
            return None
        counts = {unit: int(core_units.get(unit.value, 0) or 0) for unit in BASIC_UNITS}
        order = sorted(BASIC_UNITS, key=lambda unit: (counts[unit], int(UNIT_COST[unit])))
        for unit in order:
            if spendable >= UNIT_COST[unit]:
                return Action(ActionType.PRODUCE, unit_type=unit)
        return None

    def _produce_if_affordable(
        self,
        resources: float,
        unit_type: UnitType,
        completed: set[str],
    ) -> Optional[Action]:
        if any(required.value not in completed for required in unit_tech_requirements(unit_type)):
            return None
        if resources >= UNIT_COST[unit_type]:
            return Action(ActionType.PRODUCE, unit_type=unit_type)
        return None

    def _next_build_reserve(self, obs: dict) -> float:
        nodes = obs.get("nodes", {}) or {}
        core_node = obs.get("core_node")
        core = nodes.get(core_node, {}) or {}
        completed = self._completed_buildings(obs)
        existing = {
            building.get("building_type")
            for building in core.get("buildings", []) or []
            if building.get("owner") == self.player_id
        }
        for building_type in self.config.build_plan:
            if building_type.value in existing:
                continue
            if building_type == BuildingType.TECH_2 and BuildingType.TECH_1.value not in completed:
                continue
            return float(BUILDING_COST[building_type])
        return 0.0

    def _completed_buildings(self, obs: dict) -> set[str]:
        return {
            building.get("building_type")
            for node in (obs.get("nodes", {}) or {}).values()
            for building in node.get("buildings", []) or []
            if (
                building.get("owner") == self.player_id
                and building.get("complete")
                and float(building.get("hp", 0.0) or 0.0) > 0
            )
        }

    def _move_action(self, obs: dict) -> Optional[Action]:
        nodes = obs.get("nodes", {}) or {}
        baron = obs.get("baron", {}) or {}
        candidates = []
        for node_id, node in nodes.items():
            if node.get("owner") != self.player_id:
                continue
            count = unit_total(node.get("units", {}) or {})
            if count < self.config.send_threshold:
                continue
            targets = self._accessible_targets(node_id, nodes, baron)
            if targets:
                candidates.append((count, node_id, targets))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        _, source, targets = candidates[0]
        target = self._choose_target(targets, nodes, baron)
        if target is None:
            return None
        return Action(ActionType.MOVE, source=source, destination=target)

    def _choose_target(self, targets: list[str], nodes: dict, baron: dict) -> Optional[str]:
        if self.config.target_baron and "Baron" in targets and self._baron_is_worth_targeting(nodes, baron):
            return "Baron"
        if self.config.enemy_priority:
            core_targets = [
                node_id for node_id in targets
                if NODE_TYPES.get(node_id) == NodeType.CORE
                and nodes[node_id]["owner"] not in {"NEUTRAL"}
                and not self._is_friendly(nodes[node_id]["owner"])
            ]
            if core_targets:
                return sorted(core_targets)[0]
            enemy_targets = [
                node_id for node_id in targets
                if node_id != "Baron"
                and nodes[node_id]["owner"] not in {"NEUTRAL"}
                and not self._is_friendly(nodes[node_id]["owner"])
            ]
            if enemy_targets:
                return sorted(enemy_targets)[0]
        for node_id in targets:
            if node_id == "Baron":
                continue
            owner = nodes[node_id]["owner"]
            if owner == "NEUTRAL":
                return node_id
        for node_id in targets:
            if node_id == "Baron":
                continue
            owner = nodes[node_id]["owner"]
            if owner not in {"NEUTRAL"} and not self._is_friendly(owner):
                return node_id
        return targets[0] if targets else None

    def _baron_is_worth_targeting(self, nodes: dict, baron: dict) -> bool:
        if nodes.get("Baron", {}).get("owner") != "NEUTRAL":
            return True
        return int(baron.get("hp", 0) or 0) > 0

    def _accessible_targets(self, src: str, nodes: dict, baron: dict) -> list[str]:
        targets: list[str] = []
        seen_targets = set()
        visited = {src}
        queue = [src]
        while queue:
            node_id = queue.pop(0)
            for neighbor in ADJACENCY.get(node_id, []):
                if neighbor != src and self._is_destination(neighbor, node_id, nodes, baron):
                    if neighbor not in seen_targets:
                        seen_targets.add(neighbor)
                        targets.append(neighbor)
                if neighbor not in visited and self._is_transit(neighbor, nodes, baron):
                    visited.add(neighbor)
                    queue.append(neighbor)
        return targets

    def _is_destination(self, node_id: str, via_node: str, nodes: dict, baron: dict) -> bool:
        if node_id == "Baron":
            return True
        owner = nodes[node_id]["owner"]
        if owner == "NEUTRAL" or self._is_friendly(owner):
            return True
        return self._is_transit(via_node, nodes, baron)

    def _is_transit(self, node_id: str, nodes: dict, baron: dict) -> bool:
        if node_id == "Baron" and nodes[node_id]["owner"] == "NEUTRAL" and int(baron.get("hp", 0) or 0) > 0:
            return False
        owner = nodes[node_id]["owner"]
        return owner == "NEUTRAL" or self._is_friendly(owner)

    def _is_friendly(self, owner: str) -> bool:
        if owner not in PLAYER_IDS:
            return False
        return get_team(owner) == self.team


def make_strategy_bot(model_id: str, player_id: str, team: str, seed: int = 0):
    if model_id == "random":
        return RandomBot(player_id, team, seed=seed)
    if model_id == "scripted":
        return ScriptedBot(player_id, team)
    if model_id in STRATEGIES:
        return StrategyBot(player_id, team, model_id)
    raise ValueError(f"Unknown audit strategy: {model_id}")


def make_team_bots(team_a: str, team_b: str, seed_offset: int) -> dict:
    bots = {}
    for index, player_id in enumerate(PLAYER_IDS):
        team = get_team(player_id)
        model_id = team_a if team == "TEAM_A" else team_b
        bots[player_id] = make_strategy_bot(model_id, player_id, team, seed=seed_offset + index)
    return bots


def run_match_with_metrics(request: dict[str, Any]) -> dict[str, Any]:
    seed = int(request["seed"])
    team_a = str(request["team_a"])
    team_b = str(request["team_b"])
    scenario = str(request.get("scenario", ""))
    state = make_initial_state(seed=seed)

    production_counts: Counter[str] = Counter()
    first_production_tick: dict[str, int] = {}
    build_counts: Counter[str] = Counter()

    original_apply_production = engine_module.apply_production
    original_apply_build = engine_module.apply_build

    def counted_apply_production(state_arg, player_id, unit_type):
        applied = original_apply_production(state_arg, player_id, unit_type)
        if applied:
            unit_value = unit_type.value
            team = get_team(player_id)
            for key in (f"{player_id}:{unit_value}", f"{team}:{unit_value}", f"ALL:{unit_value}"):
                production_counts[key] += 1
                first_production_tick.setdefault(key, int(state_arg.tick))
        return applied

    def counted_apply_build(state_arg, player_id, node_id, building_type):
        applied = original_apply_build(state_arg, player_id, node_id, building_type)
        if applied:
            building_value = building_type.value if isinstance(building_type, BuildingType) else str(building_type)
            team = get_team(player_id)
            for key in (f"{player_id}:{building_value}", f"{team}:{building_value}", f"ALL:{building_value}"):
                build_counts[key] += 1
        return applied

    engine_module.apply_production = counted_apply_production
    engine_module.apply_build = counted_apply_build
    try:
        engine = GameEngine(
            state,
            make_team_bots(team_a, team_b, seed_offset=seed * 10),
            record_trace=False,
            record_replay=True,
        )
        winner = engine.run()
    finally:
        engine_module.apply_production = original_apply_production
        engine_module.apply_build = original_apply_build

    snapshots = engine.replay_ticks
    tick_269 = snapshot_at(snapshots, DEATH_TIME_START_TICK - 1)
    first_complete = first_completed_building_ticks(snapshots)
    incomes = income_breakdown_from_snapshots(snapshots, state)
    final_buildings = final_building_counter(state)
    final_units = final_unit_counter(state)
    final_cp_income = final_cp_income_by_owner(state)
    core_hp_269 = core_hp_from_snapshot(tick_269) if tick_269 else {}
    destroyed_cores_269 = {
        team: sum(1 for player_id in players if core_hp_269.get(player_id, CORE_HP) <= 0)
        for team, players in (("TEAM_A", TEAM_A_PLAYERS), ("TEAM_B", TEAM_B_PLAYERS))
    }

    return {
        "scenario": scenario,
        "seed": seed,
        "team_a": team_a,
        "team_b": team_b,
        "winner": winner,
        "winner_type": state.winner_type or "unknown",
        "ticks": int(state.tick),
        "core_combat_damage": round(float(state.core_combat_damage), 3),
        "core_damage_by_team": {team: round(float(value), 3) for team, value in state.core_damage_by_team.items()},
        "final_core_hp": {pid: round(float(player.core_hp), 3) for pid, player in state.players.items()},
        "core_hp_269": core_hp_269,
        "destroyed_cores_269": destroyed_cores_269,
        "production_counts": counter_dict(production_counts),
        "first_production_tick": dict(sorted(first_production_tick.items())),
        "build_counts": counter_dict(build_counts),
        "first_completed_building_tick": dict(sorted(first_complete.items())),
        "income_breakdown": incomes,
        "final_buildings": counter_dict(final_buildings),
        "final_units": counter_dict(final_units),
        "final_cp_income_by_owner": counter_dict(final_cp_income),
        "final_resources": {pid: round(float(player.resources), 3) for pid, player in state.players.items()},
        "allied_lord": {
            "owner": state.baron.allied_lord_owner,
            "target": state.baron.allied_lord_target,
            "hp": round(float(state.baron.allied_lord_hp), 3),
            "alive": bool(state.baron.allied_lord_alive),
            "expended": bool(state.baron.allied_lord_expended),
        },
    }


def snapshot_at(snapshots: list[dict], tick: int) -> Optional[dict]:
    for snapshot in snapshots:
        if int(snapshot.get("tick", -1)) == int(tick):
            return snapshot
    return None


def core_hp_from_snapshot(snapshot: dict) -> dict[str, float]:
    return {
        player_id: round(float(snapshot["players"][player_id]["core_hp"]), 3)
        for player_id in PLAYER_IDS
    }


def first_completed_building_ticks(snapshots: list[dict]) -> dict[str, int]:
    first: dict[str, int] = {}
    for snapshot in snapshots:
        tick = int(snapshot.get("tick", 0))
        for node in (snapshot.get("nodes") or {}).values():
            for building in node.get("buildings", []) or []:
                if not building.get("complete"):
                    continue
                owner = building.get("owner")
                building_type = building.get("building_type")
                if owner and building_type:
                    first.setdefault(f"{owner}:{building_type}", tick)
                    first.setdefault(f"{get_team(owner)}:{building_type}", tick)
                    first.setdefault(f"ALL:{building_type}", tick)
    return first


def income_breakdown_from_snapshots(snapshots: list[dict], state) -> dict[str, dict[str, float]]:
    totals = {
        "base": defaultdict(float),
        "cp": defaultdict(float),
        "collector": defaultdict(float),
    }
    neutral_income = {node_id: int(node.neutral_income) for node_id, node in state.nodes.items()}
    for snapshot in snapshots:
        for player_id, pdata in (snapshot.get("players") or {}).items():
            if pdata.get("alive"):
                totals["base"][get_team(player_id)] += float(RESOURCE_PER_TICK)
        for node_id, node in (snapshot.get("nodes") or {}).items():
            owner = node.get("owner")
            if owner in PLAYER_IDS and int(node.get("neutral_hp", 0) or 0) <= 0:
                income = neutral_income.get(node_id, 0)
                if income > 0:
                    totals["cp"][get_team(owner)] += float(income)
            for building in node.get("buildings", []) or []:
                if not building.get("complete"):
                    continue
                if building.get("building_type") != BuildingType.ECONOMY_COLLECTOR.value:
                    continue
                owner = building.get("owner")
                if owner in PLAYER_IDS:
                    totals["collector"][get_team(owner)] += float(BUILDING_STATS[BuildingType.ECONOMY_COLLECTOR]["income"])
    return {
        bucket: {team: round(value, 3) for team, value in sorted(values.items())}
        for bucket, values in totals.items()
    }


def final_building_counter(state) -> Counter:
    counter: Counter[str] = Counter()
    for node_id, node in state.nodes.items():
        for building in node.buildings:
            status = "complete" if building.is_complete(state.tick) else "building"
            for key in (
                f"{building.owner}:{building.building_type}:{status}",
                f"{get_team(building.owner)}:{building.building_type}:{status}",
                f"ALL:{building.building_type}:{status}",
                f"{node_id}:{building.building_type}:{status}",
            ):
                counter[key] += 1
    return counter


def final_unit_counter(state) -> Counter:
    counter: Counter[str] = Counter()
    for node in state.nodes.values():
        if node.owner in PLAYER_IDS:
            for unit_type, count in (node.units or {}).items():
                counter[f"{node.owner}:{unit_type}"] += int(count or 0)
                counter[f"{get_team(node.owner)}:{unit_type}"] += int(count or 0)
                counter[f"ALL:{unit_type}"] += int(count or 0)
        for owner, units in (node.visiting_units or {}).items():
            for unit_type, count in (units or {}).items():
                counter[f"{owner}:{unit_type}"] += int(count or 0)
                counter[f"{get_team(owner)}:{unit_type}"] += int(count or 0)
                counter[f"ALL:{unit_type}"] += int(count or 0)
    for group in state.moving_groups:
        for unit_type, count in (group.units or {}).items():
            counter[f"{group.player_id}:{unit_type}"] += int(count or 0)
            counter[f"{get_team(group.player_id)}:{unit_type}"] += int(count or 0)
            counter[f"ALL:{unit_type}"] += int(count or 0)
    return counter


def final_cp_income_by_owner(state) -> Counter:
    counter: Counter[str] = Counter()
    for node in state.nodes.values():
        if node.owner in PLAYER_IDS and node.neutral_income > 0 and node.neutral_hp <= 0:
            counter[node.owner] += int(node.neutral_income)
            counter[get_team(node.owner)] += int(node.neutral_income)
    return counter


def chunked(items: list[Any], workers: int) -> list[list[Any]]:
    workers = max(1, min(workers, len(items)))
    return [items[index::workers] for index in range(workers)]


def run_chunk(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run_match_with_metrics(request) for request in requests]


def run_scenario(name: str, team_a: str, team_b: str, seed: int, matches: int, workers: int) -> dict[str, Any]:
    requests = [
        {"seed": seed + index, "team_a": team_a, "team_b": team_b}
        for index in range(matches)
    ]
    if workers <= 1 or matches <= 1:
        rows = run_chunk(requests)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(workers, len(requests))) as pool:
            nested = pool.map(run_chunk, chunked(requests, workers))
        rows = [row for chunk in nested for row in chunk]
        rows.sort(key=lambda row: row["seed"])

    return summarize_scenario(name, team_a, team_b, rows)


def run_scenarios(
    scenarios: list[tuple[str, str, str]],
    seed: int,
    matches: int,
    workers: int,
) -> list[dict[str, Any]]:
    requests = []
    scenario_order = {}
    for index, (name, team_a, team_b) in enumerate(scenarios):
        scenario_order[name] = (team_a, team_b)
        scenario_seed = seed + index * 10_000
        for match_index in range(matches):
            requests.append({
                "scenario": name,
                "seed": scenario_seed + match_index,
                "team_a": team_a,
                "team_b": team_b,
            })

    if workers <= 1 or len(requests) <= 1:
        rows = run_chunk(requests)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(workers, len(requests))) as pool:
            nested = pool.map(run_chunk, chunked(requests, workers))
        rows = [row for chunk in nested for row in chunk]
        rows.sort(key=lambda row: (row["scenario"], row["seed"]))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["scenario"]].append(row)

    return [
        summarize_scenario(name, scenario_order[name][0], scenario_order[name][1], grouped.get(name, []))
        for name, _, _ in scenarios
    ]


def summarize_scenario(name: str, team_a: str, team_b: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    winners = Counter(row["winner"] for row in rows)
    winner_types = Counter(row["winner_type"] for row in rows)
    production = Counter()
    builds = Counter()
    final_buildings = Counter()
    final_units = Counter()
    final_cp_income = Counter()
    first_ticks: dict[str, list[int]] = defaultdict(list)
    income_totals: dict[str, Counter] = {
        "base": Counter(),
        "cp": Counter(),
        "collector": Counter(),
    }
    for row in rows:
        production.update(row["production_counts"])
        builds.update(row["build_counts"])
        final_buildings.update(row["final_buildings"])
        final_units.update(row["final_units"])
        final_cp_income.update(row["final_cp_income_by_owner"])
        for key, tick in row["first_production_tick"].items():
            first_ticks[f"first_prod:{key}"].append(int(tick))
        for key, tick in row["first_completed_building_tick"].items():
            first_ticks[f"first_complete:{key}"].append(int(tick))
        for bucket, team_values in row["income_breakdown"].items():
            for team, value in team_values.items():
                income_totals[bucket][team] += float(value)

    matches = max(1, len(rows))
    return {
        "name": name,
        "team_a": team_a,
        "team_b": team_b,
        "matches": len(rows),
        "winners": counter_dict(winners),
        "winner_types": counter_dict(winner_types),
        "ticks": numeric_stats([row["ticks"] for row in rows]),
        "core_combat_damage": numeric_stats([row["core_combat_damage"] for row in rows]),
        "destroyed_cores_269": {
            team: numeric_stats([row["destroyed_cores_269"].get(team, 0) for row in rows])
            for team in ("TEAM_A", "TEAM_B")
        },
        "core_hp_269_team_sum": {
            team: numeric_stats([
                sum(row["core_hp_269"].get(pid, CORE_HP) for pid in (TEAM_A_PLAYERS if team == "TEAM_A" else TEAM_B_PLAYERS))
                for row in rows
                if row["core_hp_269"]
            ])
            for team in ("TEAM_A", "TEAM_B")
        },
        "production_total": counter_dict(production),
        "production_per_match": per_match(production, matches),
        "build_total": counter_dict(builds),
        "build_per_match": per_match(builds, matches),
        "final_buildings_total": counter_dict(final_buildings),
        "final_buildings_per_match": per_match(final_buildings, matches),
        "final_units_per_match": per_match(final_units, matches),
        "final_cp_income_per_match": per_match(final_cp_income, matches),
        "income_per_match": {
            bucket: {team: round(value / matches, 3) for team, value in sorted(counter.items())}
            for bucket, counter in income_totals.items()
        },
        "first_tick_stats": {
            key: numeric_stats(values)
            for key, values in sorted(first_ticks.items())
        },
        "samples": rows[:3],
    }


def per_match(counter: Counter, matches: int) -> dict[str, float]:
    return {
        key: round(value / max(1, matches), 3)
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def building_objective_tests() -> list[dict[str, Any]]:
    rows = []
    for budget in (20, 30, 40, 50, 60, 80):
        for attacker in ALL_UNITS:
            rows.append(run_building_objective(attacker, budget, [BuildingType.TURRET], label="turret"))
            rows.append(run_building_objective(attacker, budget, [BuildingType.TURRET], include_core=True, label="turret_core"))
    return rows


def run_building_objective(
    attacker_type: UnitType,
    budget: int,
    buildings: list[BuildingType],
    *,
    include_core: bool = False,
    label: str,
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
    assign_formation(attackers, base_x=170.0, base_y=205.0, forward_x=1.0)
    assign_formation(defenders, base_x=440.0, base_y=205.0, forward_x=-1.0)
    result = simulate_units(units, neutral_aggro_teams=set(), max_ticks=100)
    return {
        "kind": "building_objective",
        "budget": budget,
        "attacker": army_label(attacker_type, attacker_count),
        "objective": label,
        **result,
    }


def tech_cost_summary() -> dict[str, Any]:
    collector = BuildingType.ECONOMY_COLLECTOR
    tech1 = BuildingType.TECH_1
    tech2 = BuildingType.TECH_2
    income = float(RESOURCE_PER_TICK)
    tech1_save_ticks = int((BUILDING_COST[tech1] + income - 1) // income)
    tech1_complete = tech1_save_ticks + BUILDING_BUILD_TICKS[tech1]
    resources_after_tech1 = tech1_save_ticks * income - BUILDING_COST[tech1] + BUILDING_BUILD_TICKS[tech1] * income
    tech2_wait_ticks = int((max(0.0, BUILDING_COST[tech2] - resources_after_tech1) + income - 1) // income)
    tech2_complete = tech1_complete + tech2_wait_ticks + BUILDING_BUILD_TICKS[tech2]
    return {
        "base_income_per_player": RESOURCE_PER_TICK,
        "collector": {
            "cost": BUILDING_COST[collector],
            "build_ticks": BUILDING_BUILD_TICKS[collector],
            "income": BUILDING_STATS[collector]["income"],
            "payback_after_completion": round(BUILDING_COST[collector] / BUILDING_STATS[collector]["income"], 3),
            "payback_from_start": round(BUILDING_BUILD_TICKS[collector] + BUILDING_COST[collector] / BUILDING_STATS[collector]["income"], 3),
        },
        "tech1_raider_unlock": {
            "tech_cost": BUILDING_COST[tech1],
            "build_ticks": BUILDING_BUILD_TICKS[tech1],
            "raider_cost": UNIT_COST[UnitType.RAIDER],
            "earliest_completion_from_0_resources": tech1_complete,
        },
        "tech2_bmp_unlock_without_collector": {
            "total_build_cost": BUILDING_COST[tech1] + BUILDING_COST[tech2],
            "total_build_ticks": BUILDING_BUILD_TICKS[tech1] + BUILDING_BUILD_TICKS[tech2],
            "bmp_cost": UNIT_COST[UnitType.BMP],
            "earliest_tech2_complete_from_0_resources": tech2_complete,
            "note": "Assumes base income only and spends as soon as each tech is affordable.",
        },
        "core_slot_conflict": {
            "core_slots": 2,
            "collector_plus_tech1_blocks_tech2": True,
            "tech1_plus_tech2_blocks_collector": True,
            "collector_plus_enhanced_turret_blocks_tech": True,
        },
    }


def build_report(seed: int, matches: int, workers: int, scenario_filter: str = "") -> dict[str, Any]:
    symmetric = [
        ("random_mirror", "random", "random"),
        ("scripted_mirror", "scripted", "scripted"),
        ("basic_mix_mirror", "basic_mix", "basic_mix"),
        ("ranger_spam_mirror", "ranger_spam", "ranger_spam"),
        ("skirm_flood_mirror", "skirm_flood", "skirm_flood"),
        ("raider_eco_mirror", "raider_eco", "raider_eco"),
        ("raider_fast_mirror", "raider_fast", "raider_fast"),
        ("bmp_rush_mirror", "bmp_rush", "bmp_rush"),
        ("eco_turtle_mirror", "eco_turtle", "eco_turtle"),
    ]
    asymmetric = [
        ("raider_fast_vs_basic", "raider_fast", "basic_mix"),
        ("raider_eco_vs_basic", "raider_eco", "basic_mix"),
        ("bmp_rush_vs_basic", "bmp_rush", "basic_mix"),
        ("eco_turtle_vs_basic", "eco_turtle", "basic_mix"),
        ("ranger_spam_vs_basic", "ranger_spam", "basic_mix"),
        ("basic_vs_scripted", "basic_mix", "scripted"),
        ("bmp_rush_vs_scripted", "bmp_rush", "scripted"),
        ("raider_fast_vs_scripted", "raider_fast", "scripted"),
        ("core_basic_vs_scripted", "core_basic", "scripted"),
        ("core_raider_fast_vs_scripted", "core_raider_fast", "scripted"),
        ("core_basic_vs_basic", "core_basic", "basic_mix"),
        ("core_raider_fast_vs_basic", "core_raider_fast", "basic_mix"),
        ("core_bmp_rush_vs_basic", "core_bmp_rush", "basic_mix"),
    ]
    scenarios = symmetric + asymmetric
    if scenario_filter:
        scenarios = [
            scenario for scenario in scenarios
            if scenario_filter in scenario[0]
            or scenario_filter in scenario[1]
            or scenario_filter in scenario[2]
        ]
    scenario_reports = run_scenarios(scenarios, seed, matches, workers)

    return {
        "kind": "construction_balance_deep_dive",
        "config": {
            "seed": seed,
            "matches_per_scenario": matches,
            "workers": workers,
            "commit": current_code_commit(),
            "rules": rules_meta(),
        },
        "constants": {
            "units": {unit.value: {**dict(UNIT_STATS[unit]), "cost": UNIT_COST[unit]} for unit in ALL_UNITS},
            "buildings": {
                building.value: {
                    **dict(BUILDING_STATS[building]),
                    "cost": BUILDING_COST[building],
                    "build_ticks": BUILDING_BUILD_TICKS[building],
                }
                for building in BuildingType
            },
            "tech_costs": tech_cost_summary(),
        },
        "micro": {
            "key_duels": key_duels(),
            "building_objectives": building_objective_tests(),
        },
        "scenarios": scenario_reports,
    }


def key_duels() -> list[dict[str, Any]]:
    rows = []
    for budget in (20, 30, 40, 50, 60, 80):
        for left, right in (
            (UnitType.VANGUARD, UnitType.RANGER),
            (UnitType.RANGER, UnitType.SKIRMISHER),
            (UnitType.SKIRMISHER, UnitType.VANGUARD),
            (UnitType.RAIDER, UnitType.VANGUARD),
            (UnitType.RAIDER, UnitType.RANGER),
            (UnitType.RAIDER, UnitType.SKIRMISHER),
            (UnitType.BMP, UnitType.RAIDER),
            (UnitType.BMP, UnitType.RANGER),
            (UnitType.BMP, UnitType.SKIRMISHER),
            (UnitType.BMP, UnitType.VANGUARD),
        ):
            rows.append(run_unit_duel(left, right, budget))
    return rows


def print_compact(report: dict[str, Any], output: Path, elapsed_sec: float) -> None:
    scenario_summary = {}
    for scenario in report["scenarios"]:
        scenario_summary[scenario["name"]] = {
            "teams": f"{scenario['team_a']} vs {scenario['team_b']}",
            "winners": scenario["winners"],
            "winner_types": scenario["winner_types"],
            "ticks": scenario["ticks"],
            "core_combat_damage": scenario["core_combat_damage"],
            "destroyed_cores_269": scenario["destroyed_cores_269"],
            "income_per_match": scenario["income_per_match"],
            "build_per_match_excerpt": {
                key: value
                for key, value in scenario["build_per_match"].items()
                if key.startswith("ALL:")
            },
            "production_per_match_excerpt": {
                key: value
                for key, value in scenario["production_per_match"].items()
                if key.startswith("ALL:")
            },
        }
    compact = {
        "out": str(output),
        "elapsed_sec": round(elapsed_sec, 3),
        "tech_costs": report["constants"]["tech_costs"],
        "scenario_summary": scenario_summary,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construction/economy/tech balance deep-dive.")
    parser.add_argument("--matches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", default="")
    parser.add_argument("--scenario-filter", default="", help="Substring filter for scenario/strategy names.")
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(int(args.seed), int(args.matches), int(args.workers), str(args.scenario_filter or ""))
    elapsed = time.perf_counter() - started

    output = Path(args.output or f"analysis/construction_balance_deep_dive_{args.seed}_{args.matches}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_compact(report, output, elapsed)


if __name__ == "__main__":
    main()
