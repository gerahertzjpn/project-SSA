from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.constants import (
    BUILDING_COST,
    CORE_BUILDINGS,
    CONTROL_POINT_BUILDINGS,
    FACTION_UNIQUE_BUILDINGS,
    MAX_CONTROL_POINT_BUILDINGS,
    MAX_CORE_BUILDINGS,
    PLAYER_IDS,
    SPECIAL_LORD_DESTINATIONS,
    TEAM_A_PLAYERS,
    TEAM_B_PLAYERS,
    UNIT_COST,
    UNIT_FACTION,
    BuildingType,
    UnitType,
    unit_tech_requirements,
)
from src.core.engine import GameEngine
from src.core.map import ADJACENCY, NODE_TYPES, NodeType
from src.core.map2d import MOVEMENT_MAP_UNITS_PER_TICK
from src.core.runner import make_bot
from src.core.state import make_initial_state
from src.core.units import UNIT_STATS
from src.training.spaces import (
    ACTION_DIM,
    ACTION_HEAD_SLICES,
    BUILD_ACTION_SPECS,
    BUILD_ACTION_START,
    MOVE_ACTION_SPECS,
    MOVE_ACTION_START,
    UNIT_ORDER,
    action_is_legal,
    decode_action,
    find_route_obs,
    legal_action_mask,
    movable_sources,
    sample_legal_action,
)


PLAYER_SET = set(PLAYER_IDS)
VALID_UNIT_TYPES = {unit.value for unit in UnitType} | {building.value for building in BuildingType} | {"NEUTRAL", "BARON", "CORE"}
VALID_NODE_IDS = set(NODE_TYPES)


def _team_of(player_id: str) -> str:
    if player_id.startswith("A"):
        return "TEAM_A"
    if player_id.startswith("B"):
        return "TEAM_B"
    return player_id


def _make_bots(bot_type: str, seed: int) -> dict:
    return {
        player_id: make_bot(bot_type, player_id, _team_of(player_id), seed=seed * 10 + index)
        for index, player_id in enumerate(PLAYER_IDS)
    }


def _make_matchup_bots(team_a_bot: str, team_b_bot: str, seed: int) -> dict:
    bots = {}
    for index, player_id in enumerate(PLAYER_IDS):
        team = _team_of(player_id)
        model_id = team_a_bot if team == "TEAM_A" else team_b_bot
        bots[player_id] = make_bot(model_id, player_id, team, seed=seed * 10 + index)
    return bots


def _factions_for_setup(setup: str) -> dict[str, str]:
    setup = str(setup or "all_humans").strip()
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
    raise ValueError(f"Unknown faction setup: {setup}")


def _side_unit_total(side: dict[str, Any]) -> int:
    return sum(int(value) for value in (side.get("units") or {}).values())


def _battle_has_hostiles(battle: dict[str, Any]) -> bool:
    sides = battle.get("sides") or {}
    live_players = [
        player_id
        for player_id, data in sides.items()
        if player_id in PLAYER_SET and _side_unit_total(data) > 0
    ]
    live_teams = {_team_of(player_id) for player_id in live_players}
    if len(live_teams) >= 2:
        return True

    neutral_live = (
        _side_unit_total(sides.get("NEUTRAL", {})) > 0
        or _side_unit_total(sides.get("BARON", {})) > 0
    )
    aggro_teams = set(battle.get("neutral_aggro_teams") or [])
    if neutral_live and live_teams and (aggro_teams & live_teams or battle.get("location_id") == "Baron"):
        return True

    return False


def _physical_lord_count(tick: dict[str, Any]) -> tuple[int, list[tuple]]:
    count = 0
    locations = []
    for group in tick.get("moving_groups") or []:
        lord_count = int((group.get("units") or {}).get(UnitType.LORD.value, 0))
        if lord_count:
            count += lord_count
            locations.append((
                "moving",
                group.get("player"),
                group.get("src"),
                group.get("dst"),
                group.get("intent"),
            ))

    for battle in tick.get("active_battles") or []:
        for row in battle.get("formation") or []:
            if len(row) >= 4 and row[2] == "L" and float(row[3]) > 0:
                count += 1
                locations.append((
                    "battle",
                    row[1],
                    battle.get("battle_id"),
                    battle.get("location_id"),
                    None,
                ))

    for visual in tick.get("visual_units") or []:
        if (
            visual.get("unit_type") == UnitType.LORD.value
            and visual.get("layer") in {"stationed", "visiting"}
            and float(visual.get("hp", 0)) > 0
        ):
            count += 1
            locations.append((
                "stationed_visual",
                visual.get("owner"),
                visual.get("node_id"),
                visual.get("layer"),
                None,
            ))

    return count, locations


def _inspect_replay(seed: int, engine: GameEngine) -> list[tuple]:
    issues = []
    for tick in engine.replay_ticks:
        tick_index = tick.get("tick")
        players = tick.get("players") or {}
        dead_players = {
            player_id
            for player_id, data in players.items()
            if not data.get("alive", True) or float(data.get("core_hp", 0)) <= 0
        }
        nodes = tick.get("nodes") or {}
        active_node_battles = {
            battle.get("location_id")
            for battle in tick.get("active_battles") or []
            if battle.get("location_kind") in {"node", "approach"}
        }

        for battle in tick.get("active_battles") or []:
            sides = battle.get("sides") or {}
            if not _battle_has_hostiles(battle):
                issues.append((
                    "battle_without_hostiles",
                    seed,
                    tick_index,
                    battle.get("battle_id"),
                    {
                        "location": battle.get("location_id"),
                        "participants": battle.get("participants"),
                        "sides": sorted(sides.keys()),
                    },
                ))
            for player_id in dead_players:
                if _side_unit_total(sides.get(player_id, {})) > 0:
                    issues.append(("dead_player_battle_side", seed, tick_index, battle.get("battle_id"), player_id))

        for node_id, node in nodes.items():
            owner = node.get("owner")
            if owner in dead_players:
                issues.append(("dead_player_owns_node", seed, tick_index, node_id, owner))
            if node_id not in active_node_battles:
                for visitor_id, units in (node.get("visiting_units") or {}).items():
                    if visitor_id in dead_players and sum(units.values()) > 0:
                        issues.append(("dead_player_visiting_node", seed, tick_index, node_id, visitor_id))
            issues.extend(_inspect_node_buildings(seed, tick_index, node_id, node, players, dead_players))

        for group in tick.get("moving_groups") or []:
            issues.extend(_inspect_moving_group(seed, tick_index, group, players, nodes, dead_players))
            player_id = group.get("player")
            if player_id in dead_players and sum((group.get("units") or {}).values()) > 0:
                issues.append((
                    "dead_player_moving_group",
                    seed,
                    tick_index,
                    player_id,
                    {"src": group.get("src"), "dst": group.get("dst"), "units": group.get("units")},
                ))
            if int((group.get("units") or {}).get(UnitType.LORD.value, 0)) > 0:
                baron = tick.get("baron") or {}
                if group.get("intent") != "LORD_PUSH":
                    issues.append(("lord_group_bad_intent", seed, tick_index, player_id, group.get("intent")))
                if baron.get("allied_lord_owner") and player_id != baron.get("allied_lord_owner"):
                    issues.append((
                        "lord_group_global_mismatch",
                        seed,
                        tick_index,
                        player_id,
                        baron.get("allied_lord_owner"),
                    ))
                target = baron.get("allied_lord_target")
                if target in players and (
                    not players[target].get("alive", True) or float(players[target].get("core_hp", 0)) <= 0
                ):
                    issues.append(("allied_lord_targets_dead_core", seed, tick_index, target, players[target]))

        for visual in tick.get("visual_units") or []:
            owner = visual.get("owner")
            issues.extend(_inspect_visual_unit(seed, tick_index, visual))
            if owner in dead_players and float(visual.get("hp", 0)) > 0 and visual.get("state") != "dead":
                issues.append((
                    "dead_player_live_visual",
                    seed,
                    tick_index,
                    owner,
                    {
                        "unit_type": visual.get("unit_type"),
                        "layer": visual.get("layer"),
                        "state": visual.get("state"),
                    },
                ))

        baron = tick.get("baron") or {}
        if baron.get("allied_lord_alive") and not baron.get("allied_lord_expended"):
            ready_tick = int(baron.get("allied_lord_ready_tick") or 0)
            if ready_tick <= 0 or int(tick_index) >= ready_tick:
                count, locations = _physical_lord_count(tick)
                if count == 0:
                    issues.append((
                        "allied_lord_visual_missing",
                        seed,
                        tick_index,
                        baron.get("allied_lord_owner"),
                        dict(baron),
                    ))
                elif count != 1:
                    issues.append(("allied_lord_physical_count_bad", seed, tick_index, count, locations))

    return issues


def _inspect_moving_group(
    seed: int,
    tick_index: int,
    group: dict[str, Any],
    players: dict[str, Any],
    nodes: dict[str, Any],
    dead_players: set[str],
) -> list[tuple]:
    issues: list[tuple] = []
    player_id = group.get("player")
    if player_id not in PLAYER_SET:
        issues.append(("moving_group_bad_player", seed, tick_index, player_id, group))
    elif player_id in dead_players and sum((group.get("units") or {}).values()) > 0:
        issues.append(("moving_group_dead_player_units", seed, tick_index, player_id, group))
    elif player_id in players and not players[player_id].get("alive", True):
        issues.append(("moving_group_inactive_player", seed, tick_index, player_id, group))

    for field in ("src", "dst", "final_dst"):
        node_id = group.get(field)
        if node_id is not None and node_id not in VALID_NODE_IDS:
            issues.append(("moving_group_bad_node", seed, tick_index, field, node_id, group))

    path = group.get("path") or []
    if path and path[0] not in VALID_NODE_IDS:
        issues.append(("moving_group_bad_path_start", seed, tick_index, path, group))
    if path and path[-1] != group.get("final_dst"):
        issues.append(("moving_group_final_dst_mismatch", seed, tick_index, path, group.get("final_dst")))
    for left, right in zip(path, path[1:]):
        if right not in ADJACENCY.get(left, []):
            issues.append(("moving_group_non_adjacent_path", seed, tick_index, left, right, path))

    progress = float(group.get("progress", 0.0) or 0.0)
    if progress < -1e-6 or progress > 1.0 + 1e-6:
        issues.append(("moving_group_progress_invalid", seed, tick_index, progress, group))

    speed = float(group.get("speed", 0.0) or 0.0)
    if speed <= 0 and sum((group.get("units") or {}).values()) > 0:
        issues.append(("moving_group_speed_invalid", seed, tick_index, speed, group))

    for unit_type, count in (group.get("units") or {}).items():
        if unit_type not in VALID_UNIT_TYPES:
            issues.append(("moving_group_unknown_unit", seed, tick_index, unit_type, group))
        if int(count or 0) < 0:
            issues.append(("moving_group_negative_units", seed, tick_index, unit_type, count, group))

    for visual_ids in (group.get("visual_units") or {}).values():
        if any(not str(visual_id) for visual_id in visual_ids):
            issues.append(("moving_group_empty_visual_id", seed, tick_index, group))
    return issues


def _inspect_visual_unit(seed: int, tick_index: int, visual: dict[str, Any]) -> list[tuple]:
    issues: list[tuple] = []
    unit_type = visual.get("unit_type")
    if unit_type not in VALID_UNIT_TYPES:
        issues.append(("visual_unknown_unit_type", seed, tick_index, unit_type, visual))
    owner = visual.get("owner")
    if owner not in PLAYER_SET and owner not in {"NEUTRAL", "BARON", None}:
        issues.append(("visual_bad_owner", seed, tick_index, owner, visual))
    for axis in ("x", "y"):
        value = visual.get(axis)
        if value is None or not math.isfinite(float(value)):
            issues.append(("visual_coordinate_invalid", seed, tick_index, axis, value, visual))
    hp = float(visual.get("hp", 0.0) or 0.0)
    max_hp = float(visual.get("max_hp", 0.0) or 0.0)
    if max_hp < 0 or hp < -1e-6 or (max_hp > 0 and hp > max_hp + 1e-6):
        issues.append(("visual_hp_invalid", seed, tick_index, hp, max_hp, visual))
    return issues


def _inspect_visual_continuity(
    seed: int,
    ticks: list[dict[str, Any]],
    *,
    max_ticks: int,
    jump_threshold: float,
    speed_ratio_threshold: float,
) -> list[tuple]:
    issues: list[tuple] = []
    previous: dict[str, dict[str, Any]] = {}
    limit = max_ticks if max_ticks > 0 else len(ticks)
    for tick in ticks[:limit]:
        tick_index = int(tick.get("tick", 0) or 0)
        seen_ids: set[str] = set()
        current: dict[str, dict[str, Any]] = {}
        for visual in tick.get("visual_units") or []:
            visual_id = str(visual.get("visual_id") or "")
            if not visual_id:
                continue
            if visual_id in seen_ids:
                issues.append(("visual_duplicate_id", seed, tick_index, visual_id))
            seen_ids.add(visual_id)
            current[visual_id] = visual
            if visual_id not in previous:
                continue

            prev = previous[visual_id]
            dx = float(visual.get("x", 0.0) or 0.0) - float(prev.get("x", 0.0) or 0.0)
            dy = float(visual.get("y", 0.0) or 0.0) - float(prev.get("y", 0.0) or 0.0)
            distance = math.hypot(dx, dy)
            if distance >= jump_threshold:
                issues.append((
                    "visual_large_jump",
                    seed,
                    tick_index,
                    visual_id,
                    round(distance, 3),
                    {
                        "unit_type": visual.get("unit_type"),
                        "owner": visual.get("owner"),
                        "prev": {
                            "tick": tick_index - 1,
                            "x": round(float(prev.get("x", 0.0) or 0.0), 3),
                            "y": round(float(prev.get("y", 0.0) or 0.0), 3),
                            "state": prev.get("state"),
                            "layer": prev.get("layer"),
                        },
                        "now": {
                            "x": round(float(visual.get("x", 0.0) or 0.0), 3),
                            "y": round(float(visual.get("y", 0.0) or 0.0), 3),
                            "state": visual.get("state"),
                            "layer": visual.get("layer"),
                        },
                    },
                ))
            unit_type = visual.get("unit_type")
            nominal_speed = _visual_nominal_speed(unit_type)
            if nominal_speed > 0 and distance > max(jump_threshold * 0.5, nominal_speed * speed_ratio_threshold):
                issues.append((
                    "visual_speed_spike",
                    seed,
                    tick_index,
                    visual_id,
                    round(distance, 3),
                    round(nominal_speed, 3),
                    visual.get("unit_type"),
                ))
        previous = current
    return issues


def _visual_nominal_speed(unit_type: str | None) -> float:
    try:
        stats = UNIT_STATS.get(UnitType(unit_type)) if unit_type else None
    except ValueError:
        stats = None
    if not stats:
        return 0.0
    return MOVEMENT_MAP_UNITS_PER_TICK * float(stats.get("speed", 0.0) or 0.0)


def _inspect_node_buildings(
    seed: int,
    tick_index: int,
    node_id: str,
    node: dict[str, Any],
    players: dict[str, Any],
    dead_players: set[str],
) -> list[tuple]:
    issues: list[tuple] = []
    buildings = list(node.get("buildings") or [])
    if not buildings:
        return issues

    owner = node.get("owner")
    node_type = NODE_TYPES[node_id]
    allowed = {
        building.value for building in (
            CONTROL_POINT_BUILDINGS if node_type == NodeType.CONTROL_POINT else CORE_BUILDINGS
        )
    } if node_type in {NodeType.CONTROL_POINT, NodeType.CORE} else set()
    max_slots = MAX_CONTROL_POINT_BUILDINGS if node_type == NodeType.CONTROL_POINT else MAX_CORE_BUILDINGS

    if node_type == NodeType.BARON:
        issues.append(("building_on_baron_node", seed, tick_index, node_id, buildings))
        return issues

    if node_type == NodeType.CORE and (node_id in dead_players or not players.get(node_id, {}).get("alive", True)):
        issues.append(("building_on_dead_core", seed, tick_index, node_id, buildings))

    if len(buildings) > max_slots:
        issues.append(("building_slot_overflow", seed, tick_index, node_id, len(buildings)))

    seen_types = set()
    for building in buildings:
        building_type = building.get("building_type")
        building_owner = building.get("owner")
        if building_owner in dead_players:
            issues.append(("dead_player_building", seed, tick_index, node_id, building_owner, building))
        if owner != building_owner:
            issues.append(("building_owner_mismatch", seed, tick_index, node_id, owner, building))
        if building_type not in {item.value for item in BuildingType}:
            issues.append(("unknown_building_type", seed, tick_index, node_id, building))
        if building_type not in allowed:
            issues.append(("building_type_not_allowed_on_node", seed, tick_index, node_id, building_type))
        if (building_owner, building_type) in seen_types:
            issues.append(("duplicate_building_type", seed, tick_index, node_id, building_owner, building_type))
        seen_types.add((building_owner, building_type))
        hp = float(building.get("hp", 0.0) or 0.0)
        max_hp = float(building.get("max_hp", 0.0) or 0.0)
        progress = float(building.get("progress", 0.0) or 0.0)
        if hp < -1e-6 or max_hp <= 0 or hp > max_hp + 1e-6:
            issues.append(("building_hp_invalid", seed, tick_index, node_id, building))
        if progress < -1e-6 or progress > 1.0 + 1e-6:
            issues.append(("building_progress_invalid", seed, tick_index, node_id, building))
        if bool(building.get("complete")) != (int(tick_index) >= int(building.get("complete_tick", 10**9) or 10**9)):
            issues.append(("building_complete_flag_invalid", seed, tick_index, node_id, building))

    return issues


def _completed_building_types(obs: dict, player_id: str) -> set[str]:
    return {
        str(building.get("building_type"))
        for node in (obs.get("nodes", {}) or {}).values()
        for building in node.get("buildings", []) or []
        if (
            building.get("owner") == player_id
            and building.get("complete")
            and float(building.get("hp", 0.0) or 0.0) > 0
        )
    }


def _friendly_owner(owner: str | None, player_id: str) -> bool:
    return owner in PLAYER_SET and _team_of(str(owner)) == _team_of(player_id)


def _enemy_owner(owner: str | None, player_id: str) -> bool:
    return owner in PLAYER_SET and _team_of(str(owner)) != _team_of(player_id)


def _inspect_legal_produce_action(
    seed: int,
    tick_index: int,
    player_id: str,
    obs: dict,
    local_index: int,
) -> list[tuple]:
    issues: list[tuple] = []
    unit_index = int(local_index) - 1
    if unit_index < 0 or unit_index >= len(UNIT_ORDER):
        issues.append(("legal_produce_bad_local_index", seed, tick_index, player_id, local_index))
        return issues
    unit_type = UNIT_ORDER[unit_index]
    resources = float(obs.get("resources", 0.0) or 0.0)
    faction = obs.get("faction", "humans")
    if UNIT_FACTION.get(unit_type) != faction:
        issues.append(("legal_produce_wrong_faction", seed, tick_index, player_id, unit_type.value, faction))
    if resources + 1e-6 < float(UNIT_COST[unit_type]):
        issues.append(("legal_produce_insufficient_resources", seed, tick_index, player_id, unit_type.value, resources))
    completed = _completed_building_types(obs, player_id)
    missing = [required.value for required in unit_tech_requirements(unit_type) if required.value not in completed]
    if missing:
        issues.append(("legal_produce_missing_tech", seed, tick_index, player_id, unit_type.value, missing))
    return issues


def _inspect_legal_build_action(
    seed: int,
    tick_index: int,
    player_id: str,
    obs: dict,
    local_index: int,
) -> list[tuple]:
    issues: list[tuple] = []
    spec_index = int(local_index) - 1
    if spec_index < 0 or spec_index >= len(BUILD_ACTION_SPECS):
        issues.append(("legal_build_bad_local_index", seed, tick_index, player_id, local_index))
        return issues

    action = decode_action([0, 0, local_index], obs, player_id, assume_legal=True)
    building_type = action.building_type
    node_id = action.destination
    if building_type is None or node_id is None:
        issues.append(("legal_build_decodes_badly", seed, tick_index, player_id, local_index))
        return issues

    resources = float(obs.get("resources", 0.0) or 0.0)
    if resources + 1e-6 < float(BUILDING_COST[building_type]):
        issues.append(("legal_build_insufficient_resources", seed, tick_index, player_id, building_type.value, resources))
    unique_owner_faction = {
        building.value: faction
        for faction, building in FACTION_UNIQUE_BUILDINGS.items()
    }.get(building_type.value)
    if unique_owner_faction is not None and obs.get("faction", "humans") != unique_owner_faction:
        issues.append(("legal_build_wrong_faction_unique", seed, tick_index, player_id, building_type.value, obs.get("faction")))

    nodes = obs.get("nodes", {}) or {}
    node = nodes.get(node_id)
    if not node:
        issues.append(("legal_build_missing_node", seed, tick_index, player_id, building_type.value, node_id))
        return issues
    if node.get("owner") != player_id:
        issues.append(("legal_build_wrong_owner", seed, tick_index, player_id, building_type.value, node_id, node.get("owner")))
    if int(node.get("neutral_hp", 0) or 0) > 0:
        issues.append(("legal_build_uncleared_neutral", seed, tick_index, player_id, building_type.value, node_id))
    if any(
        battle.get("location_kind") == "node" and battle.get("location_id") == node_id
        for battle in obs.get("active_battles", []) or []
    ):
        issues.append(("legal_build_during_node_battle", seed, tick_index, player_id, building_type.value, node_id))

    node_type = NODE_TYPES.get(node_id)
    existing = [building for building in node.get("buildings", []) or [] if building.get("owner") == player_id]
    existing_types = {str(building.get("building_type")) for building in existing}
    if node_type == NodeType.CONTROL_POINT:
        if building_type not in CONTROL_POINT_BUILDINGS:
            issues.append(("legal_build_bad_cp_type", seed, tick_index, player_id, building_type.value, node_id))
        if len(existing) >= MAX_CONTROL_POINT_BUILDINGS:
            issues.append(("legal_build_cp_slot_full", seed, tick_index, player_id, node_id))
    elif node_type == NodeType.CORE:
        if node_id != obs.get("core_node"):
            issues.append(("legal_build_on_non_own_core", seed, tick_index, player_id, building_type.value, node_id))
        if building_type not in CORE_BUILDINGS:
            issues.append(("legal_build_bad_core_type", seed, tick_index, player_id, building_type.value, node_id))
        if len(existing) >= MAX_CORE_BUILDINGS:
            issues.append(("legal_build_core_slot_full", seed, tick_index, player_id, node_id))
        if building_type == BuildingType.TECH_2 and BuildingType.TECH_1.value not in _completed_building_types(obs, player_id):
            issues.append(("legal_build_tech2_without_tech1", seed, tick_index, player_id, node_id))
    else:
        issues.append(("legal_build_on_invalid_node_type", seed, tick_index, player_id, building_type.value, node_id, str(node_type)))
    if building_type.value in existing_types:
        issues.append(("legal_build_duplicate_type", seed, tick_index, player_id, building_type.value, node_id))
    return issues


def _inspect_legal_move_action(
    seed: int,
    tick_index: int,
    player_id: str,
    obs: dict,
    local_index: int,
) -> list[tuple]:
    issues: list[tuple] = []
    spec_index = int(local_index) - 1
    if spec_index < 0 or spec_index >= len(MOVE_ACTION_SPECS):
        issues.append(("legal_move_bad_local_index", seed, tick_index, player_id, local_index))
        return issues
    action = decode_action([0, local_index, 0], obs, player_id, assume_legal=True)
    if action.source is None or action.destination is None:
        issues.append(("legal_move_decodes_badly", seed, tick_index, player_id, local_index))
        return issues
    sources = set(movable_sources(obs, player_id))
    if action.source not in sources:
        issues.append(("legal_move_source_not_movable", seed, tick_index, player_id, action.source, action.destination, sorted(sources)))
    nodes = obs.get("nodes", {}) or {}
    source_node = nodes.get(action.source)
    if source_node and sum(int(value or 0) for value in (source_node.get("units") or {}).values()) <= 0:
        visiting = (source_node.get("visiting_units") or {}).get(player_id, {}) or {}
        if sum(int(value or 0) for value in visiting.values()) <= 0:
            issues.append(("legal_move_empty_source", seed, tick_index, player_id, action.source, action.destination))
    if action.destination not in SPECIAL_LORD_DESTINATIONS:
        if action.destination not in nodes:
            issues.append(("legal_move_missing_destination", seed, tick_index, player_id, action.source, action.destination))
            return issues
        if NODE_TYPES.get(action.destination) == NodeType.CORE:
            target_player = obs.get("players", {}).get(action.destination)
            if target_player and (not target_player.get("alive", True) or float(target_player.get("core_hp", 0.0) or 0.0) <= 0):
                issues.append(("legal_move_to_dead_core", seed, tick_index, player_id, action.source, action.destination))
        route = find_route_obs(player_id, action.source, action.destination, obs)
        if not route:
            issues.append(("legal_move_no_route", seed, tick_index, player_id, action.source, action.destination))
        else:
            for transit_node_id in route[1:-1]:
                transit = nodes.get(transit_node_id, {}) or {}
                owner = transit.get("owner")
                if _enemy_owner(owner, player_id) and int(transit.get("neutral_hp", 0) or 0) <= 0:
                    issues.append((
                        "legal_move_route_through_enemy",
                        seed,
                        tick_index,
                        player_id,
                        action.source,
                        action.destination,
                        route,
                    ))
                    break
    else:
        baron = obs.get("baron", {}) or {}
        if action.destination == "ALLY_LORD" and not (
            baron.get("allied_lord_alive")
            and _friendly_owner(baron.get("allied_lord_owner"), player_id)
        ):
            issues.append(("legal_join_lord_without_ally_lord", seed, tick_index, player_id, action.source, dict(baron)))
        if action.destination == "ENEMY_LORD" and not (
            baron.get("allied_lord_alive")
            and _enemy_owner(baron.get("allied_lord_owner"), player_id)
        ):
            issues.append(("legal_hunt_lord_without_enemy_lord", seed, tick_index, player_id, action.source, dict(baron)))
    return issues


def _audit_action_masks(
    seed: int,
    tick_index: int,
    engine: GameEngine,
    rng: np.random.Generator,
    samples_per_player: int,
) -> tuple[list[tuple], list[tuple]]:
    issues: list[tuple] = []
    suspicious: list[tuple] = []
    for player_id in PLAYER_IDS:
        player = engine.state.players[player_id]
        obs = engine.observe(player_id)
        mask = legal_action_mask(obs, player_id)
        if mask.shape != (ACTION_DIM,):
            issues.append(("action_mask_bad_shape", seed, tick_index, player_id, tuple(mask.shape)))
            continue
        if not all(bool(mask[action_slice.start]) for action_slice in ACTION_HEAD_SLICES):
            issues.append(("action_mask_missing_head_none", seed, tick_index, player_id, [int(mask[s.start]) for s in ACTION_HEAD_SLICES]))
        if not player.alive:
            if int(mask.sum()) != len(ACTION_HEAD_SLICES):
                issues.append(("dead_player_has_non_none_actions", seed, tick_index, player_id, int(mask.sum())))
            continue

        produce_locals = [int(index) for index in np.flatnonzero(mask[ACTION_HEAD_SLICES[0]]) if int(index) > 0]
        move_locals = [int(index) for index in np.flatnonzero(mask[ACTION_HEAD_SLICES[1]]) if int(index) > 0]
        build_locals = [int(index) for index in np.flatnonzero(mask[ACTION_HEAD_SLICES[2]]) if int(index) > 0]

        for local in produce_locals:
            issues.extend(_inspect_legal_produce_action(seed, tick_index, player_id, obs, local))
        for local in move_locals:
            issues.extend(_inspect_legal_move_action(seed, tick_index, player_id, obs, local))
        for local in build_locals:
            issues.extend(_inspect_legal_build_action(seed, tick_index, player_id, obs, local))

        resources = float(obs.get("resources", 0.0) or 0.0)
        produce_costs = [
            int(UNIT_COST[UNIT_ORDER[local - 1]])
            for local in produce_locals
            if 0 < local <= len(UNIT_ORDER)
        ]
        build_costs = [
            int(BUILDING_COST[BUILD_ACTION_SPECS[local - 1].building_type])
            for local in build_locals
            if 0 < local <= len(BUILD_ACTION_SPECS)
        ]
        if produce_costs and build_costs and min(produce_costs) + min(build_costs) > resources + 1e-6:
            suspicious.append((
                "legal_combo_min_build_plus_produce_overspends",
                seed,
                tick_index,
                player_id,
                round(resources, 3),
                min(produce_costs),
                min(build_costs),
            ))

        for _ in range(max(0, int(samples_per_player))):
            action = sample_legal_action(mask, rng)
            if not action_is_legal(action, mask):
                issues.append(("sampled_action_illegal", seed, tick_index, player_id, action.tolist()))
                continue
            decoded = decode_action(action, obs, player_id)
            if decoded.action_type.value == "HOLD" and any(int(value) > 0 for value in action):
                issues.append(("sampled_non_none_decodes_hold", seed, tick_index, player_id, action.tolist()))
    return issues, suspicious


def _run_engine_with_optional_action_audit(
    seed: int,
    engine: GameEngine,
    *,
    action_audit: bool,
    action_audit_ticks: int,
    action_audit_samples: int,
) -> tuple[str, list[tuple], list[tuple]]:
    if not action_audit:
        return engine.run(), [], []

    issues: list[tuple] = []
    suspicious: list[tuple] = []
    rng = np.random.default_rng(seed + 93_711)
    while engine.state.winner is None and engine.state.tick < 300:
        if action_audit_ticks <= 0 or engine.state.tick < action_audit_ticks:
            tick_issues, tick_suspicious = _audit_action_masks(
                seed,
                int(engine.state.tick),
                engine,
                rng,
                samples_per_player=action_audit_samples,
            )
            issues.extend(tick_issues)
            suspicious.extend(tick_suspicious)
        engine.tick()

    if engine.state.winner is None:
        engine._finalize_death_time()
    if engine.state.winner is None:
        engine.state.winner = "DRAW"
        engine.state.winner_type = "draw_no_core_damage"
    return engine.state.winner, issues, suspicious


def run_crash_test(
    matches: int,
    seed_base: int,
    bot_types: list[str],
    matchups: list[tuple[str, str]],
    faction_setups: list[str],
    visual_audit: bool,
    visual_ticks: int,
    visual_jump: float,
    visual_speed_ratio: float,
    action_audit: bool,
    action_audit_ticks: int,
    action_audit_samples: int,
    strict_suspicious: bool,
) -> int:
    winners = Counter()
    winner_types = Counter()
    ticks = []
    issues = []
    suspicious = []

    matchup_list = list(matchups)
    if not matchup_list:
        matchup_list = [(bot_type, bot_type) for bot_type in bot_types]
    faction_setups = faction_setups or ["all_humans"]

    for matchup_index, (team_a_bot, team_b_bot) in enumerate(matchup_list):
        matchup_label = f"{team_a_bot}_vs_{team_b_bot}"
        for faction_index, faction_setup in enumerate(faction_setups):
            label = f"{matchup_label}__{faction_setup}"
            for offset in range(matches):
                seed = seed_base + matchup_index * 100_000 + faction_index * 10_000 + offset
                state = make_initial_state(seed=seed, player_factions=_factions_for_setup(faction_setup))
                engine = GameEngine(
                    state,
                    _make_matchup_bots(team_a_bot, team_b_bot, seed),
                    record_replay=True,
                    record_trace=True,
                )
                winner, action_issues, action_suspicious = _run_engine_with_optional_action_audit(
                    seed,
                    engine,
                    action_audit=action_audit,
                    action_audit_ticks=action_audit_ticks,
                    action_audit_samples=action_audit_samples,
                )
                winners[(label, winner)] += 1
                winner_types[(label, engine.state.winner_type)] += 1
                ticks.append(engine.state.tick)
                replay_issues = _inspect_replay(seed, engine)
                replay_issues.extend(action_issues)
                suspicious.extend((item[0], label, *item[1:]) for item in action_suspicious)
                if visual_audit:
                    replay_issues.extend(_inspect_visual_continuity(
                        seed,
                        engine.replay_ticks,
                        max_ticks=visual_ticks,
                        jump_threshold=visual_jump,
                        speed_ratio_threshold=visual_speed_ratio,
                    ))
                issues.extend((issue[0], label, *issue[1:]) for issue in replay_issues)

    issue_summary = Counter(kind for kind, *_ in issues)
    suspicious_summary = Counter(kind for kind, *_ in suspicious)
    print("CRASH_TEST_SIMULATION")
    print("seed_base", seed_base, "matches", matches)
    print("bot_types", bot_types)
    print("matchups", matchup_list)
    print("faction_setups", faction_setups)
    print("winners", dict(winners))
    print("winner_types", dict(winner_types))
    print("avg_ticks", round(sum(ticks) / len(ticks), 2), "min_ticks", min(ticks), "max_ticks", max(ticks))
    print("visual_audit", visual_audit, "visual_ticks", visual_ticks, "visual_jump", visual_jump)
    print("action_audit", action_audit, "action_audit_ticks", action_audit_ticks, "action_audit_samples", action_audit_samples)
    print("seeds_with_issues", len({seed for _, _, seed, *_ in issues}))
    print("issue_summary", dict(issue_summary))
    print("suspicious_summary", dict(suspicious_summary))
    if suspicious:
        print("suspicious_samples")
        for item in suspicious[:20]:
            print(item)
    if issues:
        print("samples")
        for issue in issues[:20]:
            print(issue)
        return 1
    if strict_suspicious and suspicious:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Crash-test simulation replay invariants.")
    parser.add_argument("--matches", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=2026060200)
    parser.add_argument(
        "--bot-types",
        default="random",
        help="Comma-separated bot model ids, e.g. random,scripted_core,scripted_core_raider.",
    )
    parser.add_argument(
        "--matchups",
        default="",
        help="Comma-separated TEAM_A:TEAM_B bot pairs. Overrides mirror --bot-types if set.",
    )
    parser.add_argument(
        "--faction-setups",
        default="all_humans",
        help=(
            "Comma-separated setups: all_humans, all_infected, all_robots, "
            "humans_vs_infected, humans_vs_robots, infected_vs_robots, mixed_slots."
        ),
    )
    parser.add_argument("--visual-audit", action="store_true")
    parser.add_argument("--visual-ticks", type=int, default=180, help="0 means inspect all ticks.")
    parser.add_argument("--visual-jump", type=float, default=120.0)
    parser.add_argument("--visual-speed-ratio", type=float, default=5.0)
    parser.add_argument("--action-audit", action="store_true")
    parser.add_argument("--action-audit-ticks", type=int, default=120, help="0 means inspect all pre-tick masks.")
    parser.add_argument("--action-audit-samples", type=int, default=3)
    parser.add_argument("--strict-suspicious", action="store_true")
    args = parser.parse_args()
    bot_types = [item.strip() for item in args.bot_types.split(",") if item.strip()]
    matchups = []
    for item in str(args.matchups or "").split(","):
        item = item.strip()
        if not item:
            continue
        left, sep, right = item.partition(":")
        if not sep:
            raise ValueError(f"Bad matchup {item!r}; expected TEAM_A_BOT:TEAM_B_BOT")
        matchups.append((left.strip(), right.strip()))
    faction_setups = [item.strip() for item in str(args.faction_setups or "").split(",") if item.strip()]
    return run_crash_test(
        args.matches,
        args.seed_base,
        bot_types,
        matchups,
        faction_setups,
        args.visual_audit,
        args.visual_ticks,
        args.visual_jump,
        args.visual_speed_ratio,
        args.action_audit,
        args.action_audit_ticks,
        args.action_audit_samples,
        args.strict_suspicious,
    )


if __name__ == "__main__":
    raise SystemExit(main())
