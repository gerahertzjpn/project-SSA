from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.constants import (
    CORE_BUILDINGS,
    CONTROL_POINT_BUILDINGS,
    MAX_CONTROL_POINT_BUILDINGS,
    MAX_CORE_BUILDINGS,
    PLAYER_IDS,
    BuildingType,
    UnitType,
)
from src.core.engine import GameEngine
from src.core.map import ADJACENCY, NODE_TYPES, NodeType
from src.core.map2d import MOVEMENT_MAP_UNITS_PER_TICK
from src.core.runner import make_bot
from src.core.state import make_initial_state
from src.core.units import UNIT_STATS


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


def run_crash_test(
    matches: int,
    seed_base: int,
    bot_types: list[str],
    visual_audit: bool,
    visual_ticks: int,
    visual_jump: float,
    visual_speed_ratio: float,
) -> int:
    winners = Counter()
    winner_types = Counter()
    ticks = []
    issues = []

    for bot_type in bot_types:
        for offset in range(matches):
            seed = seed_base + offset
            state = make_initial_state(seed=seed)
            engine = GameEngine(state, _make_bots(bot_type, seed), record_replay=True, record_trace=True)
            winner = engine.run()
            winners[(bot_type, winner)] += 1
            winner_types[(bot_type, engine.state.winner_type)] += 1
            ticks.append(engine.state.tick)
            replay_issues = _inspect_replay(seed, engine)
            if visual_audit:
                replay_issues.extend(_inspect_visual_continuity(
                    seed,
                    engine.replay_ticks,
                    max_ticks=visual_ticks,
                    jump_threshold=visual_jump,
                    speed_ratio_threshold=visual_speed_ratio,
                ))
            issues.extend((issue[0], bot_type, *issue[1:]) for issue in replay_issues)

    issue_summary = Counter(kind for kind, *_ in issues)
    print("CRASH_TEST_SIMULATION")
    print("seed_base", seed_base, "matches", matches)
    print("bot_types", bot_types)
    print("winners", dict(winners))
    print("winner_types", dict(winner_types))
    print("avg_ticks", round(sum(ticks) / len(ticks), 2), "min_ticks", min(ticks), "max_ticks", max(ticks))
    print("visual_audit", visual_audit, "visual_ticks", visual_ticks, "visual_jump", visual_jump)
    print("seeds_with_issues", len({seed for _, _, seed, *_ in issues}))
    print("issue_summary", dict(issue_summary))
    if issues:
        print("samples")
        for issue in issues[:20]:
            print(issue)
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
    parser.add_argument("--visual-audit", action="store_true")
    parser.add_argument("--visual-ticks", type=int, default=180, help="0 means inspect all ticks.")
    parser.add_argument("--visual-jump", type=float, default=120.0)
    parser.add_argument("--visual-speed-ratio", type=float, default=5.0)
    args = parser.parse_args()
    bot_types = [item.strip() for item in args.bot_types.split(",") if item.strip()]
    return run_crash_test(
        args.matches,
        args.seed_base,
        bot_types,
        args.visual_audit,
        args.visual_ticks,
        args.visual_jump,
        args.visual_speed_ratio,
    )


if __name__ == "__main__":
    raise SystemExit(main())
