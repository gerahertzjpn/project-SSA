"""Fast deterministic 2D combat helpers for the simple map engine.

The strategic engine still starts/stops aggregate battles, but active combat no
longer resolves on a synthetic 1D line. Units keep real map coordinates, choose
targets by Euclidean range, move on the map when no attack is available, and
emit compact replay frames from the same state the engine uses.
"""

from __future__ import annotations

from math import hypot
from typing import Optional

from .constants import TEAM_A_PLAYERS, TEAM_B_PLAYERS, BuildingType, UnitType
from .map2d import (
    MOVEMENT_MAP_UNITS_PER_TICK,
    NODE_LAYOUT,
    Point2D,
    WORLD_SIZE,
    get_approach,
    point_on_approach_edge,
)
from .units import UnitInstance


COMBAT_RANGE_TO_MAP_UNITS = 24.0
LORD_CONTACT_RANGE_MAP = 12.0
FORMATION_SPACING_MAP = 7.0
FORMATION_DEPTH_MAP = 10.0
EDGE_FRONTLINE_GAP_MAP = 4.0
RANGE_EPSILON_MAP = 1e-6
FOCUS_ASSIST_RANGE_MAP = 18.0

_SLOT_ORDER = {
    UnitType.VANGUARD.value: 0,
    UnitType.LORD.value: 0,
    UnitType.BRUTE.value: 0,
    UnitType.BEHEMOTH.value: 0,
    UnitType.WALKER.value: 0,
    UnitType.SKIRMISHER.value: 1,
    UnitType.RAIDER.value: 1,
    UnitType.SWARMLING.value: 1,
    UnitType.RAVAGER.value: 1,
    UnitType.DRONE.value: 1,
    UnitType.GUARDIAN_MECH.value: 1,
    UnitType.RANGER.value: 2,
    UnitType.BMP.value: 2,
    UnitType.SPITTER.value: 2,
    UnitType.LASER.value: 2,
    UnitType.SIEGE_PLATFORM.value: 2,
    "ARCHER": 2,
}
_BUILDING_UNIT_TYPES = {building_type.value for building_type in BuildingType}


def place_player_units_2d(
    battle,
    units: list[UnitInstance],
    player_id: str,
    src: str,
    dst: str,
    entry_side: str,
    progress: Optional[float] = None,
) -> None:
    """Place newly joined player units at their real map entry point."""

    if not units:
        return

    anchor, forward = _player_anchor_and_forward(
        battle,
        player_id=player_id,
        src=src,
        dst=dst,
        entry_side=entry_side,
        progress=progress,
    )
    _spread_units(units, anchor, forward)


def place_static_units_2d(
    units: list[UnitInstance],
    node_id: str,
    role: str = "node",
    owner: Optional[str] = None,
) -> None:
    """Place objective-owned units at the objective they belong to."""

    if not units:
        return

    node = NODE_LAYOUT[node_id]
    if role in {"core", "baron", "neutral"}:
        anchor = node.center
        forward = _default_forward(owner or units[0].team)
    else:
        anchor = node.center
        forward = _default_forward(owner or units[0].team)
    _spread_units(units, anchor, forward, compact=True)


def tick_battle_units_2d(
    units: list[UnitInstance],
    record_trace: bool = True,
    neutral_aggro_teams: Optional[set[str]] = None,
    battle=None,
) -> list:
    """Advance an active 2D battle by one world/combat tick."""

    alive = [unit for unit in units if unit.is_alive()]
    if not alive:
        return []

    start_positions = {unit.uid: (unit.x, unit.y) for unit in alive}
    attack_ready = {
        unit.uid
        for unit in alive
        if not _is_allied_lord(unit)
        and _find_target(
            unit,
            alive,
            neutral_aggro_teams=neutral_aggro_teams,
            in_range_only=True,
            positions=start_positions,
            battle=battle,
        )
        is not None
    }

    planned_positions: dict[int, Point2D] = {}
    for unit in alive:
        if unit.speed <= 0:
            continue
        if _is_allied_lord(unit):
            planned_positions[unit.uid] = _next_lord_position(
                unit,
                alive,
                neutral_aggro_teams=neutral_aggro_teams,
                positions=start_positions,
                battle=battle,
            )
            continue
        if unit.uid in attack_ready:
            continue
        target = _find_target(
            unit,
            alive,
            neutral_aggro_teams=neutral_aggro_teams,
            in_range_only=False,
            positions=start_positions,
            battle=battle,
        )
        if target is None:
            continue
        planned_positions[unit.uid] = _next_position_toward_target(
            unit,
            target,
            positions=start_positions,
        )

    raw_positions = dict(planned_positions)
    for unit in alive:
        if unit.uid not in planned_positions:
            continue
        planned_positions[unit.uid] = _clamp_position_to_first_hostile_contact(
            unit,
            alive,
            neutral_aggro_teams,
            start_positions,
            raw_positions,
        )

    for unit in alive:
        if unit.uid in planned_positions:
            unit.x, unit.y = planned_positions[unit.uid]

    pending: dict[int, float] = {}
    targeting_pending: dict[str, dict[int, float]] = {}
    damage_log: dict[int, float] = {}
    target_log: dict[int, int] = {}
    attack_log: dict[int, list[tuple[int, float]]] = {}
    alive = [unit for unit in units if unit.is_alive()]
    current_positions = {unit.uid: (unit.x, unit.y) for unit in alive}

    for unit in alive:
        side = _combat_side(unit.team)
        unit_targeting_pending = targeting_pending.setdefault(side, {})
        for damage in _attack_damage_sequence(unit):
            target = _find_attack_target(
                unit,
                alive,
                neutral_aggro_teams=neutral_aggro_teams,
                pending=unit_targeting_pending,
                positions=current_positions,
                battle=battle,
            )
            if target is None:
                break
            _queue_attack(unit, target, pending, damage_log, target_log, attack_log, damage=damage)
            unit_targeting_pending[target.uid] = (
                unit_targeting_pending.get(target.uid, 0.0) + damage
            )

    _apply_core_defense_overflow(units, pending, battle)

    for unit in units:
        if unit.uid in pending:
            unit.hp -= pending[unit.uid]

    return _frame_snapshot(units, damage_log, target_log, attack_log) if record_trace else []


def _apply_core_defense_overflow(
    units: list[UnitInstance],
    pending: dict[int, float],
    battle,
) -> None:
    """Spill excess damage dealt to a besieged core's stack down its layers.

    Without this, damage committed to the defending side is capped at whatever
    single layer the attackers could reach, so one cheap unit produced each tick
    can soak an entire enemy army indefinitely. Here the total damage aimed at
    the core owner's units this tick is re-poured by priority — troops first,
    then the buildings, then the core:

      * Troops overflow freely: surplus beyond a defender's HP spills to the next
        troop and then onto the building layer, so a token defender can no longer
        hold off a superior army.
      * Buildings stay "plates": only the current outermost building absorbs in a
        tick (up to its HP); any surplus past it is dropped rather than spilling
        into the next building or the core. A core fronted by N buildings still
        survives at least N ticks of assault.
      * The core is only reached once no buildings remain.
    """

    core_owner = _battle_core_owner(battle)
    if not core_owner:
        return

    defenders = [unit for unit in units if unit.team == core_owner and unit.is_alive()]
    if not defenders:
        return

    total = 0.0
    for unit in defenders:
        committed = pending.pop(unit.uid, 0.0)
        if committed > 0:
            total += committed
    if total <= RANGE_EPSILON_MAP:
        return

    troops = [
        unit for unit in defenders
        if not _is_building_unit(unit) and unit.unit_type != "CORE"
    ]
    buildings = sorted(
        (unit for unit in defenders if _is_building_unit(unit)),
        key=_building_slot_index,
        reverse=True,
    )

    def _absorb(unit, amount: float) -> float:
        absorbed = min(float(unit.hp), amount)
        if absorbed <= 0:
            return 0.0
        pending[unit.uid] = pending.get(unit.uid, 0.0) + absorbed
        return absorbed

    remaining = total
    # 1) Troops soak first, overflowing into one another.
    for unit in troops:
        if remaining <= RANGE_EPSILON_MAP:
            return
        remaining -= _absorb(unit, remaining)

    # 2) One building plate per tick: only the outermost building takes damage,
    #    surplus past it is dropped (keeps the core from falling in one tick).
    if buildings:
        _absorb(buildings[0], remaining)
        return

    # 3) No buildings left — the surplus finally reaches the core.
    cores = [unit for unit in defenders if unit.unit_type == "CORE"]
    if cores:
        _absorb(cores[0], remaining)


def has_hostile_units_2d(
    units: list[UnitInstance],
    neutral_aggro_teams: Optional[set[str]] = None,
) -> bool:
    alive = [unit for unit in units if unit.is_alive()]
    for index, unit in enumerate(alive):
        for other in alive[index + 1:]:
            if _is_hostile(unit, other, neutral_aggro_teams):
                return True
    return False


def unit_range_map(unit: UnitInstance) -> float:
    return max(0.0, float(unit.range) * COMBAT_RANGE_TO_MAP_UNITS)


def _player_anchor_and_forward(
    battle,
    player_id: str,
    src: str,
    dst: str,
    entry_side: str,
    progress: Optional[float],
) -> tuple[Point2D, Point2D]:
    if battle.location_kind == "edge":
        if _is_adjacent(src, dst):
            battle_progress = _edge_progress_for_direction(battle, src, dst)
            forward = _direction_between_nodes(src, dst)
            collision = point_on_approach_edge(src, dst, battle_progress)
            anchor = (
                collision[0] - forward[0] * (FORMATION_DEPTH_MAP + EDGE_FRONTLINE_GAP_MAP),
                collision[1] - forward[1] * (FORMATION_DEPTH_MAP + EDGE_FRONTLINE_GAP_MAP),
            )
            return anchor, forward
        return _battle_node_center(battle), _default_forward(player_id)

    if battle.location_kind == "approach" and _is_adjacent(src, dst):
        group_progress = battle.edge_position if progress is None else progress
        return point_on_approach_edge(src, dst, group_progress), _direction_between_nodes(src, dst)

    if _is_adjacent(src, dst) and dst == battle.location_id:
        approach = get_approach(src, dst)
        forward = _direction_between_nodes(src, dst)
        return (
            approach.entry_point[0] - forward[0] * FORMATION_DEPTH_MAP,
            approach.entry_point[1] - forward[1] * FORMATION_DEPTH_MAP,
        ), forward

    if _is_adjacent(src, dst) and src == battle.location_id:
        approach = get_approach(src, dst)
        forward = _direction_between_nodes(src, dst)
        return (
            approach.exit_point[0] - forward[0] * FORMATION_DEPTH_MAP,
            approach.exit_point[1] - forward[1] * FORMATION_DEPTH_MAP,
        ), forward

    node = NODE_LAYOUT[battle.location_id]
    side_forward = _default_forward(player_id)
    offset_sign = -1.0 if entry_side == "attacker" else 1.0
    anchor = (
        node.center[0] + side_forward[0] * offset_sign * node.radius * 0.45,
        node.center[1] + side_forward[1] * offset_sign * node.radius * 0.45,
    )
    return anchor, side_forward


def _spread_units(
    units: list[UnitInstance],
    anchor: Point2D,
    forward: Point2D,
    compact: bool = False,
) -> None:
    fx, fy = _normalize(forward)
    px, py = -fy, fx
    ordered = sorted(
        enumerate(units),
        key=lambda item: (_SLOT_ORDER.get(item[1].unit_type, 1), item[1].unit_type, item[0]),
    )
    count = len(ordered)
    spacing = FORMATION_SPACING_MAP * (0.72 if compact else 1.0)
    depth = FORMATION_DEPTH_MAP * (0.45 if compact else 1.0)

    for order_index, (_, unit) in enumerate(ordered):
        centered = order_index - (count - 1) / 2.0
        row = _SLOT_ORDER.get(unit.unit_type, 1)
        lateral = centered * spacing
        front_offset = (1 - row) * depth
        if unit.unit_type == UnitType.LORD.value and count > 1:
            front_offset += depth
        unit.x = anchor[0] + px * lateral + fx * front_offset
        unit.y = anchor[1] + py * lateral + fy * front_offset
        unit.forward_x = fx
        unit.forward_y = fy


def _next_position_toward_target(
    unit: UnitInstance,
    target: UnitInstance,
    positions: Optional[dict[int, Point2D]] = None,
) -> Point2D:
    ux, uy = _unit_point(unit, positions)
    tx, ty = _unit_point(target, positions)
    dx = tx - ux
    dy = ty - uy
    distance = hypot(dx, dy)
    desired_range = unit_range_map(unit)
    if distance <= desired_range + RANGE_EPSILON_MAP:
        return ux, uy
    step = min(_unit_step(unit), max(0.0, distance - desired_range))
    nx, ny = _normalize((dx, dy))
    return ux + nx * step, uy + ny * step


def _clamp_position_to_first_hostile_contact(
    unit: UnitInstance,
    units: list[UnitInstance],
    neutral_aggro_teams: Optional[set[str]],
    start_positions: dict[int, Point2D],
    raw_positions: dict[int, Point2D],
) -> Point2D:
    raw_unit = raw_positions.get(unit.uid, _unit_point(unit, start_positions))
    start_unit = _unit_point(unit, start_positions)
    desired_range = unit_range_map(unit)
    earliest_contact = 1.0

    for other in units:
        if not _is_hostile(unit, other, neutral_aggro_teams):
            continue
        start_other = _unit_point(other, start_positions)
        raw_other = raw_positions.get(other.uid, start_other)
        contact = _first_contact_progress(
            start_unit,
            raw_unit,
            start_other,
            raw_other,
            desired_range,
        )
        if contact is not None:
            earliest_contact = min(earliest_contact, contact)

    if earliest_contact < 1.0:
        return _lerp(start_unit, raw_unit, earliest_contact)
    return raw_unit


def _first_contact_progress(
    start_unit: Point2D,
    raw_unit: Point2D,
    start_other: Point2D,
    raw_other: Point2D,
    desired_range: float,
) -> Optional[float]:
    if _distance(start_unit, start_other) <= desired_range + RANGE_EPSILON_MAP:
        return 0.0

    unit_dx = raw_unit[0] - start_unit[0]
    unit_dy = raw_unit[1] - start_unit[1]
    other_dx = raw_other[0] - start_other[0]
    other_dy = raw_other[1] - start_other[1]
    rel_start = (start_unit[0] - start_other[0], start_unit[1] - start_other[1])
    rel_delta = (unit_dx - other_dx, unit_dy - other_dy)
    rel_speed_sq = rel_delta[0] * rel_delta[0] + rel_delta[1] * rel_delta[1]
    if rel_speed_sq <= RANGE_EPSILON_MAP:
        return None

    closest = -(
        rel_start[0] * rel_delta[0] + rel_start[1] * rel_delta[1]
    ) / rel_speed_sq
    closest = max(0.0, min(1.0, closest))
    closest_distance = hypot(
        rel_start[0] + rel_delta[0] * closest,
        rel_start[1] + rel_delta[1] * closest,
    )
    if closest_distance > desired_range + RANGE_EPSILON_MAP:
        return None

    if closest <= RANGE_EPSILON_MAP:
        return 0.0
    lo = 0.0
    hi = closest
    for _ in range(24):
        mid = (lo + hi) / 2.0
        unit_mid = _lerp(start_unit, raw_unit, mid)
        other_mid = _lerp(start_other, raw_other, mid)
        if _distance(unit_mid, other_mid) <= desired_range:
            hi = mid
        else:
            lo = mid
    return hi


def _next_lord_position(
    unit: UnitInstance,
    units: list[UnitInstance],
    neutral_aggro_teams: Optional[set[str]],
    positions: dict[int, Point2D],
    battle=None,
) -> Point2D:
    ux, uy = _unit_point(unit, positions)
    fx, fy = _forward_vector(unit)
    attackable = _find_target(
        unit,
        units,
        neutral_aggro_teams=neutral_aggro_teams,
        in_range_only=True,
        positions=positions,
        ahead_only=True,
        battle=battle,
    )
    if attackable is None:
        attackable = _find_target(
            unit,
            units,
            neutral_aggro_teams=neutral_aggro_teams,
            in_range_only=True,
            positions=positions,
            ahead_only=False,
            battle=battle,
        )
    if attackable is not None:
        return ux, uy
    contact = _find_target(
        unit,
        units,
        neutral_aggro_teams=neutral_aggro_teams,
        in_range_only=False,
        positions=positions,
        ahead_only=True,
        battle=battle,
    )
    if contact is not None and _distance((ux, uy), _unit_point(contact, positions)) <= LORD_CONTACT_RANGE_MAP:
        return ux, uy
    if _battle_location_kind(battle) == "edge":
        return ux, uy
    return _next_lord_forward_position_without_crossing(
        unit,
        units,
        neutral_aggro_teams=neutral_aggro_teams,
        positions=positions,
    )


def _next_lord_forward_position_without_crossing(
    unit: UnitInstance,
    units: list[UnitInstance],
    neutral_aggro_teams: Optional[set[str]],
    positions: dict[int, Point2D],
) -> Point2D:
    ux, uy = _unit_point(unit, positions)
    fx, fy = _forward_vector(unit)
    step = _unit_step(unit)
    max_step = step
    for other in units:
        if not _is_hostile(unit, other, neutral_aggro_teams):
            continue
        ox, oy = _unit_point(other, positions)
        dx = ox - ux
        dy = oy - uy
        forward_distance = dx * fx + dy * fy
        if forward_distance <= RANGE_EPSILON_MAP:
            continue
        lateral_distance = abs(dx * -fy + dy * fx)
        if lateral_distance > LORD_CONTACT_RANGE_MAP:
            continue
        max_step = min(max_step, max(0.0, forward_distance - LORD_CONTACT_RANGE_MAP))
    return _clamp_point_to_world((ux + fx * max_step, uy + fy * max_step))


def _battle_location_kind(battle) -> str:
    if battle is None:
        return ""
    if isinstance(battle, dict):
        return str(battle.get("location_kind") or "")
    return str(getattr(battle, "location_kind", "") or "")


def _edge_progress_for_direction(battle, src: str, dst: str) -> float:
    progress = float(getattr(battle, "edge_position", 0.5) or 0.5)
    endpoints = getattr(battle, "edge_endpoints", None)
    if endpoints:
        edge_start, edge_end = endpoints
        if src == edge_start and dst == edge_end:
            return max(0.0, min(1.0, progress))
        if src == edge_end and dst == edge_start:
            return max(0.0, min(1.0, 1.0 - progress))
    return max(0.0, min(1.0, progress))


def _find_attack_target(
    unit: UnitInstance,
    units: list[UnitInstance],
    neutral_aggro_teams: Optional[set[str]],
    pending: dict[int, float],
    positions: dict[int, Point2D],
    battle=None,
) -> Optional[UnitInstance]:
    if not _is_allied_lord(unit):
        return _find_target(
            unit,
            units,
            neutral_aggro_teams=neutral_aggro_teams,
            in_range_only=True,
            pending=pending,
            positions=positions,
            battle=battle,
        )

    target = _find_target(
        unit,
        units,
        neutral_aggro_teams=neutral_aggro_teams,
        in_range_only=True,
        pending=pending,
        positions=positions,
        ahead_only=True,
        battle=battle,
    )
    if target is not None:
        return target
    return _find_target(
        unit,
        units,
        neutral_aggro_teams=neutral_aggro_teams,
        in_range_only=True,
        pending=pending,
        positions=positions,
        ahead_only=False,
        battle=battle,
    )


def _find_target(
    unit: UnitInstance,
    units: list[UnitInstance],
    neutral_aggro_teams: Optional[set[str]] = None,
    in_range_only: bool = True,
    pending: Optional[dict[int, float]] = None,
    positions: Optional[dict[int, Point2D]] = None,
    ahead_only: bool = False,
    battle=None,
) -> Optional[UnitInstance]:
    focus = None
    focus_key = (float("inf"), float("inf"), 0)
    best = None
    best_key = (float("inf"), 0)
    pending = pending or {}
    for other in units:
        if not _is_hostile(unit, other, neutral_aggro_teams):
            continue
        structure_role = _core_structure_target_role(unit, other, units, battle)
        if structure_role == "blocked":
            continue
        if ahead_only and not _is_ahead(unit, other, positions):
            continue
        if structure_role not in {"active_building", "active_layer"} and other.hp - pending.get(other.uid, 0.0) <= 0:
            continue
        pending_damage = pending.get(other.uid, 0.0)
        distance = _distance(_unit_point(unit, positions), _unit_point(other, positions))
        range_limit = unit_range_map(unit)
        if in_range_only and distance > range_limit + RANGE_EPSILON_MAP:
            if pending_damage <= 0 or distance > range_limit + FOCUS_ASSIST_RANGE_MAP + RANGE_EPSILON_MAP:
                continue
        if pending_damage > 0:
            remaining_hp = other.hp - pending_damage
            key = (remaining_hp, distance, other.uid)
            if key < focus_key:
                focus_key = key
                focus = other
            continue
        key = (distance, other.uid)
        if key < best_key:
            best_key = key
            best = other
    return focus or best


def _core_structure_target_role(
    unit: UnitInstance,
    other: UnitInstance,
    units: list[UnitInstance],
    battle,
) -> str:
    """Gate core assaults through base building slots.

    Higher slot index is the exposed layer: slot 2, then slot 1, then core. The
    current layer stays targetable even after pending lethal damage, so one
    combat tick cannot spill over into the next building or the core.
    """

    core_owner = _battle_core_owner(battle)
    if not core_owner:
        return "normal"
    if unit.team == core_owner or _combat_side(unit.team) == _combat_side(core_owner):
        return "normal"
    if other.team != core_owner:
        return "normal"

    # Troops are the outermost layer and stay engageable even once lethally
    # committed, so the whole assaulting front commits its damage to them. The
    # overflow pass then spills the surplus down through the buildings to the
    # core instead of letting one token defender soak the entire army.
    if not _is_building_unit(other) and other.unit_type != "CORE":
        return "active_layer"

    live_buildings = [
        candidate
        for candidate in units
        if (
            candidate.team == core_owner
            and _is_building_unit(candidate)
            and candidate.hp > 0
        )
    ]
    active_building = max(live_buildings, key=_building_slot_index) if live_buildings else None

    if _is_building_unit(other):
        if active_building is not None and other.uid == active_building.uid:
            return "active_building"
        return "blocked"
    if other.unit_type == "CORE" and active_building is not None:
        return "blocked"
    return "normal"


def _battle_core_owner(battle) -> Optional[str]:
    if battle is None:
        return None
    objective = battle.get("objective") if isinstance(battle, dict) else getattr(battle, "objective", None)
    if objective != "core":
        return None
    core_owner = battle.get("core_owner") if isinstance(battle, dict) else getattr(battle, "core_owner", None)
    return str(core_owner) if core_owner else None


def _is_building_unit(unit: UnitInstance) -> bool:
    return unit.unit_type in _BUILDING_UNIT_TYPES


def _building_slot_index(unit: UnitInstance) -> int:
    visual_id = str(getattr(unit, "visual_id", "") or "")
    parts = visual_id.split(":")
    if len(parts) >= 4 and parts[0] == "building":
        try:
            return int(parts[2])
        except ValueError:
            return 0
    return 0


def _queue_attack(
    unit: UnitInstance,
    target: UnitInstance,
    pending: dict[int, float],
    damage_log: dict[int, float],
    target_log: dict[int, int],
    attack_log: dict[int, list[tuple[int, float]]],
    damage: Optional[float] = None,
) -> None:
    value = float(unit.atk if damage is None else damage)
    pending[target.uid] = pending.get(target.uid, 0.0) + value
    damage_log[unit.uid] = damage_log.get(unit.uid, 0.0) + value
    target_log[unit.uid] = target.uid
    attack_log.setdefault(unit.uid, []).append((target.uid, value))


def _frame_snapshot(
    units: list[UnitInstance],
    damage_log: dict[int, float],
    target_log: dict[int, int],
    attack_log: dict[int, list[tuple[int, float]]],
) -> list[dict]:
    return [
        {
            "uid": unit.uid,
            "type": unit.unit_type,
            "team": unit.team,
            "x": round(unit.x, 2),
            "y": round(unit.y, 2),
            "pos": round(unit.pos, 2),
            "hp": round(unit.hp, 2),
            "max_hp": round(unit.max_hp, 2),
            "range": round(unit.range, 2),
            "range_map": round(unit_range_map(unit), 2),
            "atk": round(unit.atk, 2),
            "dmg": round(damage_log.get(unit.uid, 0.0), 2),
            "target": target_log.get(unit.uid),
            "attacks": [
                {"target": target_uid, "damage": round(damage, 2)}
                for target_uid, damage in attack_log.get(unit.uid, [])
            ],
        }
        for unit in units
    ]


def _is_hostile(
    a: UnitInstance,
    b: UnitInstance,
    neutral_aggro_teams: Optional[set[str]] = None,
) -> bool:
    if not a.is_alive() or not b.is_alive() or a.uid == b.uid:
        return False

    neutral_aggro_teams = neutral_aggro_teams or set()
    a_side = _combat_side(a.team)
    b_side = _combat_side(b.team)

    if a_side == b_side and a_side not in {"NEUTRAL", "BARON"}:
        return False
    if a_side == "NEUTRAL":
        return b.team in neutral_aggro_teams or b_side in neutral_aggro_teams
    if b_side == "NEUTRAL":
        return a.team in neutral_aggro_teams or a_side in neutral_aggro_teams
    if a_side == "BARON":
        return b_side in {"TEAM_A", "TEAM_B"}
    if b_side == "BARON":
        return a_side in {"TEAM_A", "TEAM_B"}
    return a_side != b_side


def _combat_side(team: str) -> str:
    if team in TEAM_A_PLAYERS:
        return "TEAM_A"
    if team in TEAM_B_PLAYERS:
        return "TEAM_B"
    return team


def _is_allied_lord(unit: UnitInstance) -> bool:
    return unit.unit_type == UnitType.LORD.value and unit.team in TEAM_A_PLAYERS + TEAM_B_PLAYERS


def _is_ahead(
    unit: UnitInstance,
    other: UnitInstance,
    positions: Optional[dict[int, Point2D]] = None,
) -> bool:
    fx, fy = _forward_vector(unit)
    ux, uy = _unit_point(unit, positions)
    ox, oy = _unit_point(other, positions)
    return (ox - ux) * fx + (oy - uy) * fy >= -RANGE_EPSILON_MAP


def _attack_rate(unit: UnitInstance) -> int:
    return max(1, int(getattr(unit, "attack_rate", 1) or 1))


def _attack_damage_sequence(unit: UnitInstance) -> list[float]:
    damages = [float(unit.atk)] * _attack_rate(unit)
    secondary_atk = float(getattr(unit, "secondary_atk", 0.0) or 0.0)
    secondary_rate = max(0, int(getattr(unit, "secondary_attack_rate", 0) or 0))
    if secondary_atk > 0 and secondary_rate > 0:
        damages.extend([secondary_atk] * secondary_rate)
    return damages


def _unit_step(unit: UnitInstance) -> float:
    return MOVEMENT_MAP_UNITS_PER_TICK * max(0.0, float(unit.speed))


def _unit_point(
    unit: UnitInstance,
    positions: Optional[dict[int, Point2D]] = None,
) -> Point2D:
    if positions is not None and unit.uid in positions:
        return positions[unit.uid]
    return float(unit.x), float(unit.y)


def _battle_node_center(battle) -> Point2D:
    return NODE_LAYOUT[battle.location_id].center


def _direction_between_nodes(src: str, dst: str) -> Point2D:
    start = NODE_LAYOUT[src].center
    end = NODE_LAYOUT[dst].center
    return _normalize((end[0] - start[0], end[1] - start[1]))


def _default_forward(player_id: str) -> Point2D:
    if player_id in TEAM_B_PLAYERS:
        return (-1.0, 0.0)
    return (1.0, 0.0)


def _forward_vector(unit: UnitInstance) -> Point2D:
    return _normalize((float(unit.forward_x or 0.0), float(unit.forward_y or 0.0)))


def _is_adjacent(src: str, dst: str) -> bool:
    try:
        get_approach(src, dst)
        return True
    except (KeyError, ValueError):
        return False


def _normalize(vector: Point2D) -> Point2D:
    x, y = vector
    length = hypot(x, y)
    if length <= 0.0:
        return 1.0, 0.0
    return x / length, y / length


def _distance(a: Point2D, b: Point2D) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _lerp(a: Point2D, b: Point2D, progress: float) -> Point2D:
    t = max(0.0, min(1.0, float(progress)))
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def _clamp_point_to_world(point: Point2D) -> Point2D:
    return (
        max(0.0, min(float(WORLD_SIZE[0]), float(point[0]))),
        max(0.0, min(float(WORLD_SIZE[1]), float(point[1]))),
    )
