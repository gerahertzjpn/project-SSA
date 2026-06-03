"""
Tactical combat helpers (v0 — lightweight 1D simulator).

Both sides arranged on a 1D line [0..20]:
  - Attackers start at the low end (0), defenders at the high end (20).
  - Front slots (vanguards) closest to enemy, back slots (archers) furthest.
  - Each combat tick: units advance, attack nearest enemy in range, damage simultaneous.
  - Neutrals: single garrison unit (defend a point, high range so they fire first).
  - Core: stationary defender at the very back with HP + counterattack.

The public resolve_* functions are kept for focused unit tests and quick balance probes.
The engine uses the tickable helpers below to keep battles persistent in GameState:
node/edge battles live across world ticks and reinforcements join the existing battle.

record_trace=False skips _frame_snapshot entirely — use during RL training (millions of
clashes) where the agent only needs the outcome, not the per-tick position data.
"""

from typing import Optional

from .constants import (
    UnitType, CORE_HP, CORE_ATK, BARON_ATK, BARON_ATTACK_RATE, MAX_COMBAT_TICKS,
    TEAM_A_PLAYERS, TEAM_B_PLAYERS, TRAVEL_TIME, LORD_ATTACK_RANGE, CORE_ATTACK_RANGE,
    NEUTRAL_GUARD_HP,
)
from .units import UnitInstance, UNIT_STATS

_uid_counter = 0


def _new_uid() -> int:
    global _uid_counter
    _uid_counter += 1
    return _uid_counter


ATTACKER_SIDE = 0.0
DEFENDER_SIDE = 20.0
CENTER_SIDE = (ATTACKER_SIDE + DEFENDER_SIDE) / 2.0
DEPLOYMENT_DEPTH = 4.0
SIM_SLOT_SPACING = 0.08
RENDER_SLOT_SPACING = 0.32
COMBAT_MOVE_SPEED = (DEFENDER_SIDE - ATTACKER_SIDE) / TRAVEL_TIME
NEUTRAL_RANGE = float(UNIT_STATS[UnitType.RANGER]["range"])
LORD_RANGE = LORD_ATTACK_RANGE
CORE_RANGE = CORE_ATTACK_RANGE
LORD_CONTACT_RANGE = 0.5

_SLOT_ORDER = {
    UnitType.VANGUARD.value: 0,
    UnitType.LORD.value: 0,
    UnitType.SKIRMISHER.value: 1,
    UnitType.RANGER.value: 2,
    "ARCHER": 2,
}


# ---------------------------------------------------------------------------
# Unit construction helpers
# ---------------------------------------------------------------------------

def _build_side(units_dict: dict, team: str, is_attacker: bool) -> list[UnitInstance]:
    base = ATTACKER_SIDE if is_attacker else DEFENDER_SIDE
    direction = 1.0 if is_attacker else -1.0

    all_units = []
    for unit_type_val, count in units_dict.items():
        unit_type_enum = _unit_type_enum(unit_type_val)
        unit_type_val = unit_type_enum.value
        stats = UNIT_STATS.get(unit_type_enum, UNIT_STATS[UnitType.VANGUARD])
        slot_order = _SLOT_ORDER.get(unit_type_val, 1)
        for index in range(int(count or 0)):
            all_units.append((slot_order, unit_type_val, stats, index, int(count or 0)))
    all_units.sort(key=lambda x: (x[0], x[1], x[3]))

    instances = []
    for _, unit_type_val, stats, same_type_index, same_type_count in all_units:
        anchor = _deployment_anchor(stats)
        sim_offset = _formation_offset(
            same_type_index,
            same_type_count,
            anchor,
            SIM_SLOT_SPACING,
        )
        render_offset = _formation_offset(
            same_type_index,
            same_type_count,
            anchor,
            RENDER_SLOT_SPACING,
        )
        pos = base + direction * (anchor + sim_offset)
        instances.append(UnitInstance(
            uid=_new_uid(),
            unit_type=unit_type_val,
            team=team,
            hp=float(stats["hp"]),
            max_hp=float(stats["hp"]),
            atk=float(stats["atk"]),
            range=float(stats["range"]),
            speed=float(stats["speed"]),
            pos=pos,
            attack_rate=int(stats.get("attack_rate", 1)),
            move_direction=direction,
            render_offset=direction * (render_offset - sim_offset),
        ))
    return instances


def _deployment_anchor(stats: dict) -> float:
    return max(0.0, min(DEPLOYMENT_DEPTH, float(stats.get("base_pos", DEPLOYMENT_DEPTH / 2.0))))


def _formation_offset(index: int, count: int, anchor: float, spacing: float) -> float:
    if count <= 1:
        return 0.0
    max_width = max(0.0, min(anchor, DEPLOYMENT_DEPTH - anchor) * 2.0)
    if max_width <= 0.0:
        return 0.0
    width = min(max_width, spacing * (count - 1))
    actual_spacing = width / max(1, count - 1)
    return (index - (count - 1) / 2.0) * actual_spacing


def render_position(unit: UnitInstance) -> float:
    return max(ATTACKER_SIDE, min(DEFENDER_SIDE, unit.pos + float(getattr(unit, "render_offset", 0.0) or 0.0)))


def _unit_type_enum(unit_type_val: str) -> UnitType:
    if unit_type_val == "ARCHER":
        return UnitType.RANGER
    return UnitType(unit_type_val)


def _build_neutrals(neutral_hp: int) -> list[UnitInstance]:
    if neutral_hp <= 0:
        return []
    total_atk = float(neutral_hp * 2)
    per_unit_atk = total_atk / neutral_hp
    return [
        UnitInstance(
            uid=_new_uid(),
            unit_type="NEUTRAL",
            team="NEUTRAL",
            hp=float(NEUTRAL_GUARD_HP),
            max_hp=float(NEUTRAL_GUARD_HP),
            atk=per_unit_atk,
            range=NEUTRAL_RANGE,
            speed=0.0,
            pos=CENTER_SIDE,
            attack_rate=1,
            move_direction=0.0,
        )
        for _ in range(neutral_hp)
    ]


def _build_core(player_id: str, hp: float | None = None) -> UnitInstance:
    pos = ATTACKER_SIDE if player_id in TEAM_A_PLAYERS else DEFENDER_SIDE
    current_hp = float(CORE_HP if hp is None else hp)
    return UnitInstance(
        uid=_new_uid(),
        unit_type="CORE",
        team=player_id,
        hp=current_hp,
        max_hp=float(CORE_HP),
        atk=float(CORE_ATK),
        range=CORE_RANGE,
        speed=0.0,
        pos=pos,
        attack_rate=1,
        move_direction=0.0,
    )


def _build_baron(baron_hp: int) -> UnitInstance:
    return UnitInstance(
        uid=_new_uid(),
        unit_type="BARON",
        team="BARON",
        hp=float(baron_hp),
        max_hp=float(baron_hp),
        atk=float(BARON_ATK),
        range=LORD_RANGE,
        speed=0.0,
        pos=CENTER_SIDE,
        attack_rate=BARON_ATTACK_RATE,
        move_direction=0.0,
    )


# ---------------------------------------------------------------------------
# Targeting and snapshot
# ---------------------------------------------------------------------------

def _find_target(
    unit: UnitInstance,
    enemies: list[UnitInstance],
    pending: Optional[dict[int, float]] = None,
    positions: Optional[dict[int, float]] = None,
) -> Optional[UnitInstance]:
    best = None
    best_dist = float("inf")
    pending = pending or {}
    for e in enemies:
        if not e.is_alive():
            continue
        if e.hp - pending.get(e.uid, 0.0) <= 0:
            continue
        dist = abs(_unit_pos(unit, positions) - _unit_pos(e, positions))
        if dist <= unit.range and dist < best_dist:
            best_dist = dist
            best = e
    return best


def _unit_pos(unit: UnitInstance, positions: Optional[dict[int, float]] = None) -> float:
    if positions is None:
        return unit.pos
    return float(positions.get(unit.uid, unit.pos))


def _nearest_target(
    unit: UnitInstance,
    enemies: list[UnitInstance],
    positions: Optional[dict[int, float]] = None,
) -> Optional[UnitInstance]:
    best = None
    best_dist = float("inf")
    for enemy in enemies:
        if not enemy.is_alive():
            continue
        dist = abs(_unit_pos(unit, positions) - _unit_pos(enemy, positions))
        if dist < best_dist:
            best_dist = dist
            best = enemy
    return best


def _next_position_toward_target(
    unit: UnitInstance,
    target: Optional[UnitInstance],
    positions: Optional[dict[int, float]] = None,
) -> float:
    if target is None or unit.speed <= 0:
        return unit.pos
    unit_pos = _unit_pos(unit, positions)
    target_pos = _unit_pos(target, positions)
    dist = abs(target_pos - unit_pos)
    if dist <= unit.range:
        return unit.pos
    direction = 1.0 if target_pos > unit_pos else -1.0
    step = min(COMBAT_MOVE_SPEED * unit.speed, max(0.0, dist - unit.range))
    return unit_pos + direction * step


def _move_toward_target(unit: UnitInstance, target: Optional[UnitInstance]) -> None:
    unit.pos = _next_position_toward_target(unit, target)


def _is_allied_lord(unit: UnitInstance) -> bool:
    return unit.unit_type == UnitType.LORD.value and unit.team in TEAM_A_PLAYERS + TEAM_B_PLAYERS


def _forward_direction(unit: UnitInstance) -> float:
    direction = float(getattr(unit, "move_direction", 0.0) or 0.0)
    if direction:
        return 1.0 if direction > 0 else -1.0
    return 1.0 if _combat_side(unit.team) == "TEAM_A" else -1.0


def _is_ahead(
    unit: UnitInstance,
    other: UnitInstance,
    positions: Optional[dict[int, float]] = None,
) -> bool:
    return (_unit_pos(other, positions) - _unit_pos(unit, positions)) * _forward_direction(unit) >= -1e-6


def _find_allied_lord_target(
    unit: UnitInstance,
    all_units: list[UnitInstance],
    neutral_aggro_teams: Optional[set[str]] = None,
    in_range_only: bool = True,
    pending: Optional[dict[int, float]] = None,
    positions: Optional[dict[int, float]] = None,
) -> Optional[UnitInstance]:
    best = None
    best_dist = float("inf")
    pending = pending or {}
    for other in all_units:
        if not _is_hostile(unit, other, neutral_aggro_teams):
            continue
        if not _is_ahead(unit, other, positions):
            continue
        if other.hp - pending.get(other.uid, 0.0) <= 0:
            continue
        dist = abs(_unit_pos(unit, positions) - _unit_pos(other, positions))
        if in_range_only and dist > unit.range:
            continue
        if dist < best_dist:
            best_dist = dist
            best = other
    return best


def _next_allied_lord_forward_position(
    unit: UnitInstance,
    all_units: list[UnitInstance],
    neutral_aggro_teams: Optional[set[str]] = None,
    positions: Optional[dict[int, float]] = None,
) -> float:
    if unit.speed <= 0:
        return unit.pos

    target = _find_allied_lord_target(
        unit,
        all_units,
        neutral_aggro_teams=neutral_aggro_teams,
        in_range_only=False,
        positions=positions,
    )
    if target is not None:
        unit_pos = _unit_pos(unit, positions)
        target_pos = _unit_pos(target, positions)
        dist = abs(target_pos - unit_pos)
        if dist <= LORD_CONTACT_RANGE:
            return unit.pos
        step = min(COMBAT_MOVE_SPEED * unit.speed, max(0.0, dist - LORD_CONTACT_RANGE))
    else:
        step = COMBAT_MOVE_SPEED * unit.speed

    return max(ATTACKER_SIDE, min(DEFENDER_SIDE, _unit_pos(unit, positions) + _forward_direction(unit) * step))


def _move_allied_lord_forward(
    unit: UnitInstance,
    all_units: list[UnitInstance],
    neutral_aggro_teams: Optional[set[str]] = None,
) -> None:
    unit.pos = _next_allied_lord_forward_position(
        unit,
        all_units,
        neutral_aggro_teams=neutral_aggro_teams,
    )


def _combat_side(team: str) -> str:
    if team in TEAM_A_PLAYERS:
        return "TEAM_A"
    if team in TEAM_B_PLAYERS:
        return "TEAM_B"
    return team


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


def _find_persistent_target(
    unit: UnitInstance,
    all_units: list[UnitInstance],
    neutral_aggro_teams: Optional[set[str]] = None,
    in_range_only: bool = True,
    pending: Optional[dict[int, float]] = None,
    positions: Optional[dict[int, float]] = None,
) -> Optional[UnitInstance]:
    best = None
    best_dist = float("inf")
    pending = pending or {}
    for other in all_units:
        if not _is_hostile(unit, other, neutral_aggro_teams):
            continue
        if other.hp - pending.get(other.uid, 0.0) <= 0:
            continue
        dist = abs(_unit_pos(unit, positions) - _unit_pos(other, positions))
        if in_range_only and dist > unit.range:
            continue
        if dist < best_dist:
            best_dist = dist
            best = other
    return best


def _attack_rate(unit: UnitInstance) -> int:
    return max(1, int(getattr(unit, "attack_rate", 1) or 1))


def _queue_attack(
    unit: UnitInstance,
    target: UnitInstance,
    pending: dict[int, float],
    damage_log: dict[int, float],
    target_log: dict[int, int],
    attack_log: dict[int, list[tuple[int, float]]],
) -> None:
    pending[target.uid] = pending.get(target.uid, 0.0) + unit.atk
    damage_log[unit.uid] = damage_log.get(unit.uid, 0.0) + unit.atk
    target_log[unit.uid] = target.uid
    attack_log.setdefault(unit.uid, []).append((target.uid, unit.atk))


def _frame_snapshot(
    all_units: list[UnitInstance],
    damage_log: dict,
    target_log: Optional[dict[int, int]] = None,
    attack_log: Optional[dict[int, list[tuple[int, float]]]] = None,
) -> list:
    target_log = target_log or {}
    attack_log = attack_log or {}
    return [
        {
            "uid": u.uid,
            "type": u.unit_type,
            "team": u.team,
            "pos": round(u.pos, 2),
            "hp": round(u.hp, 2),
            "max_hp": round(u.max_hp, 2),
            "range": round(u.range, 2),
            "atk": round(u.atk, 2),
            "render_pos": round(render_position(u), 2),
            "dmg": round(damage_log.get(u.uid, 0.0), 2),
            "target": target_log.get(u.uid),
            "attacks": [
                {"target": target_uid, "damage": round(damage, 2)}
                for target_uid, damage in attack_log.get(u.uid, [])
            ],
        }
        for u in all_units
    ]


def build_player_units(units_dict: dict, player_id: str, entry_side: str) -> list[UnitInstance]:
    """Build persistent battle units for a player entering from attacker/defender side."""
    return _build_side(units_dict, player_id, is_attacker=entry_side == "attacker")


def build_neutral_units(neutral_hp: int) -> list[UnitInstance]:
    return _build_neutrals(neutral_hp)


def build_core_unit(player_id: str, hp: float | None = None) -> UnitInstance:
    return _build_core(player_id, hp=hp)


def build_baron_unit(baron_hp: int) -> UnitInstance:
    return _build_baron(baron_hp)


def tick_battle_units(
    units: list[UnitInstance],
    record_trace: bool = True,
    neutral_aggro_teams: Optional[set[str]] = None,
) -> list:
    """
    Advance a persistent battle by exactly one combat tick.
    Mutates units in place and returns one replay frame, or [] when trace is disabled.
    """
    alive = [u for u in units if u.is_alive()]
    if not alive:
        return []

    attack_ready = {
        unit.uid
        for unit in alive
        if not _is_allied_lord(unit)
        and _find_persistent_target(
            unit,
            alive,
            neutral_aggro_teams=neutral_aggro_teams,
            in_range_only=True,
        )
        is not None
    }

    start_positions = {unit.uid: unit.pos for unit in alive}
    planned_positions: dict[int, float] = {}
    for unit in alive:
        if unit.speed <= 0:
            continue
        if _is_allied_lord(unit):
            planned_positions[unit.uid] = _next_allied_lord_forward_position(
                unit,
                alive,
                neutral_aggro_teams=neutral_aggro_teams,
                positions=start_positions,
            )
            continue
        if unit.uid in attack_ready:
            continue
        target = _find_persistent_target(
            unit,
            alive,
            neutral_aggro_teams=neutral_aggro_teams,
            in_range_only=False,
            positions=start_positions,
        )
        if target is None:
            continue
        planned_positions[unit.uid] = _next_position_toward_target(
            unit,
            target,
            positions=start_positions,
        )

    for unit in alive:
        if unit.uid in planned_positions:
            unit.pos = planned_positions[unit.uid]

    pending: dict[int, float] = {}
    damage_log: dict[int, float] = {}
    target_log: dict[int, int] = {}
    attack_log: dict[int, list[tuple[int, float]]] = {}
    alive = [u for u in units if u.is_alive()]

    for unit in alive:
        for _ in range(_attack_rate(unit)):
            if _is_allied_lord(unit):
                target = _find_allied_lord_target(
                    unit,
                    alive,
                    neutral_aggro_teams=neutral_aggro_teams,
                    in_range_only=True,
                    pending=pending,
                )
            elif unit.uid in attack_ready:
                target = _find_persistent_target(
                    unit,
                    alive,
                    neutral_aggro_teams=neutral_aggro_teams,
                    in_range_only=True,
                    pending=pending,
                )
            else:
                target = None
            if target is None:
                break
            _queue_attack(unit, target, pending, damage_log, target_log, attack_log)

    for unit in units:
        if unit.uid in pending:
            unit.hp -= pending[unit.uid]

    return _frame_snapshot(units, damage_log, target_log, attack_log) if record_trace else []


def has_hostile_units(
    units: list[UnitInstance],
    neutral_aggro_teams: Optional[set[str]] = None,
) -> bool:
    alive = [u for u in units if u.is_alive()]
    for i, unit in enumerate(alive):
        for other in alive[i + 1:]:
            if _is_hostile(unit, other, neutral_aggro_teams):
                return True
    return False


def count_survivors_by_player(units: list[UnitInstance]) -> dict[str, dict]:
    result: dict[str, dict[str, int]] = {}
    for unit in units:
        if not unit.is_alive() or unit.unit_type in ("NEUTRAL", "CORE", "BARON"):
            continue
        result.setdefault(unit.team, {})
        result[unit.team][unit.unit_type] = result[unit.team].get(unit.unit_type, 0) + 1
    return result


def alive_player_ids(units: list[UnitInstance]) -> list[str]:
    players = []
    for unit in units:
        if unit.is_alive() and unit.unit_type not in ("NEUTRAL", "CORE", "BARON"):
            if unit.team not in players:
                players.append(unit.team)
    return players


def alive_combat_sides(units: list[UnitInstance]) -> set[str]:
    return {_combat_side(pid) for pid in alive_player_ids(units)}


# ---------------------------------------------------------------------------
# Core combat loop (single source of truth)
# ---------------------------------------------------------------------------

def _run_combat(
    attackers: list[UnitInstance],
    defenders: list[UnitInstance],
    record_trace: bool,
) -> tuple[list[UnitInstance], list[UnitInstance], list]:
    """
    Run the tick loop until one side is eliminated or MAX_COMBAT_TICKS is reached.
    Returns (alive_attackers, alive_defenders, trace).
    trace is [] when record_trace=False.
    """
    all_units = attackers + defenders
    trace: list = []

    for _ in range(MAX_COMBAT_TICKS):
        alive_atk = [u for u in attackers if u.is_alive()]
        alive_def = [u for u in defenders if u.is_alive()]

        if not alive_atk or not alive_def:
            break

        attack_ready_atk = {
            u.uid
            for u in alive_atk
            if not _is_allied_lord(u) and _find_target(u, alive_def) is not None
        }
        attack_ready_def = {
            u.uid
            for u in alive_def
            if not _is_allied_lord(u) and _find_target(u, alive_atk) is not None
        }

        start_positions = {u.uid: u.pos for u in alive_atk + alive_def}
        planned_positions: dict[int, float] = {}
        for u in alive_atk:
            if _is_allied_lord(u):
                planned_positions[u.uid] = _next_allied_lord_forward_position(
                    u,
                    all_units,
                    positions=start_positions,
                )
            elif u.uid not in attack_ready_atk:
                planned_positions[u.uid] = _next_position_toward_target(
                    u,
                    _nearest_target(u, alive_def, positions=start_positions),
                    positions=start_positions,
                )
        for u in alive_def:
            if _is_allied_lord(u):
                planned_positions[u.uid] = _next_allied_lord_forward_position(
                    u,
                    all_units,
                    positions=start_positions,
                )
            elif u.uid not in attack_ready_def:
                planned_positions[u.uid] = _next_position_toward_target(
                    u,
                    _nearest_target(u, alive_atk, positions=start_positions),
                    positions=start_positions,
                )
        for u in alive_atk + alive_def:
            if u.uid in planned_positions:
                u.pos = planned_positions[u.uid]

        pending: dict[int, float] = {}
        damage_log: dict[int, float] = {}
        target_log: dict[int, int] = {}
        attack_log: dict[int, list[tuple[int, float]]] = {}

        for u in alive_atk:
            for _ in range(_attack_rate(u)):
                if _is_allied_lord(u):
                    t = _find_allied_lord_target(u, all_units, in_range_only=True, pending=pending)
                elif u.uid in attack_ready_atk:
                    t = _find_target(u, alive_def, pending=pending)
                else:
                    t = None
                if not t:
                    break
                _queue_attack(u, t, pending, damage_log, target_log, attack_log)

        for u in alive_def:
            for _ in range(_attack_rate(u)):
                if _is_allied_lord(u):
                    t = _find_allied_lord_target(u, all_units, in_range_only=True, pending=pending)
                elif u.uid in attack_ready_def:
                    t = _find_target(u, alive_atk, pending=pending)
                else:
                    t = None
                if not t:
                    break
                _queue_attack(u, t, pending, damage_log, target_log, attack_log)

        for unit in all_units:
            if unit.uid in pending:
                unit.hp -= pending[unit.uid]

        if record_trace:
            trace.append(_frame_snapshot(all_units, damage_log, target_log, attack_log))

    return (
        [u for u in attackers if u.is_alive()],
        [u for u in defenders if u.is_alive()],
        trace,
    )


def _count_survivors(units: list[UnitInstance]) -> dict:
    result: dict[str, int] = {}
    for u in units:
        if u.unit_type in ("NEUTRAL", "CORE", "BARON"):
            continue
        result[u.unit_type] = result.get(u.unit_type, 0) + 1
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_combat(
    attacker_units: dict,
    attacker_team: str,
    defender_units: dict,
    defender_team: str,
    neutral_hp: int = 0,
    record_trace: bool = True,
) -> tuple[dict, str, list, int]:
    """
    Resolve a clash at a neutral CP or enemy-held CP.
    Returns (survivors, winner_team, trace, remaining_neutral_hp).
    remaining_neutral_hp > 0 only when winner_team == "NEUTRAL" and neutrals survived.
    """
    attackers = _build_side(attacker_units, attacker_team, is_attacker=True)

    if neutral_hp > 0:
        defenders = _build_neutrals(neutral_hp)
        defender_team = "NEUTRAL"
    else:
        defenders = _build_side(defender_units, defender_team, is_attacker=False)

    if not attackers:
        return defender_units.copy(), defender_team, [], 0
    if not defenders:
        return attacker_units.copy(), attacker_team, [], 0

    alive_atk, alive_def, trace = _run_combat(attackers, defenders, record_trace)

    if alive_atk and not alive_def:
        return _count_survivors(alive_atk), attacker_team, trace, 0
    if neutral_hp > 0:
        remaining = sum(1 for u in alive_def if u.unit_type == "NEUTRAL")
        return {}, "NEUTRAL", trace, remaining
    return _count_survivors([u for u in alive_def if u.unit_type != "CORE"]), defender_team, trace, 0


def resolve_combat_with_core(
    attacker_units: dict,
    attacker_team: str,
    defender_units: dict,
    defender_team: str,
    core_owner: str,
    record_trace: bool = True,
) -> tuple[dict, str, list, bool]:
    """
    Resolve an attack on an enemy core node.
    Returns (survivors, winner_team, trace, core_destroyed).
    """
    attackers = _build_side(attacker_units, attacker_team, is_attacker=True)
    defenders = _build_side(defender_units, defender_team, is_attacker=False)
    core_unit = _build_core(core_owner)
    defenders.append(core_unit)

    if not attackers:
        return defender_units.copy(), defender_team, [], False

    alive_atk, alive_def, trace = _run_combat(attackers, defenders, record_trace)
    core_alive = core_unit.is_alive()

    if alive_atk and not alive_def:
        return _count_survivors(alive_atk), attacker_team, trace, True
    survivors = _count_survivors([u for u in alive_def if u.unit_type != "CORE"])
    return survivors, defender_team, trace, not core_alive


def resolve_baron_combat(
    attacker_units: dict,
    attacker_team: str,
    baron_hp: int,
    record_trace: bool = True,
) -> tuple[dict, bool, list]:
    """
    Attack the Baron entity.
    Returns (survivors, baron_killed, trace).
    """
    baron = _build_baron(baron_hp)
    attackers = _build_side(attacker_units, attacker_team, is_attacker=True)

    if not attackers:
        return {}, False, []

    alive_atk, _, trace = _run_combat(attackers, [baron], record_trace)
    baron_killed = not baron.is_alive()
    return _count_survivors(alive_atk), baron_killed, trace
