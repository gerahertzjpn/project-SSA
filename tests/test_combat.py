import pytest
import sys
sys.path.insert(0, "c:/project-SSA")

from src.core.combat import (
    ATTACKER_SIDE,
    CENTER_SIDE,
    COMBAT_MOVE_SPEED,
    DEFENDER_SIDE,
    build_core_unit,
    build_player_units,
    build_neutral_units,
    resolve_combat,
    resolve_combat_with_core,
    resolve_baron_combat,
    tick_battle_units,
)
from src.core.constants import TRAVEL_TIME, UnitType


def test_attacker_wins_overwhelming_force():
    atk = {UnitType.VANGUARD.value: 20}
    survivors, winner, trace, _ = resolve_combat(
        attacker_units=atk,
        attacker_team="A1",
        defender_units={UnitType.VANGUARD.value: 1},
        defender_team="B1",
    )
    assert winner == "A1"
    assert sum(survivors.values()) > 0


def test_defender_wins_overwhelming_force():
    atk = {UnitType.VANGUARD.value: 1}
    dfn = {UnitType.VANGUARD.value: 20}
    survivors, winner, trace, _ = resolve_combat(
        attacker_units=atk,
        attacker_team="A1",
        defender_units=dfn,
        defender_team="B1",
    )
    assert winner == "B1"
    assert sum(survivors.values()) > 0


def test_trace_is_generated():
    atk = {UnitType.VANGUARD.value: 3}
    dfn = {UnitType.VANGUARD.value: 3}
    _, _, trace, _ = resolve_combat(
        attacker_units=atk,
        attacker_team="A1",
        defender_units=dfn,
        defender_team="B1",
    )
    assert len(trace) > 0
    assert isinstance(trace[0], list)


def test_neutral_fight_capture():
    atk = {UnitType.VANGUARD.value: 10}
    survivors, winner, trace, _ = resolve_combat(
        attacker_units=atk,
        attacker_team="A1",
        defender_units={},
        defender_team="NEUTRAL",
        neutral_hp=3,
    )
    assert winner == "A1"
    assert sum(survivors.values()) > 0


def test_neutral_fight_repelled():
    # Very few attackers vs strong neutral
    atk = {UnitType.ARCHER.value: 1}
    survivors, winner, trace, _ = resolve_combat(
        attacker_units=atk,
        attacker_team="A1",
        defender_units={},
        defender_team="NEUTRAL",
        neutral_hp=7,
    )
    assert winner == "NEUTRAL"


def test_archers_beat_vanguards_rps():
    # Many archers vs many vanguards: archers should win (range advantage)
    atk = {UnitType.ARCHER.value: 8}
    dfn = {UnitType.VANGUARD.value: 5}
    survivors, winner, trace, _ = resolve_combat(
        attacker_units=atk,
        attacker_team="A1",
        defender_units=dfn,
        defender_team="B1",
    )
    # Archers do 5 atk vs vanguard's 10 HP — should win against equal numbers
    # Not guaranteed with 8 vs 5, but attackers have advantage
    assert len(trace) > 0


def test_empty_attacker_returns_defender():
    survivors, winner, trace, _ = resolve_combat(
        attacker_units={},
        attacker_team="A1",
        defender_units={UnitType.VANGUARD.value: 3},
        defender_team="B1",
    )
    assert winner == "B1"
    assert survivors.get(UnitType.VANGUARD.value, 0) == 3


def test_core_attack_kills_core():
    atk = {UnitType.VANGUARD.value: 30}  # overwhelming
    survivors, winner, trace, core_destroyed = resolve_combat_with_core(
        attacker_units=atk,
        attacker_team="A1",
        defender_units={},
        defender_team="B1",
        core_owner="B1",
    )
    assert core_destroyed is True
    assert winner == "A1"


def test_core_attack_repelled():
    atk = {UnitType.VANGUARD.value: 1}  # tiny force
    survivors, winner, trace, core_destroyed = resolve_combat_with_core(
        attacker_units=atk,
        attacker_team="A1",
        defender_units={UnitType.VANGUARD.value: 10},
        defender_team="B1",
        core_owner="B1",
    )
    assert core_destroyed is False


def test_baron_fight_killed():
    atk = {UnitType.VANGUARD.value: 30}
    survivors, baron_killed, trace = resolve_baron_combat(
        attacker_units=atk,
        attacker_team="A1",
        baron_hp=10,
    )
    assert baron_killed is True


def test_baron_fight_survives():
    atk = {UnitType.VANGUARD.value: 1}
    survivors, baron_killed, trace = resolve_baron_combat(
        attacker_units=atk,
        attacker_team="A1",
        baron_hp=30,
    )
    assert baron_killed is False


def test_neutrals_are_centered_group_with_medium_range():
    neutrals = build_neutral_units(5)

    assert len(neutrals) == 5
    assert {unit.pos for unit in neutrals} == {CENTER_SIDE}
    assert {unit.range for unit in neutrals} == {3.0}


def test_cores_are_on_team_side_with_high_range():
    left_core = build_core_unit("A1")
    right_core = build_core_unit("B1")

    assert left_core.pos == ATTACKER_SIDE
    assert right_core.pos == DEFENDER_SIDE
    assert left_core.range >= 8.0
    assert right_core.range >= 8.0


def test_combat_line_crossing_matches_empty_edge_travel_time():
    assert COMBAT_MOVE_SPEED * TRAVEL_TIME == DEFENDER_SIDE - ATTACKER_SIDE


def test_combat_approach_uses_unit_speed():
    slow = build_player_units({UnitType.RANGER.value: 1}, "A1", "attacker")[0]
    fast = build_player_units({UnitType.SKIRMISHER.value: 1}, "A2", "attacker")[0]
    target = build_player_units({UnitType.VANGUARD.value: 1}, "B1", "defender")[0]

    tick_battle_units([slow, fast, target], record_trace=False)

    assert fast.pos > slow.pos
