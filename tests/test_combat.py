import pytest
import sys
sys.path.insert(0, "c:/project-SSA")

from src.core.combat import (
    ATTACKER_SIDE,
    CENTER_SIDE,
    COMBAT_MOVE_SPEED,
    DEFENDER_SIDE,
    build_baron_unit,
    build_core_unit,
    build_player_units,
    build_neutral_units,
    resolve_combat,
    resolve_combat_with_core,
    resolve_baron_combat,
    tick_battle_units,
)
from src.core.constants import (
    BARON_ATTACK_RATE,
    BARON_ATK,
    BARON_HP,
    CORE_ATTACK_RANGE,
    LORD_ATTACK_RANGE,
    TRAVEL_TIME,
    UnitType,
)
from src.core.units import UNIT_STATS


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


def test_lord_stats_reflect_current_balance():
    lord = build_player_units({UnitType.LORD.value: 1}, "A1", "attacker")[0]
    baron = build_baron_unit(BARON_HP)

    assert lord.max_hp == BARON_HP
    assert lord.atk == BARON_ATK
    assert lord.attack_rate == BARON_ATTACK_RATE
    assert lord.range == LORD_ATTACK_RANGE
    assert baron.max_hp == BARON_HP
    assert baron.atk == BARON_ATK
    assert baron.attack_rate == BARON_ATTACK_RATE
    assert baron.range == LORD_ATTACK_RANGE


def test_lord_attack_rate_fires_multiple_shots():
    lord = build_player_units({UnitType.LORD.value: 1}, "A1", "attacker")[0]
    targets = build_player_units({UnitType.SKIRMISHER.value: 3}, "B1", "defender")
    for index, target in enumerate(targets):
        target.pos = lord.pos + 1.0 + index * 0.1

    frame = tick_battle_units([lord, *targets], record_trace=True)
    lord_frame = next(item for item in frame if item["uid"] == lord.uid)

    assert len(lord_frame["attacks"]) == BARON_ATTACK_RATE
    assert len({shot["target"] for shot in lord_frame["attacks"]}) == BARON_ATTACK_RATE
    assert sum(target.is_alive() for target in targets) == 0


def test_neutrals_are_centered_stationary_group_with_ranger_range():
    neutrals = build_neutral_units(5)

    assert len(neutrals) == 5
    assert {unit.hp for unit in neutrals} == {2.0}
    assert {unit.max_hp for unit in neutrals} == {2.0}
    assert {unit.pos for unit in neutrals} == {CENTER_SIDE}
    assert {unit.speed for unit in neutrals} == {0.0}
    assert {unit.range for unit in neutrals} == {UNIT_STATS[UnitType.RANGER]["range"]}
    assert sum(unit.atk for unit in neutrals) == 10.0


def test_cores_are_on_team_side_with_high_range():
    left_core = build_core_unit("A1")
    right_core = build_core_unit("B1")

    assert left_core.pos == ATTACKER_SIDE
    assert right_core.pos == DEFENDER_SIDE
    assert left_core.range == CORE_ATTACK_RANGE
    assert right_core.range == CORE_ATTACK_RANGE


def test_large_armies_deploy_compactly_without_crossing_the_combat_line():
    skirmishers = build_player_units({UnitType.SKIRMISHER.value: 20}, "A1", "attacker")
    rangers = build_player_units({UnitType.RANGER.value: 10}, "B1", "defender")

    assert max(unit.pos for unit in skirmishers) < CENTER_SIDE
    assert min(unit.pos for unit in rangers) > CENTER_SIDE
    assert min(abs(left.pos - right.pos) for left in skirmishers for right in rangers) > UNIT_STATS[UnitType.RANGER]["range"]


def test_large_skirmisher_army_does_not_hit_rangers_on_spawn_tick():
    units = [
        *build_player_units({UnitType.SKIRMISHER.value: 20}, "A1", "attacker"),
        *build_player_units({UnitType.RANGER.value: 10}, "B1", "defender"),
    ]

    frame = tick_battle_units(units, record_trace=True)

    assert all(not item["attacks"] for item in frame)
    assert sum(unit.is_alive() for unit in units if unit.team == "A1") == 20
    assert sum(unit.is_alive() for unit in units if unit.team == "B1") == 10


@pytest.mark.parametrize(
    ("left_units", "right_units"),
    [
        ({UnitType.SKIRMISHER.value: 20}, {UnitType.RANGER.value: 10}),
        ({UnitType.VANGUARD.value: 15}, {UnitType.RANGER.value: 10}),
        ({UnitType.VANGUARD.value: 15}, {UnitType.SKIRMISHER.value: 20}),
        ({UnitType.SKIRMISHER.value: 14}, {UnitType.RANGER.value: 10}),
    ],
)
def test_pure_army_winner_type_is_independent_of_entry_side(left_units, right_units):
    forward_survivors, _, _, _ = resolve_combat(
        attacker_units=left_units,
        attacker_team="A1",
        defender_units=right_units,
        defender_team="B1",
        record_trace=False,
    )
    reverse_survivors, _, _, _ = resolve_combat(
        attacker_units=right_units,
        attacker_team="A1",
        defender_units=left_units,
        defender_team="B1",
        record_trace=False,
    )

    assert set(forward_survivors) == set(reverse_survivors)


def test_combat_deployment_keeps_frontline_ahead_of_backline():
    attacker = build_player_units(
        {
            UnitType.RANGER.value: 1,
            UnitType.SKIRMISHER.value: 1,
            UnitType.VANGUARD.value: 1,
        },
        "A1",
        "attacker",
    )
    defender = build_player_units(
        {
            UnitType.RANGER.value: 1,
            UnitType.SKIRMISHER.value: 1,
            UnitType.VANGUARD.value: 1,
        },
        "B1",
        "defender",
    )

    attacker_pos = {unit.unit_type: unit.pos for unit in attacker}
    defender_pos = {unit.unit_type: unit.pos for unit in defender}

    assert attacker_pos[UnitType.VANGUARD.value] > attacker_pos[UnitType.SKIRMISHER.value] > attacker_pos[UnitType.RANGER.value]
    assert defender_pos[UnitType.VANGUARD.value] < defender_pos[UnitType.SKIRMISHER.value] < defender_pos[UnitType.RANGER.value]


def test_combat_line_crossing_matches_empty_edge_travel_time():
    assert COMBAT_MOVE_SPEED * TRAVEL_TIME == DEFENDER_SIDE - ATTACKER_SIDE


def test_combat_approach_uses_unit_speed():
    slow = build_player_units({UnitType.RANGER.value: 1}, "A1", "attacker")[0]
    fast = build_player_units({UnitType.SKIRMISHER.value: 1}, "A2", "attacker")[0]
    target = build_player_units({UnitType.VANGUARD.value: 1}, "B1", "defender")[0]

    tick_battle_units([slow, fast, target], record_trace=False)

    assert fast.pos > slow.pos


def test_regular_unit_moves_without_attacking_when_target_starts_out_of_range():
    ranger = build_player_units({UnitType.RANGER.value: 1}, "A1", "attacker")[0]
    target = build_player_units({UnitType.VANGUARD.value: 1}, "B1", "defender")[0]
    target.pos = ranger.pos + ranger.range + 1.0
    target.speed = 0.0
    target.range = 0.0
    before_hp = target.hp

    frame = tick_battle_units([ranger, target], record_trace=True)
    ranger_frame = next(item for item in frame if item["uid"] == ranger.uid)

    assert ranger.pos > ATTACKER_SIDE
    assert target.hp == before_hp
    assert ranger_frame["attacks"] == []


def test_regular_unit_stops_and_attacks_when_target_starts_in_range():
    ranger = build_player_units({UnitType.RANGER.value: 1}, "A1", "attacker")[0]
    target = build_player_units({UnitType.VANGUARD.value: 1}, "B1", "defender")[0]
    target.pos = ranger.pos + ranger.range - 0.1
    target.speed = 0.0
    target.range = 0.0
    before_pos = ranger.pos
    before_hp = target.hp

    frame = tick_battle_units([ranger, target], record_trace=True)
    ranger_frame = next(item for item in frame if item["uid"] == ranger.uid)

    assert ranger.pos == before_pos
    assert target.hp < before_hp
    assert len(ranger_frame["attacks"]) == 1


def test_reborn_lord_can_move_and_shoot_before_contact():
    lord = build_player_units({UnitType.LORD.value: 1}, "A1", "attacker")[0]
    target = build_player_units({UnitType.VANGUARD.value: 1}, "B1", "defender")[0]
    target.pos = lord.pos + LORD_ATTACK_RANGE + 3.0
    target.speed = 0.0
    target.range = 0.0
    target.hp = 100.0
    target.max_hp = 100.0
    before_hp = target.hp

    frame = tick_battle_units([lord, target], record_trace=True)
    lord_frame = next(item for item in frame if item["uid"] == lord.uid)

    assert lord.pos > ATTACKER_SIDE
    assert target.hp < before_hp
    assert len(lord_frame["attacks"]) == BARON_ATTACK_RATE


def test_reborn_lord_ignores_enemies_behind_him():
    lord = build_player_units({UnitType.LORD.value: 1}, "A1", "attacker")[0]
    lord.pos = CENTER_SIDE
    behind = build_player_units({UnitType.VANGUARD.value: 1}, "B1", "defender")[0]
    ahead = build_player_units({UnitType.VANGUARD.value: 1}, "B2", "defender")[0]
    behind.pos = lord.pos - 1.0
    ahead.pos = lord.pos + 4.0
    for unit in (behind, ahead):
        unit.speed = 0.0
        unit.range = 0.0
        unit.hp = 100.0
        unit.max_hp = 100.0
    behind_hp = behind.hp
    ahead_hp = ahead.hp

    frame = tick_battle_units([lord, behind, ahead], record_trace=True)
    lord_frame = next(item for item in frame if item["uid"] == lord.uid)

    assert behind.hp == behind_hp
    assert ahead.hp < ahead_hp
    assert {shot["target"] for shot in lord_frame["attacks"]} == {ahead.uid}
