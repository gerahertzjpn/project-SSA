import sys
from collections import Counter

sys.path.insert(0, "c:/project-SSA")

from math import hypot

import pytest

from src.core.combat import build_core_unit, build_player_units
from src.core.combat2d import (
    COMBAT_RANGE_TO_MAP_UNITS,
    EDGE_FRONTLINE_GAP_MAP,
    FORMATION_DEPTH_MAP,
    place_player_units_2d,
    tick_battle_units_2d,
)
from src.core.constants import (
    ActionType,
    BARON_ATTACK_RATE,
    BARON_ATK,
    BARON_SECONDARY_ATTACK_RATE,
    BARON_SECONDARY_ATK,
    CORE_ATTACK_RATE,
    EDGE_COLLISION_THRESHOLD,
    UnitType,
)
from src.core.engine import Action, GameEngine
from src.core.map2d import NODE_LAYOUT, approach_edge_length, point_on_approach_edge
from src.core.state import Battle, MovingGroup, make_initial_state
from src.core.units import UNIT_STATS


class QueueBot:
    def __init__(self, actions=None):
        self.actions = list(actions or [])

    def act(self, obs):
        if self.actions:
            return self.actions.pop(0)
        return Action(ActionType.HOLD)


def make_bots(actions_by_player=None):
    actions_by_player = actions_by_player or {}
    return {
        player_id: QueueBot(actions_by_player.get(player_id, []))
        for player_id in ["A1", "A2", "A3", "B1", "B2", "B3"]
    }


def run_until_battle(engine: GameEngine, battle_key: str, max_ticks: int = 16):
    for _ in range(max_ticks):
        engine.tick()
        if battle_key in engine.state.active_battles:
            return engine.state.active_battles[battle_key]
    return None


def test_engine_battle_units_have_real_map_positions():
    state = make_initial_state(seed=501)
    state.nodes["A1"].units = {UnitType.VANGUARD.value: 7}
    state.nodes["CP1"].owner = "B1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.RANGER.value: 2}
    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(ActionType.MOVE, source="A1", destination="CP1")],
        }),
    )

    battle = run_until_battle(engine, "node:CP1")

    assert battle is not None
    assert all(unit.x or unit.y for unit in battle.units)
    assert any(unit.x != NODE_LAYOUT["CP1"].center[0] for unit in battle.units)
    formation = engine._battle_snapshots()[0]["formation"]
    assert len(formation[0]) == 11
    assert {row[8] for row in formation}
    assert {row[9] for row in formation}


def test_lord_approach_battle_uses_baron_center_and_approach_positions():
    state = make_initial_state(seed=502)
    state.nodes["CP1"].owner = "A1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.SKIRMISHER.value: 4}
    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(ActionType.MOVE, source="CP1", destination="Baron")],
        }),
    )

    battle = run_until_battle(engine, "node:Baron")

    assert battle is not None
    assert battle.location_kind == "approach"
    baron = next(unit for unit in battle.units if unit.unit_type == "BARON")
    attackers = [unit for unit in battle.units if unit.team == "A1"]
    assert (baron.x, baron.y) == NODE_LAYOUT["Baron"].center
    assert attackers
    assert any(
        hypot(unit.x - baron.x, unit.y - baron.y) > NODE_LAYOUT["Baron"].radius
        for unit in attackers
    )


def test_2d_combat_range_is_map_distance_based():
    state = make_initial_state(seed=503)
    state.nodes["A1"].units = {UnitType.RANGER.value: 4}
    state.nodes["CP1"].owner = "B1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.VANGUARD.value: 1}
    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(ActionType.MOVE, source="A1", destination="CP1")],
        }),
    )

    battle = run_until_battle(engine, "node:CP1")
    assert battle is not None
    ranger = next(unit for unit in battle.units if unit.unit_type == UnitType.RANGER.value)

    assert ranger.range == UNIT_STATS[UnitType.RANGER]["range"]
    assert COMBAT_RANGE_TO_MAP_UNITS * ranger.range == (
        COMBAT_RANGE_TO_MAP_UNITS * UNIT_STATS[UnitType.RANGER]["range"]
    )


def test_allied_lord_stops_when_core_is_ahead_in_attack_range():
    lord = build_player_units({UnitType.LORD.value: 1}, "A1", "attacker")[0]
    core = build_core_unit("B1", hp=80.0)
    lord.x = 100.0
    lord.y = 50.0
    lord.forward_x = 1.0
    lord.forward_y = 0.0
    core.x = lord.x + lord.range * COMBAT_RANGE_TO_MAP_UNITS - 5.0
    core.y = lord.y
    before = (lord.x, lord.y)

    frame = tick_battle_units_2d([lord, core], record_trace=True)

    assert (lord.x, lord.y) == before
    assert core.hp < core.max_hp
    lord_frame = next(row for row in frame if row["uid"] == lord.uid)
    assert lord_frame["attacks"]


def test_allied_lord_stops_when_enemy_unit_is_ahead_in_attack_range():
    lord = build_player_units({UnitType.LORD.value: 1}, "A1", "attacker")[0]
    enemy = build_player_units({UnitType.VANGUARD.value: 1}, "B1", "defender")[0]
    lord.x = 100.0
    lord.y = 50.0
    lord.forward_x = 1.0
    lord.forward_y = 0.0
    enemy.x = lord.x + 45.0
    enemy.y = lord.y
    before = (lord.x, lord.y)

    frame = tick_battle_units_2d([lord, enemy], record_trace=True)

    assert (lord.x, lord.y) == before
    assert enemy.hp < enemy.max_hp
    lord_frame = next(row for row in frame if row["uid"] == lord.uid)
    assert lord_frame["attacks"]


def test_2d_lord_uses_primary_and_secondary_shots():
    lord = build_player_units({UnitType.LORD.value: 1}, "A1", "attacker")[0]
    enemies = build_player_units({UnitType.VANGUARD.value: 6}, "B1", "defender")
    for index, enemy in enumerate(enemies):
        enemy.x = lord.x + 20.0 + index
        enemy.y = lord.y
        enemy.speed = 0.0

    frame = tick_battle_units_2d([lord, *enemies], record_trace=True)
    lord_frame = next(row for row in frame if row["uid"] == lord.uid)

    assert [attack["damage"] for attack in lord_frame["attacks"]] == (
        [float(BARON_ATK)] * BARON_ATTACK_RATE
        + [float(BARON_SECONDARY_ATK)] * BARON_SECONDARY_ATTACK_RATE
    )


def test_2d_core_uses_current_attack_rate():
    core = build_core_unit("A1")
    enemies = build_player_units({UnitType.RANGER.value: 3}, "B1", "defender")
    core.x = 100.0
    core.y = 100.0
    for index, enemy in enumerate(enemies):
        enemy.x = core.x + 20.0 + index
        enemy.y = core.y
        enemy.speed = 0.0

    frame = tick_battle_units_2d([core, *enemies], record_trace=True)
    core_frame = next(row for row in frame if row["uid"] == core.uid)

    assert len(core_frame["attacks"]) == CORE_ATTACK_RATE
    assert [attack["damage"] for attack in core_frame["attacks"]] == [float(core.atk)] * CORE_ATTACK_RATE


def test_allied_lord_does_not_free_walk_inside_edge_battle():
    lord = build_player_units({UnitType.LORD.value: 1}, "A1", "attacker")[0]
    enemy = build_player_units({UnitType.RANGER.value: 1}, "B1", "defender")[0]
    lord.x = 100.0
    lord.y = 50.0
    lord.forward_x = 1.0
    lord.forward_y = 0.0
    enemy.x = 500.0
    enemy.y = 50.0
    before = (lord.x, lord.y)
    battle = Battle(
        battle_id="edge:A1|CP1",
        location_kind="edge",
        location_id="A1|CP1",
        units=[lord, enemy],
        started_at_tick=0,
        last_tick=0,
    )

    tick_battle_units_2d([lord, enemy], record_trace=True, battle=battle)

    assert (lord.x, lord.y) == before


def test_allied_lord_returns_fire_inside_edge_battle_even_if_formation_is_behind():
    lord = build_player_units({UnitType.LORD.value: 1}, "A2", "attacker")[0]
    enemy = build_player_units({UnitType.RANGER.value: 1}, "B2", "defender")[0]
    lord.x = 451.84
    lord.y = 158.38
    lord.forward_x = 1.0
    lord.forward_y = -0.46
    enemy.x = 423.14
    enemy.y = 167.78
    before = (lord.x, lord.y)
    battle = Battle(
        battle_id="edge:Baron|CP5",
        location_kind="edge",
        location_id="Baron|CP5",
        units=[lord, enemy],
        started_at_tick=0,
        last_tick=0,
        edge_endpoints=("Baron", "CP5"),
        edge_position=0.789,
    )

    frame = tick_battle_units_2d([lord, enemy], record_trace=True, battle=battle)

    assert (lord.x, lord.y) == before
    assert enemy.hp < enemy.max_hp
    lord_frame = next(row for row in frame if row["uid"] == lord.uid)
    assert lord_frame["attacks"]


def test_allied_lord_stops_and_returns_fire_when_attacked_from_behind():
    lord = build_player_units({UnitType.LORD.value: 1}, "A1", "attacker")[0]
    enemy = build_player_units({UnitType.RANGER.value: 1}, "B1", "defender")[0]
    lord.x = 200.0
    lord.y = 100.0
    lord.forward_x = 1.0
    lord.forward_y = 0.0
    enemy.x = lord.x - 20.0
    enemy.y = lord.y
    enemy.speed = 0.0
    before = (lord.x, lord.y)

    frame = tick_battle_units_2d([lord, enemy], record_trace=True)

    assert (lord.x, lord.y) == before
    assert enemy.hp < enemy.max_hp
    lord_frame = next(row for row in frame if row["uid"] == lord.uid)
    assert lord_frame["attacks"]


def test_lord_is_placed_ahead_of_support_units_in_2d_formation():
    lord = build_player_units({UnitType.LORD.value: 1}, "A1", "attacker")[0]
    support = build_player_units({UnitType.VANGUARD.value: 2}, "A1", "attacker")
    battle = Battle(
        battle_id="node:CP5",
        location_kind="node",
        location_id="CP5",
        units=[],
        started_at_tick=0,
        last_tick=0,
    )

    place_player_units_2d(
        battle,
        [lord, *support],
        player_id="A1",
        src="CP3",
        dst="CP5",
        entry_side="attacker",
    )

    assert lord.x > max(unit.x for unit in support)


def test_2d_attackers_focus_damaged_targets_before_spreading():
    skirmishers = build_player_units({UnitType.SKIRMISHER.value: 4}, "A1", "attacker")
    vanguards = build_player_units({UnitType.VANGUARD.value: 2}, "B1", "defender")
    for index, unit in enumerate(skirmishers):
        unit.x = 100.0
        unit.y = 180.0 + index * 7.0
    for index, unit in enumerate(vanguards):
        unit.x = 116.0
        unit.y = 183.0 + index * 14.0

    frame = tick_battle_units_2d([*skirmishers, *vanguards], record_trace=True)

    skirmisher_targets = Counter(
        attack["target"]
        for row in frame
        if row["team"] == "A1"
        for attack in row["attacks"]
    )
    assert max(skirmisher_targets.values()) >= 4
    assert sum(unit.is_alive() for unit in vanguards) == 1


def test_edge_battle_placement_uses_shared_collision_position_for_both_directions():
    battle = Battle(
        battle_id="edge:Baron|CP5",
        location_kind="edge",
        location_id="Baron|CP5",
        units=[],
        started_at_tick=0,
        last_tick=0,
        edge_endpoints=("Baron", "CP5"),
        edge_position=0.789,
    )
    a_lord = build_player_units({UnitType.LORD.value: 1}, "A2", "attacker")
    b_ranger = build_player_units({UnitType.RANGER.value: 1}, "B2", "defender")

    place_player_units_2d(battle, a_lord, "A2", "Baron", "CP5", "attacker", progress=1.0)
    place_player_units_2d(battle, b_ranger, "B2", "CP5", "Baron", "defender", progress=0.422)

    collision = point_on_approach_edge("Baron", "CP5", 0.789)
    axis = (
        NODE_LAYOUT["CP5"].center[0] - NODE_LAYOUT["Baron"].center[0],
        NODE_LAYOUT["CP5"].center[1] - NODE_LAYOUT["Baron"].center[1],
    )
    axis_length = hypot(*axis)
    axis = (axis[0] / axis_length, axis[1] / axis_length)
    a_projection = (a_lord[0].x - collision[0]) * axis[0] + (a_lord[0].y - collision[1]) * axis[1]
    b_projection = (b_ranger[0].x - collision[0]) * axis[0] + (b_ranger[0].y - collision[1]) * axis[1]
    assert a_projection < 0.0
    assert b_projection > 0.0


def test_edge_battle_survivors_resume_with_directional_progress():
    state = make_initial_state(seed=505)
    engine = GameEngine(state, {}, record_trace=False, record_replay=True)
    battle = Battle(
        battle_id="edge:B1|CP6",
        location_kind="edge",
        location_id="B1|CP6",
        units=build_player_units({UnitType.RANGER.value: 1}, "A3", "attacker"),
        started_at_tick=state.tick,
        last_tick=state.tick,
        edge_endpoints=("B1", "CP6"),
        edge_position=0.82,
        objective="pvp",
    )
    battle.participant_routes["A3"] = ("CP6", "B1")
    battle.participant_paths["A3"] = ["CP6", "B1"]
    battle.participant_segment_indices["A3"] = 1
    battle.participant_speeds["A3"] = 1.0
    battle.participant_final_dsts["A3"] = "B1"
    battle.participant_intents["A3"] = "CAPTURE"

    engine._finish_edge_battle(battle)

    group = state.moving_groups[-1]
    assert group.src == "CP6"
    assert group.dst == "B1"
    assert group.progress == pytest.approx(0.18)


def test_node_enter_reinforcement_joins_approach_battle_at_target_boundary():
    state = make_initial_state(seed=506)
    engine = GameEngine(state, {}, record_trace=False, record_replay=True)
    battle = Battle(
        battle_id="node:B3",
        location_kind="approach",
        location_id="B3",
        units=[],
        started_at_tick=state.tick,
        last_tick=state.tick,
        approach_id="CP6->B3",
        edge_position=0.8,
        objective="pvp",
    )
    group = MovingGroup(
        player_id="B1",
        team="TEAM_B",
        units={UnitType.RANGER.value: 1},
        src="CP5",
        dst="B3",
        arrival_tick=state.tick + 1,
        intent="CAPTURE",
        path=["B1", "CP5", "B3"],
        segment_index=2,
        phase="node_enter",
        segment_node="B3",
        progress=0.0,
        speed=1.0,
        final_dst="B3",
    )

    engine._join_battle(battle, group, reinforced=True)

    unit = battle.units[-1]
    assert hypot(unit.x - NODE_LAYOUT["B3"].center[0], unit.y - NODE_LAYOUT["B3"].center[1]) < 70.0
    assert hypot(unit.x - NODE_LAYOUT["CP5"].center[0], unit.y - NODE_LAYOUT["CP5"].center[1]) > 100.0


def test_node_exit_reinforcement_joins_approach_battle_at_source_boundary():
    state = make_initial_state(seed=507)
    engine = GameEngine(state, {}, record_trace=False, record_replay=True)
    battle = Battle(
        battle_id="node:B1",
        location_kind="approach",
        location_id="B1",
        units=[],
        started_at_tick=state.tick,
        last_tick=state.tick,
        approach_id="CP5->B1",
        edge_position=0.8,
        objective="pvp",
    )
    group = MovingGroup(
        player_id="B1",
        team="TEAM_B",
        units={UnitType.VANGUARD.value: 1},
        src="B1",
        dst="CP5",
        arrival_tick=state.tick + 1,
        intent="CAPTURE",
        path=["B1", "CP5"],
        segment_index=0,
        phase="node_exit",
        segment_node="B1",
        progress=0.9,
        speed=1.0,
        final_dst="CP5",
    )

    engine._join_battle(battle, group, reinforced=True)

    unit = battle.units[-1]
    assert hypot(unit.x - NODE_LAYOUT["B1"].center[0], unit.y - NODE_LAYOUT["B1"].center[1]) < 70.0
    assert hypot(unit.x - NODE_LAYOUT["CP5"].center[0], unit.y - NODE_LAYOUT["CP5"].center[1]) > 100.0


def test_fast_2d_units_stop_at_first_contact_instead_of_crossing_mid_tick():
    skirmisher = build_player_units({UnitType.SKIRMISHER.value: 1}, "A1", "attacker")[0]
    vanguard = build_player_units({UnitType.VANGUARD.value: 1}, "B1", "defender")[0]
    skirmisher.x = 0.0
    skirmisher.y = 0.0
    skirmisher.forward_x = 1.0
    skirmisher.forward_y = 0.0
    skirmisher.speed = 3.0
    skirmisher.hp = skirmisher.max_hp = 40.0
    vanguard.x = 100.0
    vanguard.y = 0.0
    vanguard.forward_x = -1.0
    vanguard.forward_y = 0.0
    vanguard.speed = 3.0
    vanguard.hp = vanguard.max_hp = 40.0

    frame = tick_battle_units_2d([skirmisher, vanguard], record_trace=True)

    assert skirmisher.x < vanguard.x
    assert vanguard.x - skirmisher.x == pytest.approx(
        skirmisher.range * COMBAT_RANGE_TO_MAP_UNITS,
        abs=1e-4,
    )
    assert any(item["attacks"] for item in frame)


def test_node_entry_into_hostile_visitors_starts_before_center_overlap():
    state = make_initial_state(seed=504)
    state.nodes["CP5"].owner = "A1"
    state.nodes["CP5"].neutral_hp = 0
    state.nodes["CP5"].units = {UnitType.VANGUARD.value: 2}
    state.players["B1"].alive = False
    state.players["B1"].core_hp = 0.0
    state.nodes["B1"].owner = "NEUTRAL"
    state.nodes["B1"].units = {}
    state.nodes["B1"].visiting_units = {"B2": {UnitType.SKIRMISHER.value: 1}}
    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(ActionType.MOVE, source="CP5", destination="B1")],
        }),
    )

    battle = run_until_battle(engine, "node:B1", max_ticks=12)

    assert battle is not None
    assert battle.location_kind == "node"
    assert battle.objective == "pvp"
    attackers = [unit for unit in battle.units if unit.team == "A1"]
    assert attackers
    assert all(
        hypot(unit.x - NODE_LAYOUT["B1"].center[0], unit.y - NODE_LAYOUT["B1"].center[1])
        >= NODE_LAYOUT["B1"].radius - 1e-6
        for unit in attackers
    )
    assert not any(group.dst == "B1" and group.phase == "node_enter" for group in state.moving_groups)


def test_hostile_node_transit_groups_collide_inside_intermediate_node():
    state = make_initial_state(seed=505)
    state.nodes["CP1"].owner = "A1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.VANGUARD.value: 2}
    state.nodes["CP5"].owner = "B1"
    state.nodes["CP5"].neutral_hp = 0
    state.nodes["CP5"].units = {UnitType.VANGUARD.value: 2}
    state.nodes["CP3"].owner = "NEUTRAL"
    state.nodes["CP3"].neutral_hp = 0
    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(ActionType.MOVE, source="CP1", destination="CP5")],
            "B1": [Action(ActionType.MOVE, source="CP5", destination="CP1")],
        }),
    )

    battle = run_until_battle(engine, "node:CP3", max_ticks=12)

    assert battle is not None
    assert battle.location_kind == "node"
    assert battle.objective == "pvp"
    assert {unit.team for unit in battle.units if unit.is_alive()} == {"A1", "B1"}
    assert not any(group.phase == "node_transit" and group.segment_node == "CP3" for group in state.moving_groups)


def test_node_enter_collides_with_hostile_group_transiting_same_node():
    state = make_initial_state(seed=506)
    state.nodes["CP3"].neutral_hp = 0
    state.moving_groups = [
        MovingGroup(
            player_id="A1",
            team="TEAM_A",
            units={UnitType.VANGUARD.value: 1},
            src="CP1",
            dst="CP3",
            arrival_tick=10,
            path=["CP1", "CP3"],
            phase="node_enter",
            segment_node="CP3",
            speed=1.0,
            progress=0.0,
        ),
        MovingGroup(
            player_id="B1",
            team="TEAM_B",
            units={UnitType.VANGUARD.value: 1},
            src="CP5",
            dst="CP1",
            arrival_tick=10,
            path=["CP5", "CP3", "CP1"],
            phase="node_transit",
            segment_node="CP3",
            speed=1.0,
            progress=0.5,
        ),
    ]
    engine = GameEngine(state, make_bots())

    engine.tick()

    battle = state.active_battles.get("node:CP3")
    assert battle is not None
    assert battle.location_kind == "node"
    assert {unit.team for unit in battle.units if unit.is_alive()} == {"A1", "B1"}
    assert state.moving_groups == []


def test_edge_battle_forms_at_engagement_range_not_at_overlap():
    # Regression for replay bug 2: two head-on groups must engage when their fronts
    # reach combat range (+ the formation depth the battle introduces), BEFORE they
    # glide through each other. The legacy trigger only fired at progress-sum >= 0.95
    # (a ~4 map-unit gap on this edge), so the fronts overlapped and then snapped back
    # into formation — the visible "pass through, then run back".
    state = make_initial_state(seed=611)
    engine = GameEngine(state, {}, record_trace=False, record_replay=True)

    def grp(player, team, src, dst, progress):
        return MovingGroup(
            player_id=player, team=team, units={UnitType.VANGUARD.value: 2},
            src=src, dst=dst, arrival_tick=state.tick + 1, intent="CAPTURE",
            path=[src, dst], segment_index=1, phase="edge", progress=progress,
            speed=1.0, final_dst=dst,
        )

    engage_sum = engine._edge_engage_sum("CP2", "CP4")
    # Fires earlier than the legacy near-overlap threshold and well before overlap (1.0).
    assert engage_sum < 1.0 - EDGE_COLLISION_THRESHOLD
    # The gap at the trigger equals the engagement band, in map units.
    band = COMBAT_RANGE_TO_MAP_UNITS + FORMATION_DEPTH_MAP + EDGE_FRONTLINE_GAP_MAP
    gap = (1.0 - engage_sum) * approach_edge_length("CP2", "CP4")
    assert gap == pytest.approx(band, abs=1.0)

    a = grp("A1", "TEAM_A", "CP2", "CP4", 0.30)
    # Sum 0.50 < engage threshold: fronts still apart, no battle.
    assert engine._groups_collide_on_edge(a, grp("B2", "TEAM_B", "CP4", "CP2", 0.20)) is False
    # Sum ~0.57 >= engage threshold: fronts now within range, battle forms...
    b = grp("B2", "TEAM_B", "CP4", "CP2", 0.27)
    assert engine._groups_collide_on_edge(a, b) is True
    # ...and that is far short of overlap (sum 1.0): the groups never pass through.
    assert a.progress + b.progress < 1.0 - EDGE_COLLISION_THRESHOLD
