import sys
sys.path.insert(0, "c:/project-SSA")

from src.core.constants import ActionType, UnitType, PLAYER_IDS, TRAVEL_TIME
from src.core.engine import Action, GameEngine
from src.core.combat import ATTACKER_SIDE, DEFENDER_SIDE
from src.core.state import MovingGroup, make_initial_state


class QueueBot:
    def __init__(self, actions=None):
        self.actions = list(actions or [])

    def act(self, obs):
        if self.actions:
            return self.actions.pop(0)
        return Action(action_type=ActionType.HOLD)


def make_bots(actions_by_player=None):
    actions_by_player = actions_by_player or {}
    return {
        player_id: QueueBot(actions_by_player.get(player_id, []))
        for player_id in PLAYER_IDS
    }


def run_ticks(engine, count):
    for _ in range(count):
        engine.tick()


DIRECT_MOVE_TICKS = TRAVEL_TIME * 2 + 1
EDGE_COLLISION_TICKS = TRAVEL_TIME + 1


def test_battle_is_persistent_not_instant():
    state = make_initial_state(seed=1)
    state.nodes["A1"].units = {UnitType.VANGUARD.value: 8}
    state.nodes["CP1"].owner = "B1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.VANGUARD.value: 8}

    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(ActionType.MOVE, source="A1", destination="CP1")],
        }),
    )

    engine.tick()
    run_ticks(engine, DIRECT_MOVE_TICKS)

    battle = state.active_battles.get("node:CP1")
    assert battle is not None
    assert battle.location_kind == "node"
    assert battle.started_at_tick < state.tick
    assert any(
        event.get("type") == "BATTLE_STARTED" and event.get("battle_type") == "node_battle"
        for tick in engine.replay_ticks
        for event in tick["events"]
    )


def test_opposing_groups_create_persistent_edge_battle():
    state = make_initial_state(seed=2)
    state.nodes["A1"].units = {UnitType.VANGUARD.value: 8}
    state.nodes["CP1"].owner = "B1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.VANGUARD.value: 8}

    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(ActionType.MOVE, source="A1", destination="CP1")],
            "B1": [Action(ActionType.MOVE, source="CP1", destination="A1")],
        }),
    )

    engine.tick()
    run_ticks(engine, EDGE_COLLISION_TICKS)

    edge_battles = [
        battle for battle in state.active_battles.values()
        if battle.location_kind == "edge"
    ]
    assert len(edge_battles) == 1
    assert edge_battles[0].location_id == "A1|CP1"


def test_reinforcement_joins_edge_battle():
    state = make_initial_state(seed=3)
    state.nodes["A1"].units = {UnitType.VANGUARD.value: 8}
    state.nodes["CP1"].owner = "B1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.VANGUARD.value: 8}

    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(ActionType.MOVE, source="A1", destination="CP1")],
            "B1": [Action(ActionType.MOVE, source="CP1", destination="A1")],
        }),
    )
    engine.tick()
    run_ticks(engine, EDGE_COLLISION_TICKS)

    battle = next(b for b in state.active_battles.values() if b.location_kind == "edge")
    before = len(battle.units)
    state.moving_groups.append(MovingGroup(
        player_id="A1",
        team="TEAM_A",
        units={UnitType.VANGUARD.value: 2},
        src="A1",
        dst="CP1",
        arrival_tick=state.tick + 1,
        started_tick=state.tick,
        travel_ticks=TRAVEL_TIME,
        progress=0.49,
        intent="CAPTURE",
    ))

    engine.tick()

    assert len(battle.units) > before
    assert any(
        event.get("type") == "BATTLE_REINFORCEMENT"
        and event.get("battle_type") == "edge_battle"
        for event in engine.replay_ticks[-1]["events"]
    )


def test_reinforcement_joins_node_battle():
    state = make_initial_state(seed=31)
    state.nodes["A1"].units = {UnitType.VANGUARD.value: 8}
    state.nodes["CP1"].owner = "B1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.VANGUARD.value: 8}

    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(ActionType.MOVE, source="A1", destination="CP1")],
        }),
    )
    engine.tick()
    run_ticks(engine, DIRECT_MOVE_TICKS)

    battle = state.active_battles["node:CP1"]
    before = len(battle.units)
    state.moving_groups.append(MovingGroup(
        player_id="A2",
        team="TEAM_A",
        units={UnitType.VANGUARD.value: 2},
        src="A2",
        dst="CP1",
        arrival_tick=state.tick,
        started_tick=state.tick,
        travel_ticks=TRAVEL_TIME,
        progress=1.0,
        intent="CAPTURE",
    ))

    engine.tick()

    assert len(battle.units) > before
    assert any(
        event.get("type") == "BATTLE_REINFORCEMENT"
        and event.get("battle_type") == "node_battle"
        for event in engine.replay_ticks[-1]["events"]
    )


def test_neutrals_passive_until_attacked():
    state = make_initial_state(seed=4)
    state.nodes["A1"].units = {UnitType.VANGUARD.value: 3}
    state.nodes["CP1"].owner = "NEUTRAL"
    state.nodes["CP1"].neutral_hp = 5

    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(
                ActionType.MOVE,
                source="A1",
                destination="CP1",
                intent="TRANSIT",
            )],
        }),
    )

    engine.tick()
    run_ticks(engine, DIRECT_MOVE_TICKS)

    assert state.nodes["CP1"].neutral_hp == 5
    assert "A1" in state.nodes["CP1"].visiting_units
    assert not state.active_battles


def test_arrived_units_cannot_leave_node_until_next_tick():
    state = make_initial_state(seed=14)
    state.nodes["A1"].units = {UnitType.VANGUARD.value: 2}
    state.nodes["CP1"].owner = "NEUTRAL"
    state.nodes["CP1"].neutral_hp = 0

    engine = GameEngine(
        state,
        make_bots({
            "A1": [
                Action(ActionType.MOVE, source="A1", destination="CP1"),
                *[Action(ActionType.HOLD) for _ in range(DIRECT_MOVE_TICKS - 1)],
                Action(ActionType.MOVE, source="CP1", destination="CP3"),
                Action(ActionType.MOVE, source="CP1", destination="CP3"),
            ],
        }),
    )

    engine.tick()
    run_ticks(engine, DIRECT_MOVE_TICKS)

    assert state.nodes["CP1"].owner == "A1"
    assert state.nodes["CP1"].units == {UnitType.VANGUARD.value: 2}
    assert state.moving_groups == []

    engine.tick()

    assert state.nodes["CP1"].units == {}
    assert len(state.moving_groups) == 1
    assert state.moving_groups[0].src == "CP1"
    assert state.moving_groups[0].dst == "CP3"


def test_node_remembers_arrival_side_for_stationed_units():
    state = make_initial_state(seed=15)
    state.nodes["CP1"].owner = "B1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.VANGUARD.value: 1}
    state.nodes["CP3"].owner = "NEUTRAL"
    state.nodes["CP3"].neutral_hp = 0

    engine = GameEngine(
        state,
        make_bots({
            "B1": [Action(ActionType.MOVE, source="CP1", destination="CP3")],
        }),
    )

    engine.tick()
    run_ticks(engine, DIRECT_MOVE_TICKS)

    assert state.nodes["CP3"].owner == "B1"
    assert state.nodes["CP3"].unit_entry_side["B1"] == "attacker"


def test_long_move_builds_shortest_route_and_walks_through_node():
    state = make_initial_state(seed=17)
    state.nodes["A3"].units = {UnitType.VANGUARD.value: 2}
    state.nodes["CP1"].owner = "NEUTRAL"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP3"].owner = "NEUTRAL"
    state.nodes["CP3"].neutral_hp = 0

    engine = GameEngine(
        state,
        make_bots({
            "A3": [Action(ActionType.MOVE, source="A3", destination="CP3")],
        }),
    )

    engine.tick()
    group = state.moving_groups[0]

    assert group.path == ["A3", "CP1", "CP3"]
    assert group.final_dst == "CP3"
    assert group.phase == "node_exit"

    run_ticks(engine, TRAVEL_TIME + 2)
    group = state.moving_groups[0]

    assert group.phase == "node_transit"
    assert group.segment_node == "CP1"
    assert group.src == "A3"
    assert group.dst == "CP3"


def test_guarded_neutrals_are_valid_transit_nodes():
    state = make_initial_state(seed=19)
    engine = GameEngine(state, make_bots())

    assert state.nodes["CP1"].owner == "NEUTRAL"
    assert state.nodes["CP1"].neutral_hp > 0
    assert state.nodes["CP3"].owner == "NEUTRAL"
    assert state.nodes["CP3"].neutral_hp > 0

    assert engine._find_route("A1", "A1", "CP3") == ["A1", "CP1", "CP3"]


def test_enemy_frontline_limits_closed_core_to_adjacent_targets():
    state = make_initial_state(seed=19)
    state.nodes["CP1"].owner = "B1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP2"].owner = "B1"
    state.nodes["CP2"].neutral_hp = 0
    engine = GameEngine(state, make_bots())

    reachable = {
        node_id
        for node_id in state.nodes
        if engine._find_route("A1", "A1", node_id)
    }

    assert reachable == {"CP1", "CP2"}


def test_route_does_not_transit_enemy_nodes_or_living_lord():
    state = make_initial_state(seed=20)
    state.nodes["CP1"].owner = "A1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP3"].owner = "B1"
    state.nodes["CP3"].neutral_hp = 0
    state.nodes["CP5"].owner = "NEUTRAL"
    state.nodes["CP5"].neutral_hp = 0

    engine = GameEngine(state, make_bots())

    assert engine._find_route("A1", "CP1", "CP5") is None

    state.baron.hp = 0
    assert engine._find_route("A1", "CP1", "CP5") == ["CP1", "Baron", "CP5"]


def test_movement_speed_uses_slowest_unit_in_group():
    state = make_initial_state(seed=18)
    state.nodes["A1"].units = {UnitType.SKIRMISHER.value: 2}
    state.nodes["A2"].units = {UnitType.SKIRMISHER.value: 2, UnitType.RANGER.value: 1}
    state.nodes["CP1"].owner = "NEUTRAL"
    state.nodes["CP1"].neutral_hp = 0

    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(ActionType.MOVE, source="A1", destination="CP1")],
            "A2": [Action(ActionType.MOVE, source="A2", destination="CP1")],
        }),
    )

    engine.tick()
    groups = {group.player_id: group for group in state.moving_groups}

    assert groups["A1"].speed == 1.5
    assert groups["A2"].speed == 0.5
    assert groups["A1"].travel_ticks < groups["A2"].travel_ticks


def test_stationed_units_use_saved_entry_side_in_node_battle():
    state = make_initial_state(seed=16)
    state.nodes["CP3"].owner = "B1"
    state.nodes["CP3"].neutral_hp = 0
    state.nodes["CP3"].units = {UnitType.VANGUARD.value: 1}
    state.nodes["CP3"].unit_entry_side = {"B1": "attacker"}
    state.nodes["CP5"].owner = "A1"
    state.nodes["CP5"].neutral_hp = 0
    state.nodes["CP5"].units = {UnitType.VANGUARD.value: 1}

    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(ActionType.MOVE, source="CP5", destination="CP3")],
        }),
    )

    engine.tick()
    run_ticks(engine, DIRECT_MOVE_TICKS)

    battle = state.active_battles["node:CP3"]
    b_unit = next(unit for unit in battle.units if unit.team == "B1")
    a_unit = next(unit for unit in battle.units if unit.team == "A1")

    assert battle.participant_entry_sides["B1"] == "attacker"
    assert battle.participant_entry_sides["A1"] == "defender"
    assert b_unit.pos == ATTACKER_SIDE
    assert a_unit.pos == DEFENDER_SIDE


def test_neutral_aggro_set():
    state = make_initial_state(seed=5)
    state.nodes["A1"].units = {UnitType.VANGUARD.value: 8}
    state.nodes["CP1"].owner = "NEUTRAL"
    state.nodes["CP1"].neutral_hp = 7

    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(
                ActionType.MOVE,
                source="A1",
                destination="CP1",
                intent="CAPTURE",
            )],
        }),
    )

    engine.tick()
    run_ticks(engine, DIRECT_MOVE_TICKS)

    battle = state.active_battles["node:CP1"]
    assert battle.neutral_aggro_teams == {"TEAM_A"}
    assert state.nodes["CP1"].neutral_aggro_teams == {"TEAM_A"}


def test_capture_requires_neutral_clear():
    state = make_initial_state(seed=6)
    state.nodes["A1"].units = {UnitType.VANGUARD.value: 1}
    state.nodes["CP1"].owner = "NEUTRAL"
    state.nodes["CP1"].neutral_hp = 7

    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(
                ActionType.MOVE,
                source="A1",
                destination="CP1",
                intent="CAPTURE",
            )],
        }),
    )

    engine.tick()
    run_ticks(engine, TRAVEL_TIME + 60)

    assert state.nodes["CP1"].owner == "NEUTRAL"
    assert state.nodes["CP1"].neutral_hp > 0


def test_baron_blocks_transit():
    state = make_initial_state(seed=7)
    state.nodes["CP1"].owner = "A1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.VANGUARD.value: 2}

    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(
                ActionType.MOVE,
                source="CP1",
                destination="Baron",
                intent="TRANSIT",
            )],
        }),
    )

    engine.tick()

    assert state.nodes["CP1"].units == {UnitType.VANGUARD.value: 2}
    assert state.moving_groups == []
    assert any(event.get("type") == "BARON_BLOCKED" for event in engine.replay_ticks[-1]["events"])


def test_replay_has_node_and_edge_battle_events():
    state = make_initial_state(seed=8)
    state.nodes["A1"].units = {UnitType.VANGUARD.value: 8}
    state.nodes["CP1"].owner = "B1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.VANGUARD.value: 8}

    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(ActionType.MOVE, source="A1", destination="CP1")],
            "B1": [Action(ActionType.MOVE, source="CP1", destination="A1")],
        }),
    )
    engine.tick()
    run_ticks(engine, EDGE_COLLISION_TICKS)

    event_types = [
        event.get("battle_type")
        for tick in engine.replay_ticks
        for event in tick["events"]
        if event.get("type") in {"BATTLE_STARTED", "BATTLE_TICK", "BATTLE_REINFORCEMENT"}
    ]
    assert "edge_battle" in event_types

    state = make_initial_state(seed=9)
    state.nodes["A1"].units = {UnitType.VANGUARD.value: 8}
    state.nodes["CP1"].owner = "B1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.VANGUARD.value: 8}
    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(ActionType.MOVE, source="A1", destination="CP1")],
        }),
    )
    engine.tick()
    run_ticks(engine, DIRECT_MOVE_TICKS)

    event_types = [
        event.get("battle_type")
        for tick in engine.replay_ticks
        for event in tick["events"]
        if event.get("type") in {"BATTLE_STARTED", "BATTLE_TICK", "BATTLE_REINFORCEMENT"}
    ]
    assert "node_battle" in event_types
