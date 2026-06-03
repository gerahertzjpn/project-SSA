import sys
sys.path.insert(0, "c:/project-SSA")

from src.core.constants import ActionType, UnitType
from src.core.engine import Action, GameEngine
from src.core.production import apply_production, collect_income
from src.core.state import Battle, MovingGroup, make_initial_state
from src.core.combat import build_player_units


class CountingBot:
    def __init__(self):
        self.calls = 0

    def act(self, obs):
        self.calls += 1
        return Action(ActionType.PRODUCE, unit_type=UnitType.VANGUARD)


def make_engine_with_counting_bot(state, player_id="A1"):
    bots = {pid: CountingBot() for pid in state.players}
    return GameEngine(state, bots), bots[player_id]


def test_dead_player_gets_no_income_and_cannot_produce():
    state = make_initial_state(seed=51)
    state.players["A1"].resources = 10
    state.players["A1"].alive = False
    state.nodes["CP1"].owner = "A1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].neutral_income = 5

    collect_income(state)

    assert state.players["A1"].resources == 10
    assert apply_production(state, "A1", UnitType.VANGUARD) is False
    assert state.nodes["A1"].units == {UnitType.VANGUARD.value: 2}


def test_dead_player_bot_is_not_called():
    state = make_initial_state(seed=52)
    state.players["A1"].alive = False
    engine, bot = make_engine_with_counting_bot(state, "A1")

    engine.tick()

    assert bot.calls == 0


def test_eliminated_player_units_are_removed_from_owned_nodes_visits_moves_and_battles():
    state = make_initial_state(seed=53)
    engine = GameEngine(state, bots={})

    state.nodes["CP1"].owner = "A1"
    state.nodes["CP1"].units = {UnitType.VANGUARD.value: 4}
    state.nodes["CP2"].owner = "A2"
    state.nodes["CP2"].visiting_units["A1"] = {UnitType.ARCHER.value: 2}
    state.moving_groups.append(MovingGroup(
        player_id="A1",
        team="TEAM_A",
        units={UnitType.SKIRMISHER.value: 3},
        src="CP1",
        dst="CP3",
        arrival_tick=state.tick + 3,
    ))
    state.active_battles["node:CP3"] = Battle(
        battle_id="node:CP3",
        location_kind="node",
        location_id="CP3",
        units=build_player_units({UnitType.VANGUARD.value: 2}, "A1", "attacker"),
        started_at_tick=state.tick,
        last_tick=state.tick,
    )

    engine._remove_player_units("A1")

    assert state.nodes["CP1"].owner == "NEUTRAL"
    assert state.nodes["CP1"].units == {}
    assert "A1" not in state.nodes["CP2"].visiting_units
    assert all(group.player_id != "A1" for group in state.moving_groups)
    assert all(not unit.is_alive() for unit in state.active_battles["node:CP3"].units)


def test_core_destroyed_marks_player_dead_and_removes_existing_units():
    state = make_initial_state(seed=54)
    engine = GameEngine(state, bots={})
    state.nodes["A1"].owner = "A1"
    state.nodes["A1"].units = {UnitType.VANGUARD.value: 2}
    state.nodes["CP1"].owner = "A1"
    state.nodes["CP1"].units = {UnitType.VANGUARD.value: 4}

    battle = Battle(
        battle_id="node:A1",
        location_kind="node",
        location_id="A1",
        units=build_player_units({UnitType.VANGUARD.value: 5}, "B1", "defender"),
        started_at_tick=state.tick,
        last_tick=state.tick,
        objective="core",
        core_owner="A1",
    )

    event = engine._finish_node_battle(battle)

    assert event["core_destroyed"] is True
    assert state.players["A1"].alive is False
    assert state.nodes["A1"].owner == "NEUTRAL"
    assert state.nodes["A1"].units == {}
    assert state.nodes["A1"].visiting_units["B1"] == {UnitType.VANGUARD.value: 5}
    assert state.nodes["CP1"].owner == "NEUTRAL"
    assert state.nodes["CP1"].units == {}


def test_destroyed_core_cannot_be_reinforced_or_captured_by_arrival():
    state = make_initial_state(seed=55)
    engine = GameEngine(state, bots={})
    state.players["A3"].alive = False
    state.players["A3"].core_hp = 0.0
    state.nodes["A3"].owner = "B2"
    state.nodes["A3"].units = {}

    group = MovingGroup(
        player_id="A2",
        team="TEAM_A",
        units={UnitType.SKIRMISHER.value: 1},
        src="CP2",
        dst="A3",
        arrival_tick=state.tick,
    )

    event = engine._handle_core_arrival(group)

    assert event["type"] == "MOVE_IN"
    assert event["core_destroyed"] is True
    assert state.nodes["A3"].owner == "NEUTRAL"
    assert state.nodes["A3"].units == {}
    assert state.nodes["A3"].visiting_units == {
        "A2": {UnitType.SKIRMISHER.value: 1},
    }


def test_destroyed_core_pvp_resolution_keeps_core_neutral():
    state = make_initial_state(seed=56)
    engine = GameEngine(state, bots={})
    state.players["A3"].alive = False
    state.players["A3"].core_hp = 0.0
    state.nodes["A3"].owner = "NEUTRAL"
    state.nodes["A3"].units = {}

    battle = Battle(
        battle_id="node:A3",
        location_kind="node",
        location_id="A3",
        units=build_player_units({UnitType.VANGUARD.value: 2}, "B2", "defender"),
        started_at_tick=state.tick,
        last_tick=state.tick,
        objective="pvp",
    )

    event = engine._finish_node_battle(battle)

    assert event["result"] == "DEAD_CORE_PVP_RESOLVED"
    assert state.nodes["A3"].owner == "NEUTRAL"
    assert state.nodes["A3"].units == {}
    assert state.nodes["A3"].visiting_units == {
        "B2": {UnitType.VANGUARD.value: 2},
    }
