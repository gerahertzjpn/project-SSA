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
    assert state.nodes["CP1"].owner == "NEUTRAL"
    assert state.nodes["CP1"].units == {}
