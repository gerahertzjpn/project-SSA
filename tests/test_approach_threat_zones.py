import sys

sys.path.insert(0, "c:/project-SSA")

from src.core.constants import ActionType, UnitType
from src.core.engine import Action, GameEngine
from src.core.state import make_initial_state


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


def run_until_battle(engine: GameEngine, battle_key: str, max_ticks: int = 12):
    for _ in range(max_ticks):
        engine.tick()
        if battle_key in engine.state.active_battles:
            return engine.state.active_battles[battle_key]
    return None


def test_lord_starts_approach_battle_before_group_enters_baron_node():
    state = make_initial_state(seed=301)
    state.nodes["CP1"].owner = "A1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.VANGUARD.value: 2}
    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(ActionType.MOVE, source="CP1", destination="Baron")],
        }),
    )

    battle = run_until_battle(engine, "node:Baron")

    assert battle is not None
    assert battle.location_kind == "approach"
    assert battle.location_id == "Baron"
    assert battle.objective == "baron"
    assert battle.approach_id == "CP1->Baron"
    assert battle.threat_zone_ids == ["Lord"]
    assert battle.edge_position < 1.0
    assert state.moving_groups == []

    started = next(
        event
        for tick in engine.replay_ticks
        for event in tick["events"]
        if event.get("type") == "BATTLE_STARTED"
    )
    assert started["battle_type"] == "approach_battle"
    assert started["approach_id"] == "CP1->Baron"
    assert started["threat_zone_ids"] == ["Lord"]


def test_enemy_core_starts_approach_battle_before_group_enters_core_node():
    state = make_initial_state(seed=302)
    state.nodes["CP1"].owner = "B1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.VANGUARD.value: 3}
    engine = GameEngine(
        state,
        make_bots({
            "B1": [Action(ActionType.MOVE, source="CP1", destination="A1")],
        }),
    )

    battle = run_until_battle(engine, "node:A1")

    assert battle is not None
    assert battle.location_kind == "approach"
    assert battle.location_id == "A1"
    assert battle.objective == "core"
    assert battle.core_owner == "A1"
    assert battle.approach_id == "CP1->A1"
    assert battle.threat_zone_ids == ["core:A1"]
    assert 0.0 <= battle.edge_position < 1.0


def test_allied_core_approach_does_not_start_core_threat_battle():
    state = make_initial_state(seed=303)
    state.nodes["CP1"].owner = "A1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.VANGUARD.value: 2}
    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(ActionType.MOVE, source="CP1", destination="A1")],
        }),
    )

    for _ in range(5):
        engine.tick()

    assert "node:A1" not in state.active_battles


def test_ranged_garrison_starts_approach_battle_before_node_entry():
    state = make_initial_state(seed=304)
    state.nodes["A1"].units = {UnitType.VANGUARD.value: 7}
    state.nodes["CP1"].owner = "B1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.RANGER.value: 1}
    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(ActionType.MOVE, source="A1", destination="CP1")],
        }),
    )

    battle = run_until_battle(engine, "node:CP1")

    assert battle is not None
    assert battle.location_kind == "approach"
    assert battle.location_id == "CP1"
    assert battle.objective == "pvp"
    assert battle.approach_id == "A1->CP1"
    assert battle.threat_zone_ids == ["ranged:CP1"]
    assert battle.edge_position < 1.0


def test_melee_garrison_does_not_start_ranged_approach_battle():
    state = make_initial_state(seed=305)
    state.nodes["A1"].units = {UnitType.VANGUARD.value: 7}
    state.nodes["CP1"].owner = "B1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.VANGUARD.value: 4}
    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(ActionType.MOVE, source="A1", destination="CP1")],
        }),
    )

    battle = run_until_battle(engine, "node:CP1")

    assert battle is not None
    assert battle.location_kind == "node"
    assert battle.approach_id is None
