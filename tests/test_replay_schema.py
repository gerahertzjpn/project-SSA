import sys
sys.path.insert(0, "c:/project-SSA")

from src.core.constants import ActionType, UnitType, PLAYER_IDS, TRAVEL_TIME
from src.core.engine import Action, GameEngine
from src.core.replay import (
    COMBAT_TRACE_FORMAT,
    REPLAY_SCHEMA_VERSION,
    load_replay,
    save_replay,
)
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
        for player_id in PLAYER_IDS
    }


def make_node_battle_engine():
    state = make_initial_state(seed=41)
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
    return state, engine


def test_save_replay_writes_versioned_schema(tmp_path):
    state, engine = make_node_battle_engine()
    for _ in range(TRAVEL_TIME + 1):
        engine.tick()

    path = tmp_path / "replay.json"
    save_replay(
        winner=state.winner or "DRAW",
        total_ticks=state.tick,
        ticks=engine.replay_ticks,
        path=str(path),
    )

    replay = load_replay(str(path))
    assert replay["meta"]["schema_version"] == REPLAY_SCHEMA_VERSION
    assert replay["meta"]["combat_trace_format"] == COMBAT_TRACE_FORMAT
    assert replay["meta"]["total_ticks"] == state.tick
    assert "unit_ready_tick" in replay["ticks"][-1]["nodes"]["CP1"]
    assert "unit_entry_side" in replay["ticks"][-1]["nodes"]["CP1"]


def test_engine_replay_uses_compact_battle_tick_summary():
    _, engine = make_node_battle_engine()
    for _ in range(TRAVEL_TIME + 2):
        engine.tick()

    battle_tick = next(
        event
        for tick in engine.replay_ticks
        for event in tick["events"]
        if event.get("type") == "BATTLE_TICK"
    )

    assert "frame" not in battle_tick
    assert battle_tick["summary"]["alive"]
    assert "damage_by_team" in battle_tick["summary"]
    assert "attacks" in battle_tick["summary"]
    assert "deaths" in battle_tick["summary"]
    if battle_tick["summary"]["attacks"]:
        assert len(battle_tick["summary"]["attacks"][0]) == 7


def test_active_battle_snapshot_uses_side_summary():
    _, engine = make_node_battle_engine()
    for _ in range(TRAVEL_TIME + 1):
        engine.tick()

    snapshots = [
        battle
        for tick in engine.replay_ticks
        for battle in tick["active_battles"]
    ]
    assert snapshots
    assert "sides" in snapshots[-1]
    assert "entry_sides" in snapshots[-1]
    assert "formation" in snapshots[-1]
    assert isinstance(snapshots[-1]["formation"][0], list)
    assert len(snapshots[-1]["formation"][0]) == 7
    assert "units" not in snapshots[-1]
