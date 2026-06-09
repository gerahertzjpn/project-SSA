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


DIRECT_MOVE_TICKS = TRAVEL_TIME * 2 + 1


def test_save_replay_writes_versioned_schema(tmp_path):
    state, engine = make_node_battle_engine()
    for _ in range(DIRECT_MOVE_TICKS + 1):
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
    assert "neutral_max_hp" in replay["ticks"][-1]["nodes"]["CP1"]
    assert "visual_units" in replay["ticks"][-1]
    assert replay["ticks"][-1]["visual_units"]


def test_engine_replay_uses_compact_battle_tick_summary():
    _, engine = make_node_battle_engine()
    for _ in range(DIRECT_MOVE_TICKS + 2):
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
    for _ in range(DIRECT_MOVE_TICKS + 1):
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
    assert len(snapshots[-1]["formation"][0]) == 11
    assert isinstance(snapshots[-1]["formation"][0][8], (int, float))
    assert isinstance(snapshots[-1]["formation"][0][9], (int, float))
    assert "units" not in snapshots[-1]


def test_replay_moving_group_includes_stable_started_tick():
    _, engine = make_node_battle_engine()

    engine.tick()
    moving = engine.replay_ticks[-1]["moving_groups"][0]
    observed = engine.observe("A1")["moving_groups"][0]

    assert moving["player"] == "A1"
    assert moving["started"] == engine.replay_ticks[-1]["tick"]
    assert moving["team"] == "TEAM_A"
    assert moving["segment_src"] == "A1"
    assert moving["segment_dst"] == "CP1"
    assert moving["lane_id"] == "A1|CP1"
    assert moving["approach_id"] == "A1->CP1"
    assert moving["visual"]["lane_id"] == moving["lane_id"]
    assert moving["visual"]["approach_id"] == moving["approach_id"]
    assert moving["visual"]["phase"] == moving["phase"]
    assert "x" in moving["visual"]
    assert "y" in moving["visual"]
    assert moving["visual_units"][UnitType.VANGUARD.value]

    assert observed["segment_src"] == moving["segment_src"]
    assert observed["segment_dst"] == moving["segment_dst"]
    assert observed["lane_id"] == moving["lane_id"]
    assert observed["approach_id"] == moving["approach_id"]
    assert "visual" not in observed
    assert "visual_units" not in observed


def test_battle_entry_has_no_visual_teleport():
    # A unit marching into a node battle must walk into its combat formation at
    # roughly its own speed: the joining frame should not snap the unit across
    # the map. Vanguards move ~45 map units/tick, so every per-tick visual step
    # (including the moving->battle hand-off) must stay near that, never the
    # 100px+ jump the raw formation snap used to produce.
    _, engine = make_node_battle_engine()
    for _ in range(20):
        engine.tick()

    tracks = {}
    for snapshot in engine.replay_ticks:
        for unit in snapshot.get("visual_units", []):
            if unit.get("owner") != "A1":
                continue
            tracks.setdefault(unit["id"], {})[snapshot["tick"]] = (unit["x"], unit["y"])

    saw_entry_step = False
    max_step = 0.0
    for frames in tracks.values():
        ordered = sorted(frames)
        for earlier, later in zip(ordered, ordered[1:]):
            if later != earlier + 1:
                continue
            (x0, y0), (x1, y1) = frames[earlier], frames[later]
            step = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            max_step = max(max_step, step)
            saw_entry_step = True

    assert saw_entry_step, "no A1 visual unit movement was recorded"
    # Vanguard marching step is 45 map units; allow slack for the seed anchor
    # frame but stay far below a formation-snap teleport.
    assert max_step < 60.0, f"battle entry teleported {max_step:.1f} map units in one tick"


def test_replay_visual_unit_ids_survive_move_to_battle():
    _, engine = make_node_battle_engine()

    engine.tick()
    moving_ids = set(engine.replay_ticks[-1]["moving_groups"][0]["visual_units"][UnitType.VANGUARD.value])

    for _ in range(DIRECT_MOVE_TICKS + 2):
        engine.tick()
        battle_rows = [
            unit
            for unit in engine.replay_ticks[-1]["visual_units"]
            if unit.get("layer") == "battle" and unit.get("owner") == "A1"
        ]
        if battle_rows:
            battle_ids = {unit["id"] for unit in battle_rows}
            assert moving_ids & battle_ids
            assert all("x" in unit and "y" in unit for unit in battle_rows)
            return

    raise AssertionError("A1 moving visual ids never reached a battle visual layer")


def test_replay_moving_group_visual_preserves_transit_entry_and_exit_approaches():
    state = make_initial_state(seed=42)
    state.nodes["A3"].units = {UnitType.VANGUARD.value: 7}
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

    for _ in range(12):
        engine.tick()
        if state.moving_groups and state.moving_groups[0].phase == "node_transit":
            break

    moving = engine.replay_ticks[-1]["moving_groups"][0]

    assert moving["phase"] == "node_transit"
    assert moving["segment_node"] == "CP1"
    assert moving["segment_src"] == "CP1"
    assert moving["segment_dst"] == "CP3"
    assert moving["lane_id"] == "CP1|CP3"
    assert moving["approach_id"] == "CP1->CP3"
    assert moving["entry_approach_id"] == "A3->CP1"
    assert moving["visual"]["transit_node"] == "CP1"
    assert moving["visual"]["entry_approach_id"] == "A3->CP1"


def test_moving_group_snapshot_marks_threat_zone_approaches():
    state = make_initial_state(seed=43)
    state.nodes["CP1"].owner = "A1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.VANGUARD.value: 2}
    engine = GameEngine(
        state,
        make_bots({
            "A1": [Action(ActionType.MOVE, source="CP1", destination="Baron")],
        }),
    )

    engine.tick()
    moving = engine.replay_ticks[-1]["moving_groups"][0]
    observed = engine.observe("A1")["moving_groups"][0]

    assert moving["approach_id"] == "CP1->Baron"
    assert moving["threat_zone_ids"] == ["Lord"]
    assert observed["threat_zone_ids"] == ["Lord"]
