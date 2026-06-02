import numpy as np

from src.core.constants import ActionType, UnitType
from src.core.engine import GameEngine
from src.core.state import make_initial_state
from src.training.spaces import (
    ACTION_DIM,
    ACTION_SPECS,
    GLOBAL_FEATURES,
    NODE_FEATURES,
    OBSERVATION_DIM,
    OBSERVATION_VERSION,
    decode_action,
    encode_observation,
    find_route_obs,
    legal_action_mask,
    route_metrics_for_node,
)
from src.core.map import ALL_NODES


def observe_state(state, player_id="A1"):
    engine = GameEngine(state, {}, record_trace=False, record_replay=False)
    return engine.observe(player_id)


def action_index(kind, source=None, destination=None, intent="CAPTURE", unit_type=None):
    for index, spec in enumerate(ACTION_SPECS):
        if (
            spec.kind == kind
            and spec.source == source
            and spec.destination == destination
            and spec.intent == intent
            and spec.unit_type == unit_type
        ):
            return index
    raise AssertionError(f"Action spec not found: {kind} {destination} {intent} {unit_type}")


def test_training_observation_vector_is_fixed_and_finite():
    obs = observe_state(make_initial_state(seed=1))
    vector = encode_observation(obs, "A1")

    assert OBSERVATION_VERSION == "train_obs_v0.3"
    assert vector.shape == (OBSERVATION_DIM,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()


def test_training_action_mask_and_decode_initial_actions():
    obs = observe_state(make_initial_state(seed=2))
    mask = legal_action_mask(obs, "A1")

    assert mask.shape == (ACTION_DIM,)
    assert mask[0]
    assert mask[action_index("PRODUCE", unit_type=UnitType.VANGUARD)]
    assert mask[action_index("PRODUCE", unit_type=UnitType.RANGER)]
    assert mask[action_index("MOVE", source="A1", destination="CP3", intent="CAPTURE")]

    action = decode_action(action_index("MOVE", source="A1", destination="CP3", intent="CAPTURE"), obs, "A1")
    assert action.action_type == ActionType.MOVE
    assert action.source == "A1"
    assert action.destination == "CP3"
    assert action.intent == "CAPTURE"


def test_training_routes_through_guarded_neutral_but_not_enemy_frontline():
    state = make_initial_state(seed=3)
    obs = observe_state(state)

    assert state.nodes["CP1"].neutral_hp > 0
    assert find_route_obs("A1", "A1", "CP3", obs) == ["A1", "CP1", "CP3"]

    state.nodes["CP1"].owner = "B1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP2"].owner = "B1"
    state.nodes["CP2"].neutral_hp = 0
    blocked_obs = observe_state(state)
    mask = legal_action_mask(blocked_obs, "A1")

    assert mask[action_index("MOVE", source="A1", destination="CP1", intent="CAPTURE")]
    assert mask[action_index("MOVE", source="A1", destination="CP2", intent="CAPTURE")]
    assert not mask[action_index("MOVE", source="A1", destination="CP3", intent="CAPTURE")]


def test_training_action_space_can_choose_move_source():
    state = make_initial_state(seed=2)
    state.nodes["CP2"].owner = "A1"
    state.nodes["CP2"].neutral_hp = 0
    state.nodes["CP2"].units = {UnitType.VANGUARD.value: 1}
    obs = observe_state(state)
    mask = legal_action_mask(obs, "A1")

    from_a1 = action_index("MOVE", source="A1", destination="CP4", intent="CAPTURE")
    from_cp2 = action_index("MOVE", source="CP2", destination="CP4", intent="CAPTURE")

    assert mask[from_a1]
    assert mask[from_cp2]

    action = decode_action(from_cp2, obs, "A1")

    assert action.action_type == ActionType.MOVE
    assert action.source == "CP2"
    assert action.destination == "CP4"


def test_training_observation_exposes_route_distance_and_eta():
    obs = observe_state(make_initial_state(seed=4))

    own_node = route_metrics_for_node(obs, "A1", "A1")
    cp1 = route_metrics_for_node(obs, "A1", "CP1")
    b1 = route_metrics_for_node(obs, "A1", "B1")

    assert own_node == (1.0, 0.0, 0.0)
    assert cp1[0] == 1.0
    assert b1[0] == 1.0
    assert cp1[1] < b1[1]
    assert cp1[2] < b1[2]


def test_training_observation_exposes_per_core_hp_and_death_time_clock():
    state = make_initial_state(seed=5)
    state.tick = 270
    state.players["B1"].core_hp = 17.0
    obs = observe_state(state)
    vector = encode_observation(obs, "A1")

    b1_core_hp_offset = GLOBAL_FEATURES + ALL_NODES.index("B1") * NODE_FEATURES + 21

    assert vector[3] == 1.0
    assert vector[b1_core_hp_offset] == np.float32(17.0 / 50.0)
