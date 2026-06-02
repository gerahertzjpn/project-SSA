import numpy as np

from src.core.constants import ActionType, UnitType
from src.core.engine import GameEngine
from src.core.state import make_initial_state
from src.training.spaces import (
    ACTION_DIM,
    ACTION_SPECS,
    OBSERVATION_DIM,
    OBSERVATION_VERSION,
    decode_action,
    encode_observation,
    find_route_obs,
    legal_action_mask,
)


def observe_state(state, player_id="A1"):
    engine = GameEngine(state, {}, record_trace=False, record_replay=False)
    return engine.observe(player_id)


def action_index(kind, destination=None, intent="CAPTURE", unit_type=None):
    for index, spec in enumerate(ACTION_SPECS):
        if (
            spec.kind == kind
            and spec.destination == destination
            and spec.intent == intent
            and spec.unit_type == unit_type
        ):
            return index
    raise AssertionError(f"Action spec not found: {kind} {destination} {intent} {unit_type}")


def test_training_observation_vector_is_fixed_and_finite():
    obs = observe_state(make_initial_state(seed=1))
    vector = encode_observation(obs, "A1")

    assert OBSERVATION_VERSION == "train_obs_v0.2"
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
    assert mask[action_index("MOVE", destination="CP3", intent="CAPTURE")]

    action = decode_action(action_index("MOVE", destination="CP3", intent="CAPTURE"), obs, "A1")
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

    assert mask[action_index("MOVE", destination="CP1", intent="CAPTURE")]
    assert mask[action_index("MOVE", destination="CP2", intent="CAPTURE")]
    assert not mask[action_index("MOVE", destination="CP3", intent="CAPTURE")]
