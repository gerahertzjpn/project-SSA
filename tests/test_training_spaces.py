import numpy as np

from src.core.constants import ActionType, UnitType
from src.core.combat import build_player_units
from src.core.engine import GameEngine
from src.core.state import Battle, MovingGroup, make_initial_state
from src.training.spaces import (
    ACTION_DIM,
    ACTION_SPECS,
    BATTLE_FEATURES,
    COMBAT_RELATIONS,
    GLOBAL_FEATURES,
    MIRROR_NODES,
    MOVEMENT_BUCKETS,
    MOVEMENT_FEATURES,
    NODE_FEATURES,
    NODE_SCALAR_FEATURES,
    OWNER_RELATION_FEATURES,
    OBSERVATION_DIM,
    OBSERVATION_VERSION,
    STATIONED_UNIT_FEATURES,
    UNIT_ORDER,
    decode_action,
    encode_observation,
    find_route_obs,
    legal_action_mask,
    mirror_action_index,
    mirrored_node_id,
    mirrored_player_id,
    policy_observation,
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

    assert OBSERVATION_VERSION == "train_obs_v0.5"
    assert vector.shape == (OBSERVATION_DIM,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()


def test_mirror_maps_are_involutive_and_preserve_player_slots():
    for node_id, mirrored in MIRROR_NODES.items():
        assert mirrored_node_id(mirrored) == node_id

    assert mirrored_player_id("A1") == "B1"
    assert mirrored_player_id("A2") == "B2"
    assert mirrored_player_id("A3") == "B3"
    assert mirrored_player_id("B1") == "A1"
    assert mirrored_player_id("B2") == "A2"
    assert mirrored_player_id("B3") == "A3"


def test_initial_a_and_b_matching_slots_share_policy_observation():
    state = make_initial_state(seed=1)
    for a_player, b_player in (("A1", "B1"), ("A2", "B2"), ("A3", "B3")):
        obs_a = observe_state(state, a_player)
        obs_b = observe_state(state, b_player)

        np.testing.assert_allclose(
            encode_observation(obs_a, a_player),
            encode_observation(obs_b, b_player),
            atol=1.0e-6,
        )

    policy_obs, policy_player = policy_observation(observe_state(state, "B1"), "B1")
    assert policy_player == "A1"
    assert policy_obs["player_id"] == "A1"
    assert policy_obs["team"] == "TEAM_A"
    assert policy_obs["nodes"]["A1"]["owner"] == "A1"
    assert policy_obs["nodes"]["CP1"]["neutral_hp"] == state.nodes["CP5"].neutral_hp


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


def test_b_side_action_space_uses_policy_perspective_and_decodes_to_absolute_nodes():
    obs = observe_state(make_initial_state(seed=2), "B1")
    canonical = action_index("MOVE", source="A1", destination="CP1", intent="CAPTURE")
    absolute = action_index("MOVE", source="B1", destination="CP5", intent="CAPTURE")
    mask = legal_action_mask(obs, "B1")

    assert mask[canonical]
    assert not mask[absolute]
    assert mirror_action_index(canonical) == absolute

    action = decode_action(canonical, obs, "B1")

    assert action.action_type == ActionType.MOVE
    assert action.source == "B1"
    assert action.destination == "CP5"
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


def node_offset(node_id: str) -> int:
    return GLOBAL_FEATURES + ALL_NODES.index(node_id) * NODE_FEATURES


def movement_offset(node_id: str, bucket: str, relation: str, unit_type: UnitType) -> int:
    movement_start = (
        3
        + OWNER_RELATION_FEATURES
        + STATIONED_UNIT_FEATURES
        + NODE_SCALAR_FEATURES
    )
    bucket_index = MOVEMENT_BUCKETS.index(bucket)
    relation_index = COMBAT_RELATIONS.index(relation)
    unit_index = UNIT_ORDER.index(unit_type)
    return (
        node_offset(node_id)
        + movement_start
        + bucket_index * len(COMBAT_RELATIONS) * len(UNIT_ORDER)
        + relation_index * len(UNIT_ORDER)
        + unit_index
    )


def battle_offset(
    node_id: str,
    *,
    edge: bool,
    relation: str | None = None,
    unit_type: UnitType | None = None,
    hp_slot: int | None = None,
) -> int:
    battle_start = (
        3
        + OWNER_RELATION_FEATURES
        + STATIONED_UNIT_FEATURES
        + NODE_SCALAR_FEATURES
        + MOVEMENT_FEATURES
        + (BATTLE_FEATURES if edge else 0)
    )
    if hp_slot is not None:
        return node_offset(node_id) + battle_start + hp_slot
    assert relation is not None and unit_type is not None
    relation_index = COMBAT_RELATIONS.index(relation)
    unit_index = UNIT_ORDER.index(unit_type)
    return node_offset(node_id) + battle_start + 4 + relation_index * len(UNIT_ORDER) + unit_index


def test_training_observation_exposes_moving_unit_types_by_relation_and_endpoint():
    state = make_initial_state(seed=6)
    state.moving_groups.append(MovingGroup(
        player_id="B1",
        team="TEAM_B",
        units={UnitType.RANGER.value: 7, UnitType.SKIRMISHER.value: 2},
        src="CP1",
        dst="A3",
        arrival_tick=state.tick + 4,
        phase="edge",
        progress=0.25,
    ))
    obs = observe_state(state, "A1")
    vector = encode_observation(obs, "A1")

    assert vector[movement_offset("A3", "incoming", "enemy", UnitType.RANGER)] == np.float32(7 / 50)
    assert vector[movement_offset("A3", "incoming", "enemy", UnitType.SKIRMISHER)] == np.float32(2 / 50)
    assert vector[movement_offset("A3", "incoming", "enemy", UnitType.VANGUARD)] == np.float32(0.0)
    assert vector[movement_offset("CP1", "outgoing", "enemy", UnitType.RANGER)] == np.float32(7 / 50)


def test_training_observation_exposes_edge_battle_unit_types_on_both_endpoints():
    state = make_initial_state(seed=7)
    state.active_battles["edge:CP1|CP3"] = Battle(
        battle_id="edge:CP1|CP3",
        location_kind="edge",
        location_id="CP1|CP3",
        units=[
            *build_player_units({UnitType.VANGUARD.value: 3}, "A2", "attacker"),
            *build_player_units({UnitType.RANGER.value: 4}, "B1", "defender"),
        ],
        started_at_tick=state.tick,
        last_tick=state.tick,
        objective="pvp",
    )
    obs = observe_state(state, "A1")
    vector = encode_observation(obs, "A1")

    for endpoint in ("CP1", "CP3"):
        assert vector[battle_offset(endpoint, edge=True, relation="ally", unit_type=UnitType.VANGUARD)] == np.float32(3 / 50)
        assert vector[battle_offset(endpoint, edge=True, relation="enemy", unit_type=UnitType.RANGER)] == np.float32(4 / 50)
        assert vector[battle_offset(endpoint, edge=True, hp_slot=0)] > 0
        assert vector[battle_offset(endpoint, edge=True, hp_slot=1)] > 0
