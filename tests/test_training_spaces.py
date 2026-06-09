import copy

import numpy as np

from src.core.constants import (
    ActionType,
    BARON_HP,
    BUILDING_COST,
    CORE_HP,
    SPECIAL_DEST_ALLY_LORD,
    SPECIAL_DEST_ENEMY_LORD,
    BuildingType,
    UnitType,
)
from src.core.combat import build_player_units
from src.core.engine import GameEngine
from src.core.state import Battle, MovingGroup, make_initial_state
from src.training.spaces import (
    ACTION_DIM,
    ACTION_SPECS,
    BATTLE_FEATURES,
    BUILD_ACTION_START,
    BUILDING_FEATURES,
    COMBAT_RELATIONS,
    CORE_TYPE_FEATURES,
    GLOBAL_FEATURES,
    MIRROR_NODES,
    MOVE_ACTION_START,
    MOVEMENT_BUCKETS,
    MOVEMENT_FEATURES,
    NODE_FEATURES,
    NODE_SCALAR_FEATURES,
    NODE_TYPE_FEATURES,
    OWNER_RELATION_FEATURES,
    OBSERVATION_DIM,
    OBSERVATION_VERSION,
    STATIONED_UNIT_FEATURES,
    UNIT_ORDER,
    action_is_legal,
    decode_action,
    encode_observation,
    estimate_route_ticks,
    find_route_obs,
    legal_action_mask,
    mirror_action_index,
    mirror_observation,
    mirrored_node_id,
    mirrored_player_id,
    normalize_action,
    policy_observation,
    route_metrics_for_node,
    special_lord_destination_node,
)
from src.core.map import ALL_NODES


def observe_state(state, player_id="A1"):
    engine = GameEngine(state, {}, record_trace=False, record_replay=False)
    return engine.observe(player_id)


def action_index(kind, source=None, destination=None, intent="CAPTURE", unit_type=None, building_type=None):
    for index, spec in enumerate(ACTION_SPECS):
        if (
            spec.kind == kind
            and spec.source == source
            and spec.destination == destination
            and spec.intent == intent
            and spec.unit_type == unit_type
            and spec.building_type == building_type
        ):
            return index
    raise AssertionError(f"Action spec not found: {kind} {destination} {intent} {unit_type}")


def assert_actions_are_mirrors(a_action, b_action):
    assert b_action.action_type == a_action.action_type
    assert b_action.unit_type == a_action.unit_type
    assert b_action.intent == a_action.intent
    assert b_action.source == mirrored_node_id(a_action.source)
    assert b_action.destination == mirrored_node_id(a_action.destination)
    if a_action.sub_actions is None:
        assert b_action.sub_actions is None
        return
    assert b_action.sub_actions is not None
    assert len(b_action.sub_actions) == len(a_action.sub_actions)
    for a_sub_action, b_sub_action in zip(a_action.sub_actions, b_action.sub_actions):
        assert_actions_are_mirrors(a_sub_action, b_sub_action)


def test_training_observation_vector_is_fixed_and_finite():
    obs = observe_state(make_initial_state(seed=1))
    vector = encode_observation(obs, "A1")

    assert OBSERVATION_VERSION == "train_obs_v3.1"
    assert ACTION_DIM == 447
    assert vector.shape == (OBSERVATION_DIM,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()


def test_training_observation_cache_preserves_masks_vectors_and_mirror():
    state = make_initial_state(seed=31)
    state.tick = 17
    state.nodes["CP1"].owner = "A1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.RANGER.value: 4, UnitType.SKIRMISHER.value: 2}
    state.nodes["CP5"].owner = "B1"
    state.nodes["CP5"].neutral_hp = 0
    state.nodes["CP5"].units = {UnitType.RANGER.value: 4, UnitType.SKIRMISHER.value: 2}

    for player_id in ("A1", "B1"):
        cached_obs = observe_state(copy.deepcopy(state), player_id)
        fresh_obs = copy.deepcopy(cached_obs)

        # Warm every route-related cache used by both the mask and vector encoder.
        legal_action_mask(cached_obs, player_id)
        encode_observation(cached_obs, player_id)
        for node_id in ALL_NODES:
            route_metrics_for_node(cached_obs, player_id, node_id)

        np.testing.assert_array_equal(
            legal_action_mask(cached_obs, player_id),
            legal_action_mask(fresh_obs, player_id),
        )
        np.testing.assert_allclose(
            encode_observation(cached_obs, player_id),
            encode_observation(fresh_obs, player_id),
            atol=1.0e-6,
        )

    b_obs = observe_state(state, "B1")
    legal_action_mask(b_obs, "B1")
    mirrored = mirror_observation(b_obs)
    assert "__training_spaces_cache" not in mirrored


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


def test_mirror_action_index_is_lossless_for_full_action_space():
    for index, spec in enumerate(ACTION_SPECS):
        mirrored_index = mirror_action_index(index)
        roundtrip_index = mirror_action_index(mirrored_index)

        np.testing.assert_array_equal(roundtrip_index, normalize_action(index))


def test_b_side_legal_actions_decode_as_exact_mirror_of_a_side_actions():
    state = make_initial_state(seed=11)
    obs_a = observe_state(state, "A1")
    obs_b = observe_state(state, "B1")
    mask_a = legal_action_mask(obs_a, "A1")
    mask_b = legal_action_mask(obs_b, "B1")

    np.testing.assert_array_equal(mask_a, mask_b)
    for action_id in np.flatnonzero(mask_b):
        action_a = decode_action(int(action_id), obs_a, "A1")
        action_b = decode_action(int(action_id), obs_b, "B1")
        assert_actions_are_mirrors(action_a, action_b)


def test_b_side_policy_observation_matches_manual_mirrored_state_with_dynamic_fields():
    state = make_initial_state(seed=13)
    state.tick = 42
    state.nodes["CP5"].owner = "B1"
    state.nodes["CP5"].neutral_hp = 0
    state.nodes["CP5"].units = {UnitType.RANGER.value: 3, UnitType.SKIRMISHER.value: 1}
    state.nodes["CP5"].unit_ready_tick = {"B1": 45}
    state.nodes["CP5"].unit_entry_side = {"B1": "defender"}
    state.nodes["CP1"].owner = "A1"
    state.nodes["CP1"].neutral_hp = 0
    state.nodes["CP1"].units = {UnitType.RANGER.value: 3, UnitType.SKIRMISHER.value: 1}
    state.nodes["CP1"].unit_ready_tick = {"A1": 45}
    state.nodes["CP1"].unit_entry_side = {"A1": "attacker"}
    state.moving_groups.append(MovingGroup(
        player_id="B1",
        team="TEAM_B",
        units={UnitType.VANGUARD.value: 2, UnitType.RANGER.value: 1},
        src="CP5",
        dst="CP3",
        arrival_tick=49,
        started_tick=40,
        travel_ticks=5.0,
        progress=0.4,
        intent="CAPTURE",
        path=["CP5", "CP3"],
        phase="edge",
        speed=1.5,
        final_dst="CP3",
    ))
    state.active_battles["edge:CP3|CP5"] = Battle(
        battle_id="edge:CP3|CP5",
        location_kind="edge",
        location_id="CP3|CP5",
        units=[
            *build_player_units({UnitType.VANGUARD.value: 2}, "B1", "attacker"),
            *build_player_units({UnitType.RANGER.value: 3}, "A2", "defender"),
        ],
        started_at_tick=40,
        last_tick=42,
        participant_entry_sides={"B1": "attacker", "A2": "defender"},
        edge_endpoints=("CP3", "CP5"),
        edge_position=0.35,
        objective="pvp",
    )
    state.baron.allied_lord_owner = "B2"
    state.baron.allied_lord_hp = 111.0
    state.baron.allied_lord_target = "A3"
    state.baron.allied_lord_path = ["Baron", "CP2", "A3"]
    state.baron.allied_lord_ready_tick = 44
    state.baron.allied_lord_arrival_tick = 58
    state.baron.allied_lord_alive = True
    state.core_damage_by_team["TEAM_B"] = 7.0
    state.death_time_final_core_deaths["TEAM_A"] = 1

    obs_b = observe_state(state, "B1")
    mirrored_obs = mirror_observation(obs_b)
    policy_obs, policy_player = policy_observation(obs_b, "B1")

    assert policy_player == "A1"
    assert policy_obs == mirrored_obs
    assert mirrored_obs["nodes"]["CP1"]["owner"] == "A1"
    assert mirrored_obs["moving_groups"][0]["player"] == "A1"
    assert mirrored_obs["moving_groups"][0]["src"] == "CP1"
    assert mirrored_obs["moving_groups"][0]["dst"] == "CP3"
    assert mirrored_obs["active_battles"][0]["location_id"] == "CP1|CP3"
    assert mirrored_obs["baron"]["allied_lord_owner"] == "A2"
    assert mirrored_obs["baron"]["allied_lord_target"] == "B3"
    assert mirrored_obs["baron"]["allied_lord_path"] == ["Baron", "CP6", "B3"]
    np.testing.assert_allclose(
        encode_observation(obs_b, "B1"),
        encode_observation(mirrored_obs, "A1"),
        atol=1.0e-6,
    )


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

    bundled_index = np.array(
        [
            action_index("PRODUCE", unit_type=UnitType.SKIRMISHER),
            action_index("MOVE", source="A1", destination="CP3", intent="CAPTURE") - MOVE_ACTION_START,
            0,
        ],
        dtype=np.int64,
    )
    assert action_is_legal(bundled_index, mask)

    bundled = decode_action(bundled_index, obs, "A1")
    assert bundled.action_type == ActionType.MULTI
    assert [step.action_type for step in bundled.sub_actions] == [ActionType.PRODUCE, ActionType.MOVE]
    assert bundled.sub_actions[0].unit_type == UnitType.SKIRMISHER
    assert bundled.sub_actions[1].source == "A1"
    assert bundled.sub_actions[1].destination == "CP3"


def test_multi_action_moves_existing_units_before_production():
    class BundleBot:
        def act(self, obs):
            return decode_action(
                np.array(
                    [
                        action_index("PRODUCE", unit_type=UnitType.SKIRMISHER),
                        action_index("MOVE", source="A1", destination="CP1", intent="CAPTURE") - MOVE_ACTION_START,
                        0,
                    ],
                    dtype=np.int64,
                ),
                obs,
                "A1",
            )

    state = make_initial_state(seed=12)
    state.nodes["A1"].units = {UnitType.VANGUARD.value: 7}
    engine = GameEngine(state, {"A1": BundleBot()}, record_trace=False, record_replay=False)

    engine.tick()

    assert state.nodes["A1"].units == {UnitType.SKIRMISHER.value: 1}
    assert len(state.moving_groups) == 1
    assert state.moving_groups[0].units == {UnitType.VANGUARD.value: 7}


def test_factorized_mask_reserves_resources_for_legal_builds_before_production():
    state = make_initial_state(seed=24)
    state.players["A1"].resources = float(BUILDING_COST[BuildingType.ECONOMY_COLLECTOR])
    obs = observe_state(state, "A1")
    mask = legal_action_mask(obs, "A1")

    collector_index = action_index("BUILD", destination="A1", building_type=BuildingType.ECONOMY_COLLECTOR)
    skirmisher_index = action_index("PRODUCE", unit_type=UnitType.SKIRMISHER)

    assert mask[collector_index]
    assert not mask[skirmisher_index]

    state.players["A1"].resources = float(BUILDING_COST[BuildingType.ECONOMY_COLLECTOR] + 2)
    obs = observe_state(state, "A1")
    mask = legal_action_mask(obs, "A1")

    assert mask[collector_index]
    assert mask[skirmisher_index]


def test_b_side_action_space_uses_policy_perspective_and_decodes_to_absolute_nodes():
    obs = observe_state(make_initial_state(seed=2), "B1")
    canonical = action_index("MOVE", source="A1", destination="CP1", intent="CAPTURE")
    absolute = action_index("MOVE", source="B1", destination="CP5", intent="CAPTURE")
    mask = legal_action_mask(obs, "B1")

    assert mask[canonical]
    assert not mask[absolute]
    np.testing.assert_array_equal(mirror_action_index(canonical), normalize_action(absolute))

    action = decode_action(canonical, obs, "B1")

    assert action.action_type == ActionType.MOVE
    assert action.source == "B1"
    assert action.destination == "CP5"
    assert action.intent == "CAPTURE"


def test_dead_player_and_dead_core_are_not_legal_action_targets():
    state = make_initial_state(seed=23)
    state.players["A1"].alive = False
    state.players["A1"].core_hp = 0.0
    state.nodes["A1"].owner = "NEUTRAL"
    state.nodes["A1"].units = {}

    dead_obs = observe_state(state, "A1")
    dead_mask = legal_action_mask(dead_obs, "A1")
    assert dead_mask[0]
    assert dead_mask[MOVE_ACTION_START]
    assert dead_mask[BUILD_ACTION_START]
    assert int(dead_mask.sum()) == 3

    a2_obs = observe_state(state, "A2")
    move_to_dead_core = action_index("MOVE", source="A2", destination="A1", intent="CAPTURE")
    assert find_route_obs("A2", "A2", "A1", a2_obs) is None
    assert not legal_action_mask(a2_obs, "A2")[move_to_dead_core]


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

    scalar_start = (
        NODE_TYPE_FEATURES
        + CORE_TYPE_FEATURES
        + OWNER_RELATION_FEATURES
        + STATIONED_UNIT_FEATURES
    )
    b1_core_hp_offset = GLOBAL_FEATURES + ALL_NODES.index("B1") * NODE_FEATURES + scalar_start + 2

    assert vector[3] == 1.0
    assert vector[b1_core_hp_offset] == np.float32(17.0 / float(CORE_HP))


def test_training_observation_exposes_core_type_for_all_cores():
    obs = observe_state(make_initial_state(seed=5), "A1")
    vector = encode_observation(obs, "A1")
    core_type_start = NODE_TYPE_FEATURES

    assert vector[node_offset("A2") + core_type_start] == np.float32(1.0)
    assert vector[node_offset("B1") + core_type_start] == np.float32(1.0)
    assert vector[node_offset("CP1") + core_type_start] == np.float32(0.0)


def test_training_observation_exposes_allied_lord_route_and_eta():
    state = make_initial_state(seed=5)
    state.tick = 12
    state.baron.allied_lord_owner = "B2"
    state.baron.allied_lord_hp = 91.0
    state.baron.allied_lord_target = "A3"
    state.baron.allied_lord_path = ["Baron", "CP2", "A3"]
    state.baron.allied_lord_ready_tick = 14
    state.baron.allied_lord_arrival_tick = 22
    state.baron.allied_lord_alive = True
    obs = observe_state(state, "A1")
    vector = encode_observation(obs, "A1")

    assert vector[14] == np.float32(1.0)
    assert vector[15] == np.float32(91.0 / float(BARON_HP))
    assert vector[16] == np.float32(2.0 / 300.0)
    assert vector[17] == np.float32(10.0 / 300.0)
    assert vector[20] == np.float32(1.0)  # enemy relation one-hot

    scalar_start = (
        NODE_TYPE_FEATURES
        + CORE_TYPE_FEATURES
        + OWNER_RELATION_FEATURES
        + STATIONED_UNIT_FEATURES
    )
    lord_path_offset = scalar_start + 9
    lord_target_offset = scalar_start + 10

    assert vector[node_offset("CP2") + lord_path_offset] == np.float32(1.0)
    assert vector[node_offset("A3") + lord_path_offset] == np.float32(1.0)
    assert vector[node_offset("A3") + lord_target_offset] == np.float32(1.0)
    assert vector[node_offset("B3") + lord_path_offset] == np.float32(0.0)

    mirrored_obs, mirrored_player = policy_observation(observe_state(state, "B1"), "B1")
    assert mirrored_player == "A1"
    assert mirrored_obs["baron"]["allied_lord_owner"] == "A2"
    assert mirrored_obs["baron"]["allied_lord_target"] == "B3"
    assert mirrored_obs["baron"]["allied_lord_path"] == ["Baron", "CP6", "B3"]


def test_training_action_space_exposes_lord_join_and_hunt_targets():
    state = make_initial_state(seed=8)
    state.baron.allied_lord_owner = "A2"
    state.baron.allied_lord_hp = float(BARON_HP)
    state.baron.allied_lord_target = "B3"
    state.baron.allied_lord_path = ["Baron", "CP5", "B3"]
    state.baron.allied_lord_ready_tick = state.tick + 2
    state.baron.allied_lord_arrival_tick = state.tick + 12
    state.baron.allied_lord_alive = True
    obs_a = observe_state(state, "A1")
    obs_b = observe_state(state, "B1")

    join_index = action_index("MOVE", source="A1", destination=SPECIAL_DEST_ALLY_LORD, intent="JOIN_LORD")
    hunt_index = action_index("MOVE", source="A1", destination=SPECIAL_DEST_ENEMY_LORD, intent="HUNT_LORD")

    mask_a = legal_action_mask(obs_a, "A1")
    assert mask_a[join_index]
    assert not mask_a[hunt_index]
    assert special_lord_destination_node(obs_a, "A1", SPECIAL_DEST_ALLY_LORD, "A1") == "Baron"

    action = decode_action(join_index, obs_a, "A1")
    assert action.action_type == ActionType.MOVE
    assert action.source == "A1"
    assert action.destination == SPECIAL_DEST_ALLY_LORD
    assert action.intent == "JOIN_LORD"

    mask_b = legal_action_mask(obs_b, "B1")
    assert not mask_b[join_index]
    assert mask_b[hunt_index]
    hunt = decode_action(hunt_index, obs_b, "B1")
    assert hunt.source == "B1"
    assert hunt.destination == SPECIAL_DEST_ENEMY_LORD
    assert hunt.intent == "HUNT_LORD"


def test_route_tick_estimate_is_discrete_per_segment():
    assert estimate_route_ticks(["A1", "CP1"], speed=3.0) == 3.0
    assert estimate_route_ticks(["A1", "CP1", "CP3"], speed=3.0) == 5.0


def node_offset(node_id: str) -> int:
    return GLOBAL_FEATURES + ALL_NODES.index(node_id) * NODE_FEATURES


def movement_offset(node_id: str, bucket: str, relation: str, unit_type: UnitType) -> int:
    movement_start = (
        NODE_TYPE_FEATURES
        + CORE_TYPE_FEATURES
        + OWNER_RELATION_FEATURES
        + STATIONED_UNIT_FEATURES
        + NODE_SCALAR_FEATURES
        + BUILDING_FEATURES
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
        NODE_TYPE_FEATURES
        + CORE_TYPE_FEATURES
        + OWNER_RELATION_FEATURES
        + STATIONED_UNIT_FEATURES
        + NODE_SCALAR_FEATURES
        + BUILDING_FEATURES
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
