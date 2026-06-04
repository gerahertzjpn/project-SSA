import math

import pytest

from src.core.constants import NODE_POSITIONS, NodeType
from src.core.map import ADJACENCY, ALL_NODES, CORE_NODES, NODE_TYPES
from src.core.map2d import (
    APPROACHES,
    LANES,
    MAP2D_VERSION,
    NODE_LAYOUT,
    THREAT_ZONES,
    approach_edge_length,
    canonical_lane_id,
    directed_approach_id,
    get_approach,
    get_lane,
    incoming_approaches,
    is_core_approach,
    is_lord_approach,
    map2d_replay_meta,
    outgoing_approaches,
    node_zone_progress_interval,
    point_on_lane,
    point_on_approach_edge,
    threat_zone_entry_progress,
    threat_zone_ids_for_approach,
)


def test_map2d_has_static_version_and_all_nodes():
    assert MAP2D_VERSION == "simple_2d_map_v0.1"
    assert set(NODE_LAYOUT) == set(ADJACENCY)

    for node_id, layout in NODE_LAYOUT.items():
        assert layout.node_id == node_id
        assert layout.node_type == NODE_TYPES[node_id]
        assert layout.center == tuple(float(value) for value in NODE_POSITIONS[node_id])
        assert layout.radius > 0.0
        assert layout.combat_radius >= layout.radius

    assert NODE_LAYOUT["Baron"].node_type == NodeType.BARON


def test_map2d_lanes_match_undirected_graph_edges():
    expected_lanes = {
        canonical_lane_id(src, dst)
        for src, neighbors in ADJACENCY.items()
        for dst in neighbors
    }

    assert set(LANES) == expected_lanes

    for lane_id, lane in LANES.items():
        assert lane_id == canonical_lane_id(lane.a, lane.b)
        assert lane.b in ADJACENCY[lane.a]
        assert lane.length > 0.0
        assert len(lane.points) == 2
        assert lane.points[0] == NODE_LAYOUT[lane.a].center
        assert lane.points[1] == NODE_LAYOUT[lane.b].center


def test_map2d_has_directed_approach_for_every_adjacency():
    expected_approaches = {
        directed_approach_id(src, dst)
        for src, neighbors in ADJACENCY.items()
        for dst in neighbors
    }

    assert set(APPROACHES) == expected_approaches

    for approach_id, approach in APPROACHES.items():
        assert approach_id == directed_approach_id(approach.src, approach.dst)
        assert approach.dst in ADJACENCY[approach.src]
        assert approach.lane_id == canonical_lane_id(approach.src, approach.dst)
        assert math.dist(approach.entry_point, NODE_LAYOUT[approach.dst].center) == pytest.approx(
            NODE_LAYOUT[approach.dst].radius
        )
        assert math.dist(approach.exit_point, NODE_LAYOUT[approach.src].center) == pytest.approx(
            NODE_LAYOUT[approach.src].radius
        )


def test_map2d_convenience_lookup_and_lane_points():
    lane = get_lane("A1", "CP1")
    assert lane == get_lane("CP1", "A1")
    assert lane.tunnel is True

    non_tunnel = get_lane("CP1", "CP3")
    assert non_tunnel.tunnel is False

    assert point_on_lane("A1", "CP1", 0.0) == NODE_LAYOUT["A1"].center
    assert point_on_lane("A1", "CP1", 1.0) == NODE_LAYOUT["CP1"].center
    assert point_on_approach_edge("A1", "CP1", 0.0) == get_approach("A1", "CP1").exit_point
    assert point_on_approach_edge("A1", "CP1", 1.0) == get_approach("A1", "CP1").entry_point
    assert approach_edge_length("A1", "CP1") < get_lane("A1", "CP1").length

    midpoint = point_on_lane("A1", "CP1", 0.5)
    assert midpoint == pytest.approx((
        (NODE_LAYOUT["A1"].center[0] + NODE_LAYOUT["CP1"].center[0]) / 2.0,
        (NODE_LAYOUT["A1"].center[1] + NODE_LAYOUT["CP1"].center[1]) / 2.0,
    ))

    assert get_approach("CP1", "Baron").src == "CP1"
    assert get_approach("CP1", "Baron").dst == "Baron"


def test_map2d_incoming_and_outgoing_approaches_preserve_direction():
    incoming_baron = incoming_approaches("Baron")
    outgoing_baron = outgoing_approaches("Baron")

    assert {approach.src for approach in incoming_baron} == {"CP1", "CP2", "CP5", "CP6"}
    assert {approach.dst for approach in incoming_baron} == {"Baron"}
    assert {approach.src for approach in outgoing_baron} == {"Baron"}
    assert {approach.dst for approach in outgoing_baron} == {"CP1", "CP2", "CP5", "CP6"}


def test_map2d_lord_and_core_threat_zone_metadata():
    lord_zone = THREAT_ZONES["Lord"]
    assert lord_zone.kind == "lord"
    assert lord_zone.source_node == "Baron"
    assert set(lord_zone.approach_ids) == {
        "CP1->Baron",
        "CP2->Baron",
        "CP5->Baron",
        "CP6->Baron",
    }
    assert all(is_lord_approach(APPROACHES[approach_id].src, APPROACHES[approach_id].dst) for approach_id in lord_zone.approach_ids)
    assert not is_lord_approach("Baron", "CP1")

    for core_id in CORE_NODES:
        zone = THREAT_ZONES[f"core:{core_id}"]
        assert zone.kind == "core"
        assert zone.source_node == core_id
        assert set(zone.approach_ids) == {
            directed_approach_id(src, core_id)
            for src in ADJACENCY[core_id]
        }
        assert all(
            is_core_approach(APPROACHES[approach_id].src, APPROACHES[approach_id].dst)
            for approach_id in zone.approach_ids
        )

    assert threat_zone_ids_for_approach("CP1->Baron") == ("Lord",)
    assert threat_zone_ids_for_approach("CP1->A1") == ("core:A1",)
    assert threat_zone_ids_for_approach("A1->CP1") == ()
    assert 0.0 < threat_zone_entry_progress("CP1->Baron", "Lord") < 1.0
    assert 0.0 < threat_zone_entry_progress("CP1->A1", "core:A1") < 1.0


def test_map2d_rejects_unknown_or_non_adjacent_lookups():
    with pytest.raises(KeyError):
        get_lane("NOPE", "CP1")

    with pytest.raises(ValueError):
        get_lane("A1", "B1")

    with pytest.raises(ValueError):
        directed_approach_id("CP1", "CP6")


def test_map2d_core_tunnels_are_exactly_core_edges():
    for lane in LANES.values():
        has_core_endpoint = lane.a in CORE_NODES or lane.b in CORE_NODES
        assert lane.tunnel is has_core_endpoint

    core_directed_edges = {
        (src, dst)
        for src in ALL_NODES
        for dst in ADJACENCY[src]
        if src in CORE_NODES or dst in CORE_NODES
    }
    assert core_directed_edges


def test_map2d_replay_meta_is_json_ready_and_complete():
    meta = map2d_replay_meta()

    assert meta["version"] == MAP2D_VERSION
    assert meta["world_size"] == [680.0, 410.0]
    assert set(meta["nodes"]) == set(ALL_NODES)
    assert set(meta["lanes"]) == set(LANES)
    assert set(meta["approaches"]) == set(APPROACHES)
    assert set(meta["threat_zones"]) == set(THREAT_ZONES)
    assert meta["nodes"]["Baron"]["type"] == "BARON"
    assert meta["lanes"]["A1|CP1"]["tunnel"] is True
    assert meta["approaches"]["CP1->Baron"]["dst"] == "Baron"
    assert "CP1->Baron" in meta["threat_zones"]["Lord"]["approach_ids"]


def test_map2d_node_zone_progress_interval_handles_source_and_destination():
    dst_start, dst_end = node_zone_progress_interval("A1->CP1", "CP1", radius=75.0)
    src_start, src_end = node_zone_progress_interval("A1->CP1", "A1", radius=75.0)

    assert 0.0 < dst_start < 1.0
    assert dst_end == 1.0
    assert src_start == 0.0
    assert 0.0 < src_end < 1.0
