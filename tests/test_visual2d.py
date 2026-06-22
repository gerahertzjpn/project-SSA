import math

import pytest

from src.core.map2d import NODE_LAYOUT, get_approach, point_on_approach_edge
from src.core.visual2d import (
    moving_group_visual,
    moving_group_visual_smooth,
    group_path_arc_fraction,
)


def group(**overrides):
    base = {
        "src": "A1",
        "dst": "CP1",
        "phase": "edge",
        "progress": 0.5,
        "segment_node": None,
    }
    base.update(overrides)
    return base


def test_visual2d_projects_edge_movement_to_lane_point():
    visual = moving_group_visual(group(phase="edge", progress=0.5))

    expected = point_on_approach_edge("A1", "CP1", 0.5)
    assert visual["x"] == pytest.approx(expected[0], abs=1e-3)
    assert visual["y"] == pytest.approx(expected[1], abs=1e-3)
    assert visual["lane_id"] == "A1|CP1"
    assert visual["approach_id"] == "A1->CP1"
    assert visual["phase"] == "edge"


def test_visual2d_projects_node_exit_from_center_to_exit_boundary():
    visual = moving_group_visual(group(phase="node_exit", progress=1.0))
    approach = get_approach("A1", "CP1")

    assert (visual["x"], visual["y"]) == pytest.approx(approach.exit_point, abs=1e-3)


def test_visual2d_projects_node_enter_from_entry_boundary_to_center():
    visual = moving_group_visual(group(phase="node_enter", progress=0.0))
    approach = get_approach("A1", "CP1")

    assert (visual["x"], visual["y"]) == pytest.approx(approach.entry_point, abs=1e-3)

    visual = moving_group_visual(group(phase="node_enter", progress=1.0))
    assert (visual["x"], visual["y"]) == pytest.approx(NODE_LAYOUT["CP1"].center, abs=1e-3)


def test_visual2d_projects_node_transit_across_intermediate_node():
    visual = moving_group_visual(group(
        src="A3",
        dst="CP3",
        phase="node_transit",
        segment_node="CP1",
        progress=0.0,
    ))
    entry = get_approach("A3", "CP1").entry_point
    assert (visual["x"], visual["y"]) == pytest.approx(entry, abs=1e-3)
    assert visual["lane_id"] == "CP1|CP3"
    assert visual["approach_id"] == "CP1->CP3"
    assert visual["entry_approach_id"] == "A3->CP1"
    assert visual["transit_node"] == "CP1"

    visual = moving_group_visual(group(
        src="A3",
        dst="CP3",
        phase="node_transit",
        segment_node="CP1",
        progress=0.5,
    ))
    assert (visual["x"], visual["y"]) == pytest.approx(NODE_LAYOUT["CP1"].center, abs=1e-3)

    visual = moving_group_visual(group(
        src="A3",
        dst="CP3",
        phase="node_transit",
        segment_node="CP1",
        progress=1.0,
    ))
    exit_ = get_approach("CP1", "CP3").exit_point
    assert (visual["x"], visual["y"]) == pytest.approx(exit_, abs=1e-3)


def test_visual2d_clamps_progress_for_stable_replay_positions():
    start = moving_group_visual(group(progress=-10.0))
    end = moving_group_visual(group(progress=10.0))

    assert (start["x"], start["y"]) == pytest.approx(get_approach("A1", "CP1").exit_point, abs=1e-3)
    assert (end["x"], end["y"]) == pytest.approx(get_approach("A1", "CP1").entry_point, abs=1e-3)


def journey_group(**overrides):
    base = {
        "src": "A1",
        "dst": "CP1",
        "phase": "node_exit",
        "progress": 0.0,
        "segment_node": None,
        "path": ["A1", "CP1", "CP3", "CP5", "B1"],
        "started_tick": 4,
        "arrival_tick": 20,
    }
    base.update(overrides)
    return base


def test_smooth_visual_starts_at_origin_and_ends_at_destination():
    start = moving_group_visual_smooth(journey_group(), current_tick=4)
    end = moving_group_visual_smooth(journey_group(), current_tick=20)

    assert (start["x"], start["y"]) == pytest.approx(NODE_LAYOUT["A1"].center, abs=1e-3)
    assert (end["x"], end["y"]) == pytest.approx(NODE_LAYOUT["B1"].center, abs=1e-3)


def test_smooth_visual_advances_at_constant_speed():
    # Per-tick world displacement must stay uniform across the whole journey:
    # no slow-down at node-exit/transit waypoints and no sprint along edges.
    positions = []
    for tick in range(4, 21):
        visual = moving_group_visual_smooth(journey_group(), current_tick=tick)
        positions.append((visual["x"], visual["y"]))

    steps = [math.dist(positions[i], positions[i + 1]) for i in range(len(positions) - 1)]
    assert min(steps) > 0.0
    # Uniform: spread across ticks is tiny (only polyline-vertex rounding), far
    # below the multi-x swings the per-segment projection used to produce.
    assert max(steps) - min(steps) < 0.05 * max(steps)


def test_smooth_visual_preserves_segment_metadata():
    base = moving_group_visual(journey_group(phase="node_transit", segment_node="CP1", dst="CP3"))
    smooth = moving_group_visual_smooth(
        journey_group(phase="node_transit", segment_node="CP1", dst="CP3"),
        current_tick=10,
    )

    for key in ("lane_id", "approach_id", "phase", "entry_approach_id", "transit_node"):
        assert smooth.get(key) == base.get(key)
    assert "x" in smooth and "y" in smooth


def test_smooth_visual_falls_back_when_timing_missing():
    # Degenerate timing (no journey window) must not crash and must reuse the
    # segment-based projection unchanged.
    plain = group(phase="edge", progress=0.5)
    fallback = moving_group_visual_smooth(plain, current_tick=7)
    base = moving_group_visual(plain)

    assert (fallback["x"], fallback["y"]) == pytest.approx((base["x"], base["y"]), abs=1e-6)

    zero_window = moving_group_visual_smooth(
        journey_group(started_tick=10, arrival_tick=10),
        current_tick=10,
    )
    assert "x" in zero_window and "y" in zero_window


def test_smooth_visual_resumed_mid_edge_does_not_snap_to_source_node():
    # Regression for the post-fight "teleport onto the node" (replay bug 1). A
    # survivor group that CONTINUES after an edge battle keeps started_tick reset
    # to "now" while it is already part-way along its edge. visual_start_frac records
    # the arc fraction it resumes at, so the smooth projector renders it on the edge
    # instead of snapping to path[0] (the edge's source node centre).
    resumed = {
        "src": "CP2", "dst": "CP4", "phase": "edge", "progress": 0.4,
        "segment_node": None, "path": ["CP2", "CP4"],
        "started_tick": 30, "arrival_tick": 33,
    }
    resumed["visual_start_frac"] = group_path_arc_fraction(resumed)

    spawn = moving_group_visual_smooth(resumed, current_tick=30)
    edge_point = point_on_approach_edge("CP2", "CP4", 0.4)
    # On the spawn tick it sits on the real mid-edge point, well off the CP2 centre.
    assert math.dist((spawn["x"], spawn["y"]), edge_point) < 2.0
    assert math.dist((spawn["x"], spawn["y"]), NODE_LAYOUT["CP2"].center) > 20.0

    # ...and it then glides FORWARD toward CP4, never back onto CP2.
    later = moving_group_visual_smooth(resumed, current_tick=32)
    assert math.dist((later["x"], later["y"]), NODE_LAYOUT["CP4"].center) < \
        math.dist((spawn["x"], spawn["y"]), NODE_LAYOUT["CP4"].center)

    # Contrast: WITHOUT the arc anchor the same group snaps to path[0] (the bug).
    buggy = {k: v for k, v in resumed.items() if k != "visual_start_frac"}
    buggy_spawn = moving_group_visual_smooth(buggy, current_tick=30)
    assert (buggy_spawn["x"], buggy_spawn["y"]) == \
        pytest.approx(NODE_LAYOUT["CP2"].center, abs=1e-3)


def test_smooth_visual_fresh_group_unaffected_by_arc_anchor():
    # A normal group departing from a node has visual_start_frac 0, so the anchor
    # is a strict no-op: identical to the legacy sweep from path[0].
    fresh = journey_group()
    zeroed = journey_group(visual_start_frac=0.0)
    for tick in (4, 8, 12, 20):
        a = moving_group_visual_smooth(fresh, current_tick=tick)
        b = moving_group_visual_smooth(zeroed, current_tick=tick)
        assert (a["x"], a["y"]) == pytest.approx((b["x"], b["y"]), abs=1e-9)
    start = moving_group_visual_smooth(fresh, current_tick=4)
    assert (start["x"], start["y"]) == pytest.approx(NODE_LAYOUT["A1"].center, abs=1e-3)
