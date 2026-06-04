import pytest

from src.core.map2d import NODE_LAYOUT, get_approach, point_on_approach_edge
from src.core.visual2d import moving_group_visual


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
