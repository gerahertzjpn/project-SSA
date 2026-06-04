"""Visual-only 2D projection helpers.

These helpers turn existing strategic movement phases into stable map
positions for replay/viewer use. They must not decide gameplay outcomes.
"""

from __future__ import annotations

from typing import Any

from .map2d import (
    NODE_LAYOUT,
    Approach2D,
    Point2D,
    canonical_lane_id,
    directed_approach_id,
    get_approach,
    point_on_approach_edge,
)


def moving_group_visual(group: Any) -> dict:
    src = str(_field(group, "src"))
    dst = str(_field(group, "dst"))
    phase = str(_field(group, "phase", "edge") or "edge")
    progress = _clamp01(float(_field(group, "progress", 0.0) or 0.0))
    segment_node = _field(group, "segment_node", None)

    if phase == "node_exit":
        point = _node_exit_point(src, dst, progress)
        lane_src, lane_dst = src, dst
        entry_approach_id = None
    elif phase == "node_enter":
        point = _node_enter_point(src, dst, progress)
        lane_src, lane_dst = src, dst
        entry_approach_id = directed_approach_id(src, dst)
    elif phase == "node_transit" and segment_node:
        point = _node_transit_point(src, str(segment_node), dst, progress)
        lane_src, lane_dst = str(segment_node), dst
        entry_approach_id = directed_approach_id(src, str(segment_node))
    else:
        point = point_on_approach_edge(src, dst, progress)
        lane_src, lane_dst = src, dst
        entry_approach_id = None

    heading = _heading(lane_src, lane_dst)
    visual = {
        "x": round(point[0], 3),
        "y": round(point[1], 3),
        "heading": [round(heading[0], 4), round(heading[1], 4)],
        "phase": phase,
        "lane_id": canonical_lane_id(lane_src, lane_dst),
        "approach_id": directed_approach_id(lane_src, lane_dst),
    }
    if entry_approach_id:
        visual["entry_approach_id"] = entry_approach_id
    if phase == "node_transit" and segment_node:
        visual["transit_node"] = str(segment_node)
    return visual


def _node_exit_point(src: str, dst: str, progress: float) -> Point2D:
    approach = get_approach(src, dst)
    return _lerp(NODE_LAYOUT[src].center, approach.exit_point, progress)


def _node_enter_point(src: str, dst: str, progress: float) -> Point2D:
    approach = get_approach(src, dst)
    return _lerp(approach.entry_point, NODE_LAYOUT[dst].center, progress)


def _node_transit_point(prev_node: str, node_id: str, next_node: str, progress: float) -> Point2D:
    entry = get_approach(prev_node, node_id).entry_point
    center = NODE_LAYOUT[node_id].center
    exit_ = get_approach(node_id, next_node).exit_point
    if progress <= 0.5:
        return _lerp(entry, center, progress * 2.0)
    return _lerp(center, exit_, (progress - 0.5) * 2.0)


def _heading(src: str, dst: str) -> Point2D:
    start = NODE_LAYOUT[src].center
    end = NODE_LAYOUT[dst].center
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length <= 0.0:
        return (1.0, 0.0)
    return (dx / length, dy / length)


def _lerp(start: Point2D, end: Point2D, progress: float) -> Point2D:
    t = _clamp01(progress)
    return (
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _field(group: Any, name: str, default: Any = None) -> Any:
    if isinstance(group, dict):
        return group.get(name, default)
    return getattr(group, name, default)
