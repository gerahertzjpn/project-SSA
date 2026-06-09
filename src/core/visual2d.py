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


def moving_group_visual_smooth(group: Any, current_tick: int) -> dict:
    """Visual projection with constant on-screen speed across the whole journey.

    ``moving_group_visual`` samples the unit at the engine's *per-segment*
    ``progress``. Because every segment is advanced in whole-tick chunks (with a
    ``ceil`` minimum of one tick and any sub-tick remainder discarded), the
    per-tick world displacement is uneven: a unit crawls across short
    node-exit/transit segments and sprints along long edges. Sampled at integer
    ticks the unit visibly accelerates and decelerates at every waypoint.

    This helper keeps gameplay untouched (the engine ``progress`` and arrival
    timing are unchanged) and only overrides the rendered ``x``/``y``/``heading``
    so the unit advances at a constant speed along its full path: position is the
    point at ``frac`` of the path's *length*, where ``frac`` is the elapsed
    fraction of the journey's *time* budget ``(tick - started) / (arrival -
    started)``. Walking time linearly into distance yields uniform speed.

    All segment metadata (lane/approach/phase/transit fields) is preserved from
    :func:`moving_group_visual` so threat-zone and lane logic keep working. If
    the path or timing is missing or degenerate the segment-based projection is
    returned unchanged.
    """

    base = moving_group_visual(group)

    raw_path = _field(group, "path", None)
    started = _field(group, "started_tick", _field(group, "started", None))
    arrival = _field(group, "arrival_tick", _field(group, "arrives", None))
    if not raw_path or started is None or arrival is None:
        return base

    path = [str(node) for node in raw_path]
    if len(path) < 2:
        return base

    denominator = float(arrival) - float(started)
    if denominator <= 0.0:
        return base
    frac = _clamp01((float(current_tick) - float(started)) / denominator)

    try:
        points = _path_points(path)
    except (KeyError, ValueError):
        return base
    point, heading = _point_along_polyline(points, frac)

    smooth = dict(base)
    smooth["x"] = round(point[0], 3)
    smooth["y"] = round(point[1], 3)
    smooth["heading"] = [round(heading[0], 4), round(heading[1], 4)]
    return smooth


def _path_points(path: list[str]) -> list[Point2D]:
    """Ordered polyline of a full movement path.

    Mirrors :meth:`GameEngine._movement_segments` geometry: each node hop is
    ``center(a) -> exit_point(a->b) -> entry_point(a->b) -> center(b)`` so the
    line follows node-exit, the lane edge, and node-enter/transit in order.
    """

    points: list[Point2D] = [NODE_LAYOUT[path[0]].center]
    for index in range(len(path) - 1):
        src, dst = path[index], path[index + 1]
        approach = get_approach(src, dst)
        points.append(approach.exit_point)
        points.append(approach.entry_point)
        points.append(NODE_LAYOUT[dst].center)
    return points


def _point_along_polyline(points: list[Point2D], frac: float) -> tuple[Point2D, Point2D]:
    """Point at ``frac`` of the polyline's arc length, plus the local heading."""

    seg_lengths = [_dist(points[i], points[i + 1]) for i in range(len(points) - 1)]
    total = sum(seg_lengths)
    if total <= 0.0:
        return points[0], (1.0, 0.0)

    target = _clamp01(frac) * total
    travelled = 0.0
    last_index = len(seg_lengths) - 1
    for index, length in enumerate(seg_lengths):
        if length <= 0.0:
            continue
        if travelled + length >= target or index == last_index:
            local = max(0.0, min(1.0, (target - travelled) / length))
            start, end = points[index], points[index + 1]
            point = (
                start[0] + (end[0] - start[0]) * local,
                start[1] + (end[1] - start[1]) * local,
            )
            return point, _heading_vector(start, end)
        travelled += length

    end_start, end_point = points[-2], points[-1]
    return end_point, _heading_vector(end_start, end_point)


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


def _heading_vector(start: Point2D, end: Point2D) -> Point2D:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length <= 0.0:
        return (1.0, 0.0)
    return (dx / length, dy / length)


def _dist(a: Point2D, b: Point2D) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


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
