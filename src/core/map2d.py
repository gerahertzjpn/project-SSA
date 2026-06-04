"""Static 2D lane/zone metadata for the simple 2D engine migration.

The metadata is shared by replay/viewer projection and lightweight gameplay
zone checks. It intentionally stays deterministic and geometry-light so the RL
loop keeps operating on strategic aggregates instead of raw 2D physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .constants import CORE_ATTACK_RANGE, LORD_ATTACK_RANGE, NODE_POSITIONS, NodeType
from .map import ADJACENCY, ALL_NODES, CORE_NODES, NODE_TYPES


MAP2D_VERSION = "simple_2d_map_v0.1"

Point2D = tuple[float, float]

WORLD_SIZE = (680.0, 410.0)

CORE_RADIUS = 34.0
CONTROL_POINT_RADIUS = 30.0
BARON_RADIUS = 60.0

CORE_CAPTURE_RADIUS = 0.0
CONTROL_POINT_CAPTURE_RADIUS = 42.0
BARON_CAPTURE_RADIUS = 46.0

CORE_COMBAT_RADIUS = 72.0
CONTROL_POINT_COMBAT_RADIUS = 64.0
BARON_COMBAT_RADIUS = 88.0

ATTACK_RANGE_TO_MAP_UNITS = 24.0
LORD_THREAT_RADIUS = LORD_ATTACK_RANGE * ATTACK_RANGE_TO_MAP_UNITS
CORE_THREAT_RADIUS = CORE_ATTACK_RANGE * ATTACK_RANGE_TO_MAP_UNITS
MOVEMENT_MAP_UNITS_PER_TICK = 30.0


@dataclass(frozen=True)
class Node2D:
    node_id: str
    node_type: NodeType
    center: Point2D
    radius: float
    capture_radius: float
    combat_radius: float


@dataclass(frozen=True)
class Lane2D:
    lane_id: str
    a: str
    b: str
    points: tuple[Point2D, ...]
    length: float
    tunnel: bool = False


@dataclass(frozen=True)
class Approach2D:
    approach_id: str
    src: str
    dst: str
    lane_id: str
    entry_point: Point2D
    exit_point: Point2D


@dataclass(frozen=True)
class ThreatZone2D:
    zone_id: str
    source_node: str
    kind: str
    radius: float
    approach_ids: tuple[str, ...]


def canonical_lane_id(a: str, b: str) -> str:
    _require_adjacent(a, b)
    ai = ALL_NODES.index(a)
    bi = ALL_NODES.index(b)
    left, right = (a, b) if ai < bi else (b, a)
    return f"{left}|{right}"


def directed_approach_id(src: str, dst: str) -> str:
    _require_adjacent(src, dst)
    return f"{src}->{dst}"


def point_on_lane(src: str, dst: str, progress: float) -> Point2D:
    _require_adjacent(src, dst)
    start = NODE_LAYOUT[src].center
    end = NODE_LAYOUT[dst].center
    t = max(0.0, min(1.0, float(progress)))
    return (
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t,
    )


def point_on_approach_edge(src: str, dst: str, progress: float) -> Point2D:
    approach = get_approach(src, dst)
    return _point_between(approach.exit_point, approach.entry_point, progress)


def approach_edge_length(src: str, dst: str) -> float:
    approach = get_approach(src, dst)
    return _distance(approach.exit_point, approach.entry_point)


def node_exit_length(node_id: str) -> float:
    return NODE_LAYOUT[node_id].radius


def node_enter_length(node_id: str) -> float:
    return NODE_LAYOUT[node_id].radius


def node_transit_length(prev_node: str, node_id: str, next_node: str) -> float:
    center = NODE_LAYOUT[node_id].center
    return (
        _distance(get_approach(prev_node, node_id).entry_point, center) +
        _distance(center, get_approach(node_id, next_node).exit_point)
    )


def get_lane(src: str, dst: str) -> Lane2D:
    return LANES[canonical_lane_id(src, dst)]


def get_approach(src: str, dst: str) -> Approach2D:
    return APPROACHES[directed_approach_id(src, dst)]


def incoming_approaches(node_id: str) -> tuple[Approach2D, ...]:
    _require_node(node_id)
    return tuple(APPROACHES[directed_approach_id(src, node_id)] for src in ADJACENCY[node_id])


def outgoing_approaches(node_id: str) -> tuple[Approach2D, ...]:
    _require_node(node_id)
    return tuple(APPROACHES[directed_approach_id(node_id, dst)] for dst in ADJACENCY[node_id])


def is_lord_approach(src: str, dst: str) -> bool:
    return dst == "Baron" and src in ADJACENCY["Baron"]


def is_core_approach(src: str, dst: str) -> bool:
    return dst in CORE_NODES and src in ADJACENCY[dst]


def threat_zone_ids_for_approach(approach_id: str) -> tuple[str, ...]:
    return tuple(
        zone_id
        for zone_id, zone in THREAT_ZONES.items()
        if approach_id in zone.approach_ids
    )


def threat_zone_entry_progress(approach_id: str, zone_id: str) -> float:
    """Return lane progress at which an approach enters a threat zone."""

    approach = APPROACHES[approach_id]
    zone = THREAT_ZONES[zone_id]
    source = NODE_LAYOUT[zone.source_node].center
    start = approach.exit_point
    end = approach.entry_point

    # Current threat zones are centered on the destination objective. If a
    # future zone is not, fall back to the first point where the linear approach
    # gets close enough to the zone source.
    if zone.source_node == approach.dst:
        length = max(0.001, _distance(start, end))
        node_radius = NODE_LAYOUT[approach.dst].radius
        return max(0.0, min(1.0, 1.0 - max(0.0, zone.radius - node_radius) / length))

    samples = 20
    for index in range(samples + 1):
        progress = index / samples
        point = _point_between(start, end, progress)
        if _distance(point, source) <= zone.radius:
            return progress
    return 1.0


def node_zone_progress_interval(approach_id: str, node_id: str, radius: float) -> tuple[float, float]:
    """Return the lane progress interval covered by a node-centered radius."""

    _require_node(node_id)
    approach = APPROACHES[approach_id]
    edge_length = max(0.001, _distance(approach.exit_point, approach.entry_point))
    node_radius = NODE_LAYOUT[node_id].radius
    clipped = max(0.0, min(1.0, (float(radius) - node_radius) / edge_length))
    if node_id == approach.src:
        return 0.0, clipped
    if node_id == approach.dst:
        return max(0.0, 1.0 - clipped), 1.0

    center = NODE_LAYOUT[node_id].center
    inside: list[float] = []
    samples = 20
    for index in range(samples + 1):
        progress = index / samples
        point = _point_between(approach.exit_point, approach.entry_point, progress)
        if _distance(point, center) <= radius:
            inside.append(progress)
    if not inside:
        return 1.0, 0.0
    return min(inside), max(inside)


def map2d_replay_meta() -> dict:
    """Return JSON-serializable static map metadata for replay/viewer use."""

    return {
        "version": MAP2D_VERSION,
        "world_size": list(WORLD_SIZE),
        "nodes": {
            node_id: {
                "type": node.node_type.name,
                "center": list(node.center),
                "radius": node.radius,
                "capture_radius": node.capture_radius,
                "combat_radius": node.combat_radius,
            }
            for node_id, node in NODE_LAYOUT.items()
        },
        "lanes": {
            lane_id: {
                "a": lane.a,
                "b": lane.b,
                "points": [list(point) for point in lane.points],
                "length": lane.length,
                "tunnel": lane.tunnel,
            }
            for lane_id, lane in LANES.items()
        },
        "approaches": {
            approach_id: {
                "src": approach.src,
                "dst": approach.dst,
                "lane_id": approach.lane_id,
                "entry_point": list(approach.entry_point),
                "exit_point": list(approach.exit_point),
            }
            for approach_id, approach in APPROACHES.items()
        },
        "threat_zones": {
            zone_id: {
                "source_node": zone.source_node,
                "kind": zone.kind,
                "radius": zone.radius,
                "approach_ids": list(zone.approach_ids),
            }
            for zone_id, zone in THREAT_ZONES.items()
        },
    }


def _build_nodes() -> dict[str, Node2D]:
    nodes: dict[str, Node2D] = {}
    for node_id in ALL_NODES:
        node_type = NODE_TYPES[node_id]
        if node_type == NodeType.CORE:
            radius = CORE_RADIUS
            capture_radius = CORE_CAPTURE_RADIUS
            combat_radius = CORE_COMBAT_RADIUS
        elif node_type == NodeType.BARON:
            radius = BARON_RADIUS
            capture_radius = BARON_CAPTURE_RADIUS
            combat_radius = BARON_COMBAT_RADIUS
        else:
            radius = CONTROL_POINT_RADIUS
            capture_radius = CONTROL_POINT_CAPTURE_RADIUS
            combat_radius = CONTROL_POINT_COMBAT_RADIUS
        x, y = NODE_POSITIONS[node_id]
        nodes[node_id] = Node2D(
            node_id=node_id,
            node_type=node_type,
            center=(float(x), float(y)),
            radius=radius,
            capture_radius=capture_radius,
            combat_radius=combat_radius,
        )
    return nodes


def _build_lanes() -> dict[str, Lane2D]:
    lanes: dict[str, Lane2D] = {}
    for src, neighbors in ADJACENCY.items():
        for dst in neighbors:
            lane_id = canonical_lane_id(src, dst)
            if lane_id in lanes:
                continue
            start = NODE_LAYOUT[src].center
            end = NODE_LAYOUT[dst].center
            lanes[lane_id] = Lane2D(
                lane_id=lane_id,
                a=src if ALL_NODES.index(src) < ALL_NODES.index(dst) else dst,
                b=dst if ALL_NODES.index(src) < ALL_NODES.index(dst) else src,
                points=(start, end),
                length=_distance(start, end),
                tunnel=src in CORE_NODES or dst in CORE_NODES,
            )
    return lanes


def _build_approaches() -> dict[str, Approach2D]:
    approaches: dict[str, Approach2D] = {}
    for src, neighbors in ADJACENCY.items():
        for dst in neighbors:
            approach_id = directed_approach_id(src, dst)
            approaches[approach_id] = Approach2D(
                approach_id=approach_id,
                src=src,
                dst=dst,
                lane_id=canonical_lane_id(src, dst),
                entry_point=_boundary_point(src, dst, NODE_LAYOUT[dst].radius),
                exit_point=_boundary_point(dst, src, NODE_LAYOUT[src].radius),
            )
    return approaches


def _build_threat_zones() -> dict[str, ThreatZone2D]:
    zones: dict[str, ThreatZone2D] = {}
    zones["Lord"] = ThreatZone2D(
        zone_id="Lord",
        source_node="Baron",
        kind="lord",
        radius=LORD_THREAT_RADIUS,
        approach_ids=tuple(
            directed_approach_id(src, "Baron")
            for src in ADJACENCY["Baron"]
        ),
    )
    for core_id in CORE_NODES:
        zones[f"core:{core_id}"] = ThreatZone2D(
            zone_id=f"core:{core_id}",
            source_node=core_id,
            kind="core",
            radius=CORE_THREAT_RADIUS,
            approach_ids=tuple(
                directed_approach_id(src, core_id)
                for src in ADJACENCY[core_id]
            ),
        )
    return zones


def _boundary_point(src: str, dst: str, radius: float) -> Point2D:
    src_point = NODE_LAYOUT[src].center
    dst_point = NODE_LAYOUT[dst].center
    dx = src_point[0] - dst_point[0]
    dy = src_point[1] - dst_point[1]
    distance = hypot(dx, dy)
    if distance <= 0.0:
        return dst_point
    scale = max(0.0, float(radius)) / distance
    return (dst_point[0] + dx * scale, dst_point[1] + dy * scale)


def _distance(a: Point2D, b: Point2D) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _point_between(start: Point2D, end: Point2D, progress: float) -> Point2D:
    t = max(0.0, min(1.0, float(progress)))
    return (
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t,
    )


def _require_node(node_id: str) -> None:
    if node_id not in ADJACENCY:
        raise KeyError(f"unknown node: {node_id}")


def _require_adjacent(src: str, dst: str) -> None:
    _require_node(src)
    _require_node(dst)
    if dst not in ADJACENCY[src]:
        raise ValueError(f"{src} and {dst} are not adjacent")


NODE_LAYOUT = _build_nodes()
LANES = _build_lanes()
APPROACHES = _build_approaches()
THREAT_ZONES = _build_threat_zones()
