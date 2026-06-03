from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.core.constants import (
    BARON_HP,
    CORE_HP,
    DEATH_TIME_START_TICK,
    PLAYER_IDS,
    TEAM_A_PLAYERS,
    TEAM_B_PLAYERS,
    TICK_LIMIT,
    TRAVEL_TIME,
    UNIT_COST,
    UnitType,
    ActionType,
)
from src.core.engine import Action
from src.core.map import ADJACENCY, ALL_NODES, NODE_TYPES, NodeType
from src.core.units import UNIT_STATS


OBSERVATION_VERSION = "train_obs_v0.5"

UNIT_ORDER = (UnitType.VANGUARD, UnitType.RANGER, UnitType.SKIRMISHER)
RELATIONS = ("self", "ally", "enemy", "neutral")
COMBAT_RELATIONS = ("self", "ally", "enemy")
MOVEMENT_BUCKETS = ("inside", "outgoing", "incoming")
MOVE_INTENTS = ("CAPTURE", "TRANSIT")
GLOBAL_FEATURES = 14
NODE_TYPE_FEATURES = 3
OWNER_RELATION_FEATURES = len(RELATIONS)
STATIONED_UNIT_FEATURES = len(RELATIONS) * len(UNIT_ORDER)
NODE_SCALAR_FEATURES = 9
MOVEMENT_FEATURES = len(MOVEMENT_BUCKETS) * len(COMBAT_RELATIONS) * len(UNIT_ORDER)
BATTLE_FEATURES = 4 + len(COMBAT_RELATIONS) * len(UNIT_ORDER)
NODE_FEATURES = (
    NODE_TYPE_FEATURES
    + OWNER_RELATION_FEATURES
    + STATIONED_UNIT_FEATURES
    + NODE_SCALAR_FEATURES
    + MOVEMENT_FEATURES
    + BATTLE_FEATURES
    + BATTLE_FEATURES
)
OBSERVATION_DIM = GLOBAL_FEATURES + len(ALL_NODES) * NODE_FEATURES
NODE_HALF_CROSSING_TICKS = float(TRAVEL_TIME) / 2.0
ROUTE_EDGE_NORM = max(1, len(ALL_NODES) - 1)
ROUTE_TICK_NORM = float(TICK_LIMIT)


@dataclass(frozen=True)
class ActionSpec:
    kind: str
    unit_type: Optional[UnitType] = None
    source: Optional[str] = None
    destination: Optional[str] = None
    intent: str = "CAPTURE"


ACTION_SPECS: tuple[ActionSpec, ...] = (
    (ActionSpec("HOLD"),)
    + tuple(ActionSpec("PRODUCE", unit_type=unit_type) for unit_type in UNIT_ORDER)
    + tuple(
        ActionSpec("MOVE", source=source_id, destination=destination_id, intent=intent)
        for intent in MOVE_INTENTS
        for source_id in ALL_NODES
        for destination_id in ALL_NODES
        if source_id != destination_id
    )
)
ACTION_DIM = len(ACTION_SPECS)

MIRROR_NODES = {
    "A1": "B1",
    "A2": "B2",
    "A3": "B3",
    "B1": "A1",
    "B2": "A2",
    "B3": "A3",
    "CP1": "CP5",
    "CP2": "CP6",
    "CP3": "CP3",
    "CP4": "CP4",
    "CP5": "CP1",
    "CP6": "CP2",
    "Baron": "Baron",
}
MIRROR_PLAYERS = {
    "A1": "B1",
    "A2": "B2",
    "A3": "B3",
    "B1": "A1",
    "B2": "A2",
    "B3": "A3",
}
MIRROR_TEAMS = {
    "TEAM_A": "TEAM_B",
    "TEAM_B": "TEAM_A",
}
MIRROR_ENTRY_SIDES = {
    "attacker": "defender",
    "defender": "attacker",
}
COMBAT_LINE_MAX = 20.0

_ACTION_INDEX_BY_SPEC = {
    (spec.kind, spec.unit_type, spec.source, spec.destination, spec.intent): index
    for index, spec in enumerate(ACTION_SPECS)
}


def team_of(player_id: str) -> str:
    if player_id in TEAM_A_PLAYERS:
        return "TEAM_A"
    if player_id in TEAM_B_PLAYERS:
        return "TEAM_B"
    return "NEUTRAL"


def mirrored_node_id(node_id: str | None) -> str | None:
    if node_id is None:
        return None
    return MIRROR_NODES.get(str(node_id), str(node_id))


def mirrored_player_id(player_id: str | None) -> str | None:
    if player_id is None:
        return None
    return MIRROR_PLAYERS.get(str(player_id), str(player_id))


def mirrored_team_id(team_id: str | None) -> str | None:
    if team_id is None:
        return None
    return MIRROR_TEAMS.get(str(team_id), str(team_id))


def mirrored_entry_side(side: str | None) -> str | None:
    if side is None:
        return None
    return MIRROR_ENTRY_SIDES.get(str(side), str(side))


def uses_mirrored_policy_perspective(player_id: str) -> bool:
    return team_of(player_id) == "TEAM_B"


def policy_player_id(player_id: str) -> str:
    return mirrored_player_id(player_id) if uses_mirrored_policy_perspective(player_id) else player_id


def policy_observation(obs: dict, player_id: str) -> tuple[dict, str]:
    if not uses_mirrored_policy_perspective(player_id):
        return obs, player_id
    return mirror_observation(obs), policy_player_id(player_id)


def mirror_action_index(action_index: int) -> int:
    if action_index < 0 or action_index >= ACTION_DIM:
        return 0
    spec = ACTION_SPECS[int(action_index)]
    mirrored = ActionSpec(
        kind=spec.kind,
        unit_type=spec.unit_type,
        source=mirrored_node_id(spec.source),
        destination=mirrored_node_id(spec.destination),
        intent=spec.intent,
    )
    return _ACTION_INDEX_BY_SPEC.get(
        (mirrored.kind, mirrored.unit_type, mirrored.source, mirrored.destination, mirrored.intent),
        0,
    )


def mirror_observation(obs: dict) -> dict:
    mirrored = dict(obs)
    mirrored["player_id"] = mirrored_player_id(obs.get("player_id"))
    mirrored["team"] = mirrored_team_id(obs.get("team"))
    mirrored["core_node"] = mirrored_node_id(obs.get("core_node"))
    if "winner" in mirrored:
        mirrored["winner"] = mirrored_team_id(obs.get("winner"))

    if isinstance(obs.get("nodes"), dict):
        mirrored["nodes"] = {
            mirrored_node_id(node_id): _mirror_node(node)
            for node_id, node in obs["nodes"].items()
        }
    if isinstance(obs.get("players"), dict):
        mirrored["players"] = {
            mirrored_player_id(player_id): _mirror_player_state(player)
            for player_id, player in obs["players"].items()
        }

    mirrored["moving_groups"] = [
        _mirror_moving_group(group)
        for group in obs.get("moving_groups", []) or []
    ]
    mirrored["active_battles"] = [
        _mirror_battle(battle)
        for battle in obs.get("active_battles", []) or []
    ]

    if isinstance(obs.get("baron"), dict):
        mirrored["baron"] = _mirror_baron(obs["baron"])
    if isinstance(obs.get("core_damage_by_team"), dict):
        mirrored["core_damage_by_team"] = {
            mirrored_team_id(team): value
            for team, value in obs["core_damage_by_team"].items()
        }
    if isinstance(obs.get("death_time_final_core_deaths"), dict):
        mirrored["death_time_final_core_deaths"] = {
            mirrored_team_id(team): value
            for team, value in obs["death_time_final_core_deaths"].items()
        }
    return mirrored


def _mirror_node(node: dict) -> dict:
    mirrored = dict(node)
    mirrored["owner"] = mirrored_player_id(node.get("owner"))
    mirrored["visiting_units"] = {
        mirrored_player_id(player_id): dict(units)
        for player_id, units in (node.get("visiting_units", {}) or {}).items()
    }
    mirrored["unit_ready_tick"] = {
        mirrored_player_id(player_id): ready_tick
        for player_id, ready_tick in (node.get("unit_ready_tick", {}) or {}).items()
    }
    mirrored["unit_entry_side"] = {
        mirrored_player_id(player_id): mirrored_entry_side(side)
        for player_id, side in (node.get("unit_entry_side", {}) or {}).items()
    }
    mirrored["neutral_aggro_teams"] = [
        mirrored_team_id(team)
        for team in node.get("neutral_aggro_teams", []) or []
    ]
    return mirrored


def _mirror_player_state(player: dict) -> dict:
    mirrored = dict(player)
    if "player_id" in mirrored:
        mirrored["player_id"] = mirrored_player_id(player.get("player_id"))
    if "team" in mirrored:
        mirrored["team"] = mirrored_team_id(player.get("team"))
    if "core_node" in mirrored:
        mirrored["core_node"] = mirrored_node_id(player.get("core_node"))
    return mirrored


def _mirror_moving_group(group: dict) -> dict:
    mirrored = dict(group)
    mirrored["player"] = mirrored_player_id(group.get("player"))
    if "team" in mirrored:
        mirrored["team"] = mirrored_team_id(group.get("team"))
    for key in ("src", "dst", "final_dst", "segment_node"):
        if key in mirrored:
            mirrored[key] = mirrored_node_id(group.get(key))
    if isinstance(group.get("path"), list):
        mirrored["path"] = [mirrored_node_id(node_id) for node_id in group["path"]]
    return mirrored


def _mirror_baron(baron: dict) -> dict:
    mirrored = dict(baron)
    for key in ("controlled_by", "allied_lord_owner"):
        if key in mirrored:
            mirrored[key] = mirrored_player_id(baron.get(key))
    if "allied_lord_target" in mirrored:
        mirrored["allied_lord_target"] = mirrored_node_id(baron.get("allied_lord_target"))
    return mirrored


def _mirror_battle(battle: dict) -> dict:
    mirrored = dict(battle)
    kind = battle.get("location_kind")
    mirrored_location = _mirror_battle_location(kind, battle.get("location_id"))
    mirrored["location_id"] = mirrored_location
    if battle.get("battle_id"):
        prefix = "edge" if kind == "edge" else "node"
        mirrored["battle_id"] = f"{prefix}:{mirrored_location}"
    mirrored["participants"] = [
        mirrored_player_id(player_id)
        for player_id in battle.get("participants", []) or []
    ]
    mirrored["entry_sides"] = {
        mirrored_player_id(player_id): mirrored_entry_side(side)
        for player_id, side in (battle.get("entry_sides", {}) or {}).items()
    }
    mirrored["neutral_aggro_teams"] = [
        mirrored_team_id(team)
        for team in battle.get("neutral_aggro_teams", []) or []
    ]
    mirrored["sides"] = {
        mirrored_player_id(side_id): dict(side)
        for side_id, side in (battle.get("sides", {}) or {}).items()
    }
    mirrored["formation"] = [
        _mirror_formation_unit(unit)
        for unit in battle.get("formation", []) or []
    ]
    return mirrored


def _mirror_battle_location(kind: str | None, location_id: str | None) -> str | None:
    if location_id is None:
        return None
    if kind == "edge":
        parts = str(location_id).split("|", 1)
        if len(parts) == 2:
            return "|".join(sorted((mirrored_node_id(parts[0]), mirrored_node_id(parts[1]))))
    return mirrored_node_id(str(location_id))


def _mirror_formation_unit(unit: list) -> list:
    mirrored = list(unit)
    if len(mirrored) > 1:
        mirrored[1] = mirrored_player_id(mirrored[1])
    if len(mirrored) > 5:
        try:
            mirrored[5] = round(COMBAT_LINE_MAX - float(mirrored[5]), 2)
        except (TypeError, ValueError):
            pass
    return mirrored


def owner_relation(owner: str, player_id: str) -> str:
    if owner == player_id:
        return "self"
    if owner == "NEUTRAL" or owner in {"BARON", "LORD"}:
        return "neutral"
    if team_of(owner) == team_of(player_id):
        return "ally"
    return "enemy"


def unit_total(units: dict | None) -> int:
    return sum(int(count or 0) for count in (units or {}).values())


def relation_index(relation: str) -> int:
    return RELATIONS.index(relation)


def _friendly_owner(owner: str, player_id: str) -> bool:
    return owner != "NEUTRAL" and team_of(owner) == team_of(player_id)


def _living_neutral_lord(node_id: str, nodes: dict, baron: dict | None) -> bool:
    return (
        node_id == "Baron"
        and nodes.get("Baron", {}).get("owner") == "NEUTRAL"
        and float((baron or {}).get("hp", 0) or 0) > 0
    )


def _is_transit_node(player_id: str, node_id: str, nodes: dict, baron: dict | None) -> bool:
    if _living_neutral_lord(node_id, nodes, baron):
        return False
    owner = nodes[node_id]["owner"]
    return owner == "NEUTRAL" or _friendly_owner(owner, player_id)


def _is_accessible_destination(
    player_id: str,
    node_id: str,
    via_node: str,
    nodes: dict,
    baron: dict | None,
) -> bool:
    if node_id == "Baron":
        return True
    owner = nodes[node_id]["owner"]
    if owner == "NEUTRAL" or _friendly_owner(owner, player_id):
        return True
    return _is_transit_node(player_id, via_node, nodes, baron)


def find_route_obs(player_id: str, src: str, dst: str, obs: dict) -> Optional[list[str]]:
    nodes = obs["nodes"]
    baron = obs.get("baron", {})
    if src == dst or src not in nodes or dst not in nodes:
        return None

    visited = {src}
    queue = deque([[src]])
    while queue:
        path = queue.popleft()
        node_id = path[-1]
        for neighbor in ADJACENCY.get(node_id, []):
            if neighbor in visited:
                continue
            next_path = path + [neighbor]
            if _is_accessible_destination(player_id, neighbor, node_id, nodes, baron):
                if neighbor == dst:
                    return next_path
            if _is_transit_node(player_id, neighbor, nodes, baron):
                visited.add(neighbor)
                queue.append(next_path)
    return None


def _source_units_for_player(node: dict, player_id: str) -> dict:
    if node.get("owner") == player_id:
        return node.get("units", {})
    return (node.get("visiting_units", {}) or {}).get(player_id, {})


def group_speed(units: dict | None) -> float:
    speeds = []
    for unit_type, count in (units or {}).items():
        if int(count or 0) <= 0:
            continue
        unit_key = unit_type.value if isinstance(unit_type, UnitType) else unit_type
        unit_enum = UnitType.RANGER if unit_key == "ARCHER" else UnitType(unit_key)
        speeds.append(float(UNIT_STATS[unit_enum]["speed"]))
    return min(speeds) if speeds else 1.0


def estimate_route_ticks(path: list[str], speed: float) -> float:
    if len(path) <= 1:
        return 0.0
    edge_count = len(path) - 1
    intermediate_count = max(0, len(path) - 2)
    base_ticks = (
        NODE_HALF_CROSSING_TICKS
        + float(TRAVEL_TIME) * edge_count
        + float(TRAVEL_TIME) * intermediate_count
        + NODE_HALF_CROSSING_TICKS
    )
    return base_ticks / max(0.001, float(speed))


def movable_sources(obs: dict, player_id: str) -> list[str]:
    tick = int(obs.get("tick", 0) or 0)
    sources = []
    for node_id in ALL_NODES:
        node = obs["nodes"][node_id]
        if int((node.get("unit_ready_tick", {}) or {}).get(player_id, 0) or 0) > tick:
            continue
        if unit_total(_source_units_for_player(node, player_id)) > 0:
            sources.append(node_id)
    return sources


def select_source_for_destination(obs: dict, player_id: str, destination: str) -> Optional[str]:
    candidates = []
    for source in movable_sources(obs, player_id):
        route = find_route_obs(player_id, source, destination, obs)
        if not route:
            continue
        units = _source_units_for_player(obs["nodes"][source], player_id)
        candidates.append((-unit_total(units), len(route), ALL_NODES.index(source), source))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][3]


def route_metrics_for_node(obs: dict, player_id: str, destination: str) -> tuple[float, float, float]:
    candidates = []
    for source in movable_sources(obs, player_id):
        units = _source_units_for_player(obs["nodes"][source], player_id)
        if source == destination:
            candidates.append((0.0, 0.0, -unit_total(units), ALL_NODES.index(source)))
            continue
        route = find_route_obs(player_id, source, destination, obs)
        if not route:
            continue
        speed = group_speed(units)
        edges = float(len(route) - 1)
        ticks = estimate_route_ticks(route, speed)
        candidates.append((ticks, edges, -unit_total(units), ALL_NODES.index(source)))

    if not candidates:
        return 0.0, 1.0, 1.0

    candidates.sort()
    ticks, edges, _, _ = candidates[0]
    return (
        1.0,
        min(1.0, edges / float(ROUTE_EDGE_NORM)),
        min(1.0, ticks / ROUTE_TICK_NORM),
    )


def _intent_allowed(spec: ActionSpec, obs: dict, player_id: str) -> bool:
    if spec.kind != "MOVE":
        return True
    destination = spec.destination
    if not destination:
        return False
    if spec.intent == "CAPTURE":
        return True
    nodes = obs["nodes"]
    if _living_neutral_lord(destination, nodes, obs.get("baron", {})):
        return False
    owner = nodes[destination]["owner"]
    return owner == "NEUTRAL" or _friendly_owner(owner, player_id)


def legal_action_mask(obs: dict, player_id: str) -> np.ndarray:
    policy_obs, policy_player = policy_observation(obs, player_id)
    return _legal_action_mask_absolute(policy_obs, policy_player)


def _legal_action_mask_absolute(obs: dict, player_id: str) -> np.ndarray:
    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    resources = float(obs.get("resources", 0.0) or 0.0)
    mask[0] = True

    for index, spec in enumerate(ACTION_SPECS):
        if spec.kind == "PRODUCE" and spec.unit_type is not None:
            mask[index] = resources >= UNIT_COST[spec.unit_type]
        elif spec.kind == "MOVE" and spec.source and spec.destination:
            if not _intent_allowed(spec, obs, player_id):
                continue
            if spec.source not in movable_sources(obs, player_id):
                continue
            mask[index] = find_route_obs(player_id, spec.source, spec.destination, obs) is not None
    return mask


def decode_action(action_index: int, obs: dict, player_id: str) -> Action:
    policy_obs, policy_player = policy_observation(obs, player_id)
    return _decode_policy_action(action_index, policy_obs, policy_player, player_id)


def _decode_policy_action(action_index: int, policy_obs: dict, policy_player: str, actual_player: str) -> Action:
    if action_index < 0 or action_index >= ACTION_DIM:
        return Action(ActionType.HOLD)

    spec = ACTION_SPECS[int(action_index)]
    if spec.kind == "HOLD":
        return Action(ActionType.HOLD)
    if spec.kind == "PRODUCE" and spec.unit_type is not None:
        if _legal_action_mask_absolute(policy_obs, policy_player)[action_index]:
            return Action(ActionType.PRODUCE, unit_type=spec.unit_type)
        return Action(ActionType.HOLD)
    if spec.kind == "MOVE" and spec.source and spec.destination:
        if not _legal_action_mask_absolute(policy_obs, policy_player)[action_index]:
            return Action(ActionType.HOLD)
        source = spec.source
        destination = spec.destination
        if uses_mirrored_policy_perspective(actual_player):
            source = mirrored_node_id(source)
            destination = mirrored_node_id(destination)
        return Action(
            ActionType.MOVE,
            source=source,
            destination=destination,
            intent=spec.intent,
        )
    return Action(ActionType.HOLD)


def _normal_unit_counts(units: dict | None) -> list[float]:
    return [float((units or {}).get(unit_type.value, 0) or 0) / 50.0 for unit_type in UNIT_ORDER]


def _add_normal_unit_counts(target: list[float], offset: int, units: dict | None) -> None:
    for unit_index, unit_type in enumerate(UNIT_ORDER):
        target[offset + unit_index] += float((units or {}).get(unit_type.value, 0) or 0) / 50.0


def _team_resource_totals(obs: dict, player_id: str) -> tuple[float, float]:
    own_team = team_of(player_id)
    own = 0.0
    enemy = 0.0
    for pid, player in (obs.get("players", {}) or {}).items():
        if not player.get("alive", True):
            continue
        if team_of(pid) == own_team:
            own += float(player.get("resources", 0.0) or 0.0)
        else:
            enemy += float(player.get("resources", 0.0) or 0.0)
    return own, enemy


def _team_alive_counts(obs: dict, player_id: str) -> tuple[int, int]:
    own_team = team_of(player_id)
    own = 0
    enemy = 0
    for pid, player in (obs.get("players", {}) or {}).items():
        if not player.get("alive", True):
            continue
        if team_of(pid) == own_team:
            own += 1
        else:
            enemy += 1
    return own, enemy


def _moving_node_totals(obs: dict, player_id: str) -> dict[str, list[float]]:
    totals = {node_id: [0.0] * MOVEMENT_FEATURES for node_id in ALL_NODES}

    def add_units(node_id: str | None, bucket: str, relation: str, units: dict | None) -> None:
        if node_id not in totals or relation not in COMBAT_RELATIONS:
            return
        bucket_index = MOVEMENT_BUCKETS.index(bucket)
        relation_index_ = COMBAT_RELATIONS.index(relation)
        offset = (
            bucket_index * len(COMBAT_RELATIONS) * len(UNIT_ORDER)
            + relation_index_ * len(UNIT_ORDER)
        )
        _add_normal_unit_counts(totals[node_id], offset, units)

    for group in obs.get("moving_groups", []) or []:
        relation = owner_relation(group.get("player", "NEUTRAL"), player_id)
        if relation == "neutral":
            continue
        units = group.get("units", {})
        phase = group.get("phase", "edge")
        inside_node = group.get("segment_node") if phase != "edge" else None
        add_units(inside_node, "inside", relation, units)
        add_units(group.get("src"), "outgoing", relation, units)
        add_units(group.get("dst"), "incoming", relation, units)
    return totals


def _battle_node_totals(obs: dict, player_id: str) -> dict[str, list[float]]:
    totals = {node_id: [0.0] * BATTLE_FEATURES for node_id in ALL_NODES}
    for battle in obs.get("active_battles", []) or []:
        if battle.get("location_kind") != "node":
            continue
        node_id = battle.get("location_id")
        if node_id not in totals:
            continue
        _add_battle_features(totals[node_id], battle, player_id, weight=1.0)
    return totals


def _edge_battle_node_totals(obs: dict, player_id: str) -> dict[str, list[float]]:
    totals = {node_id: [0.0] * BATTLE_FEATURES for node_id in ALL_NODES}
    for battle in obs.get("active_battles", []) or []:
        if battle.get("location_kind") != "edge":
            continue
        endpoints = _edge_battle_endpoints(battle)
        if not endpoints:
            continue
        for node_id in endpoints:
            if node_id in totals:
                _add_battle_features(totals[node_id], battle, player_id, weight=1.0)
    return totals


def _edge_battle_endpoints(battle: dict) -> list[str]:
    location_id = str(battle.get("location_id", "") or "")
    if "|" not in location_id:
        return []
    endpoints = [part for part in location_id.split("|", 1) if part in ALL_NODES]
    return endpoints if len(endpoints) == 2 else []


def _add_battle_features(features: list[float], battle: dict, player_id: str, weight: float) -> None:
    friendly_hp = 0.0
    enemy_hp = 0.0
    neutral_hp = 0.0
    for side_id, side in (battle.get("sides", {}) or {}).items():
        hp = float(side.get("hp", 0.0) or 0.0) * weight
        relation = owner_relation(side_id, player_id)
        if relation in {"self", "ally"}:
            friendly_hp += hp
        elif relation == "enemy":
            enemy_hp += hp
        else:
            neutral_hp += hp
        if relation in COMBAT_RELATIONS:
            relation_index_ = COMBAT_RELATIONS.index(relation)
            offset = 4 + relation_index_ * len(UNIT_ORDER)
            _add_normal_unit_counts(features, offset, side.get("units", {}))

    features[0] += friendly_hp / 200.0
    features[1] += enemy_hp / 200.0
    features[2] += neutral_hp / 100.0
    features[3] = max(
        features[3],
        float(battle.get("age_ticks", 0) or 0) / float(TICK_LIMIT),
    )


def encode_observation(obs: dict, player_id: str) -> np.ndarray:
    policy_obs, policy_player = policy_observation(obs, player_id)
    return _encode_observation_absolute(policy_obs, policy_player)


def _encode_observation_absolute(obs: dict, player_id: str) -> np.ndarray:
    own_resources, enemy_resources = _team_resource_totals(obs, player_id)
    own_alive, enemy_alive = _team_alive_counts(obs, player_id)
    baron_hp = float((obs.get("baron", {}) or {}).get("hp", 0.0) or 0.0)

    features: list[float] = [
        float(obs.get("tick", 0) or 0) / float(TICK_LIMIT),
        max(0.0, float(TICK_LIMIT - int(obs.get("tick", 0) or 0))) / float(TICK_LIMIT),
        max(0.0, float(DEATH_TIME_START_TICK - int(obs.get("tick", 0) or 0))) / float(TICK_LIMIT),
        1.0 if int(obs.get("tick", 0) or 0) >= DEATH_TIME_START_TICK else 0.0,
        float(obs.get("resources", 0.0) or 0.0) / 100.0,
        own_resources / 300.0,
        enemy_resources / 300.0,
        1.0 if team_of(player_id) == "TEAM_A" else 0.0,
        1.0 if team_of(player_id) == "TEAM_B" else 0.0,
        own_alive / 3.0,
        enemy_alive / 3.0,
        baron_hp / float(BARON_HP),
        float(len(obs.get("moving_groups", []) or [])) / 40.0,
        float(len(obs.get("active_battles", []) or [])) / 10.0,
    ]

    moving_totals = _moving_node_totals(obs, player_id)
    battle_totals = _battle_node_totals(obs, player_id)
    edge_battle_totals = _edge_battle_node_totals(obs, player_id)
    own_team = team_of(player_id)

    for node_id in ALL_NODES:
        node = obs["nodes"][node_id]
        node_type = NODE_TYPES[node_id]
        route_reachable, route_edges, route_ticks = route_metrics_for_node(obs, player_id, node_id)
        features.extend([
            1.0 if node_type == NodeType.CORE else 0.0,
            1.0 if node_type == NodeType.CONTROL_POINT else 0.0,
            1.0 if node_type == NodeType.BARON else 0.0,
        ])
        relation = owner_relation(node.get("owner", "NEUTRAL"), player_id)
        features.extend(1.0 if relation == item else 0.0 for item in RELATIONS)

        relation_units = {item: {unit_type.value: 0 for unit_type in UNIT_ORDER} for item in RELATIONS}
        owner = node.get("owner", "NEUTRAL")
        owner_rel = owner_relation(owner, player_id)
        for unit_type in UNIT_ORDER:
            relation_units[owner_rel][unit_type.value] += int(node.get("units", {}).get(unit_type.value, 0) or 0)
        for visitor_id, units in (node.get("visiting_units", {}) or {}).items():
            visitor_rel = owner_relation(visitor_id, player_id)
            for unit_type in UNIT_ORDER:
                relation_units[visitor_rel][unit_type.value] += int(units.get(unit_type.value, 0) or 0)
        for relation_name in RELATIONS:
            features.extend(_normal_unit_counts(relation_units[relation_name]))

        ready_tick = int((node.get("unit_ready_tick", {}) or {}).get(player_id, 0) or 0)
        ready_remaining = max(0, ready_tick - int(obs.get("tick", 0) or 0))
        aggro = set(node.get("neutral_aggro_teams", []) or [])
        features.extend([
            float(node.get("neutral_hp", 0) or 0) / 10.0,
            float(node.get("neutral_income", 0) or 0) / 10.0,
            float(node.get("core_hp", 0.0) or 0.0) / float(CORE_HP),
            route_reachable,
            route_edges,
            route_ticks,
            min(1.0, ready_remaining / 10.0),
            1.0 if own_team in aggro else 0.0,
            1.0 if any(team in aggro for team in {"TEAM_A", "TEAM_B"} - {own_team}) else 0.0,
        ])
        features.extend(moving_totals[node_id])
        features.extend(battle_totals[node_id])
        features.extend(edge_battle_totals[node_id])

    vector = np.asarray(features, dtype=np.float32)
    if vector.shape != (OBSERVATION_DIM,):
        raise ValueError(f"Observation shape changed: {vector.shape}, expected {(OBSERVATION_DIM,)}")
    return vector
