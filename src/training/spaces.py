from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.core.constants import (
    BARON_HP,
    PLAYER_IDS,
    TEAM_A_PLAYERS,
    TEAM_B_PLAYERS,
    TICK_LIMIT,
    UNIT_COST,
    UnitType,
    ActionType,
)
from src.core.engine import Action
from src.core.map import ADJACENCY, ALL_NODES, NODE_TYPES, NodeType


OBSERVATION_VERSION = "train_obs_v0.1"

UNIT_ORDER = (UnitType.VANGUARD, UnitType.RANGER, UnitType.SKIRMISHER)
RELATIONS = ("self", "ally", "enemy", "neutral")
MOVE_INTENTS = ("CAPTURE", "TRANSIT")
GLOBAL_FEATURES = 11
NODE_FEATURES = 37
OBSERVATION_DIM = GLOBAL_FEATURES + len(ALL_NODES) * NODE_FEATURES


@dataclass(frozen=True)
class ActionSpec:
    kind: str
    unit_type: Optional[UnitType] = None
    destination: Optional[str] = None
    intent: str = "CAPTURE"


ACTION_SPECS: tuple[ActionSpec, ...] = (
    (ActionSpec("HOLD"),)
    + tuple(ActionSpec("PRODUCE", unit_type=unit_type) for unit_type in UNIT_ORDER)
    + tuple(
        ActionSpec("MOVE", destination=node_id, intent=intent)
        for intent in MOVE_INTENTS
        for node_id in ALL_NODES
    )
)
ACTION_DIM = len(ACTION_SPECS)


def team_of(player_id: str) -> str:
    if player_id in TEAM_A_PLAYERS:
        return "TEAM_A"
    if player_id in TEAM_B_PLAYERS:
        return "TEAM_B"
    return "NEUTRAL"


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
    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    resources = float(obs.get("resources", 0.0) or 0.0)
    mask[0] = True

    for index, spec in enumerate(ACTION_SPECS):
        if spec.kind == "PRODUCE" and spec.unit_type is not None:
            mask[index] = resources >= UNIT_COST[spec.unit_type]
        elif spec.kind == "MOVE" and spec.destination:
            if not _intent_allowed(spec, obs, player_id):
                continue
            mask[index] = select_source_for_destination(obs, player_id, spec.destination) is not None
    return mask


def decode_action(action_index: int, obs: dict, player_id: str) -> Action:
    if action_index < 0 or action_index >= ACTION_DIM:
        return Action(ActionType.HOLD)

    spec = ACTION_SPECS[int(action_index)]
    if spec.kind == "HOLD":
        return Action(ActionType.HOLD)
    if spec.kind == "PRODUCE" and spec.unit_type is not None:
        if legal_action_mask(obs, player_id)[action_index]:
            return Action(ActionType.PRODUCE, unit_type=spec.unit_type)
        return Action(ActionType.HOLD)
    if spec.kind == "MOVE" and spec.destination:
        if not legal_action_mask(obs, player_id)[action_index]:
            return Action(ActionType.HOLD)
        source = select_source_for_destination(obs, player_id, spec.destination)
        if source is None:
            return Action(ActionType.HOLD)
        return Action(
            ActionType.MOVE,
            source=source,
            destination=spec.destination,
            intent=spec.intent,
        )
    return Action(ActionType.HOLD)


def _normal_unit_counts(units: dict | None) -> list[float]:
    return [float((units or {}).get(unit_type.value, 0) or 0) / 50.0 for unit_type in UNIT_ORDER]


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
    totals = {node_id: [0.0] * 9 for node_id in ALL_NODES}
    for group in obs.get("moving_groups", []) or []:
        relation = owner_relation(group.get("player", "NEUTRAL"), player_id)
        if relation == "neutral":
            continue
        rel_slot = {"self": 0, "ally": 1, "enemy": 2}[relation]
        total = float(unit_total(group.get("units", {}))) / 50.0
        phase = group.get("phase", "edge")
        inside_node = group.get("segment_node") if phase != "edge" else None
        if inside_node in totals:
            totals[inside_node][rel_slot] += total
        if group.get("src") in totals:
            totals[group["src"]][3 + rel_slot] += total
        if group.get("dst") in totals:
            totals[group["dst"]][6 + rel_slot] += total
    return totals


def _battle_node_totals(obs: dict, player_id: str) -> dict[str, list[float]]:
    totals = {node_id: [0.0, 0.0, 0.0, 0.0] for node_id in ALL_NODES}
    for battle in obs.get("active_battles", []) or []:
        if battle.get("location_kind") != "node":
            continue
        node_id = battle.get("location_id")
        if node_id not in totals:
            continue
        friendly_hp = 0.0
        enemy_hp = 0.0
        neutral_hp = 0.0
        for side_id, side in (battle.get("sides", {}) or {}).items():
            hp = float(side.get("hp", 0.0) or 0.0)
            relation = owner_relation(side_id, player_id)
            if relation in {"self", "ally"}:
                friendly_hp += hp
            elif relation == "enemy":
                enemy_hp += hp
            else:
                neutral_hp += hp
        totals[node_id] = [
            friendly_hp / 200.0,
            enemy_hp / 200.0,
            neutral_hp / 100.0,
            float(battle.get("age_ticks", 0) or 0) / float(TICK_LIMIT),
        ]
    return totals


def encode_observation(obs: dict, player_id: str) -> np.ndarray:
    own_resources, enemy_resources = _team_resource_totals(obs, player_id)
    own_alive, enemy_alive = _team_alive_counts(obs, player_id)
    baron_hp = float((obs.get("baron", {}) or {}).get("hp", 0.0) or 0.0)

    features: list[float] = [
        float(obs.get("tick", 0) or 0) / float(TICK_LIMIT),
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
    own_team = team_of(player_id)

    for node_id in ALL_NODES:
        node = obs["nodes"][node_id]
        node_type = NODE_TYPES[node_id]
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
            min(1.0, ready_remaining / 10.0),
            1.0 if own_team in aggro else 0.0,
            1.0 if any(team in aggro for team in {"TEAM_A", "TEAM_B"} - {own_team}) else 0.0,
        ])
        features.extend(moving_totals[node_id])
        features.extend(battle_totals[node_id])

    vector = np.asarray(features, dtype=np.float32)
    if vector.shape != (OBSERVATION_DIM,):
        raise ValueError(f"Observation shape changed: {vector.shape}, expected {(OBSERVATION_DIM,)}")
    return vector
