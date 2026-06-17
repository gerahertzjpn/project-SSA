from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.core.constants import (
    BARON_HP,
    BUILDING_COST,
    BUILDING_STATS,
    CORE_BUILDINGS,
    CORE_HP,
    CONTROL_POINT_BUILDINGS,
    DEATH_TIME_START_TICK,
    FACTION_TYPES,
    FACTION_UNIQUE_BUILDINGS,
    MAX_CONTROL_POINT_BUILDINGS,
    MAX_CORE_BUILDINGS,
    PLAYER_IDS,
    SPECIAL_DEST_ALLY_LORD,
    SPECIAL_DEST_ENEMY_LORD,
    SPECIAL_LORD_DESTINATIONS,
    TEAM_A_PLAYERS,
    TEAM_B_PLAYERS,
    TICK_LIMIT,
    UNIT_COST,
    UNIT_FACTION,
    unit_tech_requirements,
    BuildingType,
    UnitType,
    ActionType,
)
from src.core.engine import Action
from src.core.map import ADJACENCY, ALL_NODES, NODE_TYPES, NodeType
from src.core.map2d import (
    MOVEMENT_MAP_UNITS_PER_TICK,
    WORLD_SIZE,
    approach_edge_length,
    node_enter_length,
    node_exit_length,
    node_transit_length,
)
from src.core.units import UNIT_STATS


OBSERVATION_VERSION = "train_obs_v4_orders"

UNIT_ORDER = (
    UnitType.VANGUARD,
    UnitType.RANGER,
    UnitType.SKIRMISHER,
    UnitType.RAIDER,
    UnitType.BMP,
    UnitType.SWARMLING,
    UnitType.SPITTER,
    UnitType.BRUTE,
    UnitType.RAVAGER,
    UnitType.BEHEMOTH,
    UnitType.WALKER,
    UnitType.LASER,
    UnitType.DRONE,
    UnitType.GUARDIAN_MECH,
    UnitType.SIEGE_PLATFORM,
)
BUILDING_ORDER = (
    BuildingType.TURRET,
    BuildingType.BOOSTER,
    BuildingType.POWER,
    BuildingType.LIFE,
    BuildingType.ECONOMY_COLLECTOR,
    BuildingType.TECH_1,
    BuildingType.TECH_2,
)
FACTIONS = tuple(FACTION_TYPES)
CORE_TYPES = ("standard",)
RELATIONS = ("self", "ally", "enemy", "neutral")
COMBAT_RELATIONS = ("self", "ally", "enemy")
MOVEMENT_BUCKETS = ("inside", "outgoing", "incoming")
MOVE_INTENTS = ("CAPTURE", "TRANSIT")

# --- LLM-manager order channel (see docs/cloudcode/LLM_MANAGER_ORDERS_SPEC.md) ---
# A per-match "game plan" the manager sets: team-level tempo + win_path, plus a
# per-position role. Only the order LABELS live in the observation. "Obedience"
# is NOT an obs input — it is baked into each net's weights via a per-net
# order-shaping weight during training (a recipe/genotype knob; see spec). So a
# loyal net and a maverick net are different trained policies, not the same net
# with a dial.
ORDER_TEMPO = ("turtle", "balanced", "aggressive", "all_in")
ORDER_WIN_PATH = ("economy", "core_rush", "baron_lord", "outlast")
ORDER_ROLE = ("carry_push", "anchor_defend", "tech_eco", "roam_objective", "support_flex")
# Neutral defaults: when no order is present the channel reads as "balanced /
# economy / support_flex", so an un-ordered episode is a well-defined baseline.
ORDER_TEMPO_DEFAULT = "balanced"
ORDER_WIN_PATH_DEFAULT = "economy"
ORDER_ROLE_DEFAULT = "support_flex"
ORDER_FEATURES = len(ORDER_TEMPO) + len(ORDER_WIN_PATH) + len(ORDER_ROLE)

# Coherent strategy presets — the coach picks ONE per team, instead of free
# tempo×win_path (which allowed nonsense like turtle+core_rush). Each preset is a
# self-consistent game plan = a (tempo, win_path) + per-position roles. Training
# samples ONLY these, and the UI/coach choose among these, so a net only ever sees
# coherent orders. The obs encoding stays tempo/win_path/role — presets just curate
# which combinations are valid (no obs-dim change).
STRATEGY_PRESETS = (
    {"id": "standard",  "label": "Стандарт",        "tempo": "balanced",   "win_path": "economy",    "roles": ("carry_push", "anchor_defend", "support_flex")},
    {"id": "core_rush", "label": "Раш ядра",         "tempo": "aggressive", "win_path": "core_rush",  "roles": ("carry_push", "carry_push", "roam_objective")},
    {"id": "all_in",    "label": "Олл-ин",           "tempo": "all_in",     "win_path": "core_rush",  "roles": ("carry_push", "carry_push", "carry_push")},
    {"id": "turtle",    "label": "Черепаха",         "tempo": "turtle",     "win_path": "outlast",    "roles": ("anchor_defend", "anchor_defend", "tech_eco")},
    {"id": "macro",     "label": "Макро-эконом",     "tempo": "turtle",     "win_path": "economy",    "roles": ("tech_eco", "anchor_defend", "carry_push")},
    {"id": "baron",     "label": "Контроль Барона",  "tempo": "aggressive", "win_path": "baron_lord", "roles": ("roam_objective", "carry_push", "anchor_defend")},
)
STRATEGY_PRESET_BY_ID = {p["id"]: p for p in STRATEGY_PRESETS}


def strategy_order_for_slot(strategy_id: str, slot_index: int = 0) -> dict:
    """The per-player order ({tempo, win_path, role}) for a strategy preset at a
    given team position (0/1/2). Unknown strategy -> the neutral default."""
    p = STRATEGY_PRESET_BY_ID.get(strategy_id)
    if not p:
        return {"tempo": ORDER_TEMPO_DEFAULT, "win_path": ORDER_WIN_PATH_DEFAULT,
                "role": ORDER_ROLE_DEFAULT}
    roles = p["roles"]
    return {"tempo": p["tempo"], "win_path": p["win_path"],
            "role": roles[slot_index % len(roles)], "strategy": strategy_id}

GLOBAL_FEATURES = 22 + len(UNIT_ORDER) + len(FACTIONS) + ORDER_FEATURES
NODE_TYPE_FEATURES = 3
CORE_TYPE_FEATURES = len(CORE_TYPES)
OWNER_RELATION_FEATURES = len(RELATIONS)
STATIONED_UNIT_FEATURES = len(RELATIONS) * len(UNIT_ORDER)
NODE_SCALAR_FEATURES = 11
MOVEMENT_FEATURES = len(MOVEMENT_BUCKETS) * len(COMBAT_RELATIONS) * len(UNIT_ORDER)
BATTLE_FEATURES = 4 + len(COMBAT_RELATIONS) * len(UNIT_ORDER)
BUILDING_FEATURES = len(RELATIONS) * len(BUILDING_ORDER) * 3
NODE_FEATURES = (
    NODE_TYPE_FEATURES
    + CORE_TYPE_FEATURES
    + OWNER_RELATION_FEATURES
    + STATIONED_UNIT_FEATURES
    + NODE_SCALAR_FEATURES
    + BUILDING_FEATURES
    + MOVEMENT_FEATURES
    + BATTLE_FEATURES
    + BATTLE_FEATURES
)
OBSERVATION_DIM = GLOBAL_FEATURES + len(ALL_NODES) * NODE_FEATURES
ROUTE_EDGE_NORM = max(1, len(ALL_NODES) - 1)
ROUTE_TICK_NORM = float(TICK_LIMIT)


@dataclass(frozen=True)
class ActionSpec:
    kind: str
    unit_type: Optional[UnitType] = None
    building_type: Optional[BuildingType] = None
    source: Optional[str] = None
    destination: Optional[str] = None
    intent: str = "CAPTURE"


MOVE_ACTION_SPECS: tuple[ActionSpec, ...] = (
    tuple(
        ActionSpec("MOVE", source=source_id, destination=destination_id, intent=intent)
        for intent in MOVE_INTENTS
        for source_id in ALL_NODES
        for destination_id in ALL_NODES
        if source_id != destination_id
    )
    + tuple(
        ActionSpec("MOVE", source=source_id, destination=SPECIAL_DEST_ALLY_LORD, intent="JOIN_LORD")
        for source_id in ALL_NODES
    )
    + tuple(
        ActionSpec("MOVE", source=source_id, destination=SPECIAL_DEST_ENEMY_LORD, intent="HUNT_LORD")
        for source_id in ALL_NODES
    )
)

BUILD_ACTION_SPECS: tuple[ActionSpec, ...] = tuple(
    ActionSpec("BUILD", building_type=building_type, destination=node_id)
    for node_id in ALL_NODES
    for building_type in BUILDING_ORDER
)

ACTION_SPECS: tuple[ActionSpec, ...] = (
    (ActionSpec("NONE"),)
    + tuple(ActionSpec("PRODUCE", unit_type=unit_type) for unit_type in UNIT_ORDER)
    + (ActionSpec("NONE"),)
    + MOVE_ACTION_SPECS
    + (ActionSpec("NONE"),)
    + BUILD_ACTION_SPECS
)
ACTION_DIM = len(ACTION_SPECS)
ACTION_HEADS = ("produce", "move", "build")
PRODUCE_HEAD_DIM = 1 + len(UNIT_ORDER)
MOVE_HEAD_DIM = 1 + len(MOVE_ACTION_SPECS)
BUILD_HEAD_DIM = 1 + len(BUILD_ACTION_SPECS)
ACTION_HEAD_DIMS = (PRODUCE_HEAD_DIM, MOVE_HEAD_DIM, BUILD_HEAD_DIM)
ACTION_SHAPE = (len(ACTION_HEAD_DIMS),)
_OBS_CACHE_KEY = "__training_spaces_cache"
PRODUCE_ACTION_START = 0
MOVE_ACTION_START = PRODUCE_HEAD_DIM
BUILD_ACTION_START = PRODUCE_HEAD_DIM + MOVE_HEAD_DIM
ACTION_HEAD_OFFSETS = (PRODUCE_ACTION_START, MOVE_ACTION_START, BUILD_ACTION_START)
ACTION_HEAD_SLICES = (
    slice(PRODUCE_ACTION_START, MOVE_ACTION_START),
    slice(MOVE_ACTION_START, BUILD_ACTION_START),
    slice(BUILD_ACTION_START, ACTION_DIM),
)

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
    (spec.kind, spec.unit_type, spec.building_type, spec.source, spec.destination, spec.intent): index
    for index, spec in enumerate(ACTION_SPECS)
}
_MOVE_LOCAL_INDEX_BY_SPEC = {
    (spec.source, spec.destination, spec.intent): index + 1
    for index, spec in enumerate(MOVE_ACTION_SPECS)
}
_BUILD_LOCAL_INDEX_BY_SPEC = {
    (spec.building_type, spec.destination): index + 1
    for index, spec in enumerate(BUILD_ACTION_SPECS)
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
    if str(node_id) in SPECIAL_LORD_DESTINATIONS:
        return str(node_id)
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


def flat_action_to_factorized(action_index: int) -> np.ndarray:
    action_index = int(action_index)
    action = np.zeros(len(ACTION_HEAD_DIMS), dtype=np.int64)
    if PRODUCE_ACTION_START <= action_index < MOVE_ACTION_START:
        action[0] = action_index - PRODUCE_ACTION_START
    elif MOVE_ACTION_START <= action_index < BUILD_ACTION_START:
        action[1] = action_index - MOVE_ACTION_START
    elif BUILD_ACTION_START <= action_index < ACTION_DIM:
        action[2] = action_index - BUILD_ACTION_START
    return action


def normalize_action(action_index) -> np.ndarray:
    if isinstance(action_index, np.ndarray):
        values = action_index.astype(np.int64, copy=False).reshape(-1)
    elif isinstance(action_index, (list, tuple)):
        values = np.asarray(action_index, dtype=np.int64).reshape(-1)
    else:
        return flat_action_to_factorized(int(action_index))
    action = np.zeros(len(ACTION_HEAD_DIMS), dtype=np.int64)
    limit = min(len(action), len(values))
    if limit > 0:
        action[:limit] = values[:limit]
    return action


def flatten_factorized_action(action_index) -> tuple[int, int, int]:
    action = normalize_action(action_index)
    return tuple(int(value) for value in action)


def split_action_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_array = np.asarray(mask, dtype=np.bool_)
    return tuple(mask_array[action_slice] for action_slice in ACTION_HEAD_SLICES)


def action_is_legal(action_index, mask: np.ndarray) -> bool:
    if not isinstance(action_index, (np.ndarray, list, tuple)):
        try:
            scalar_index = int(action_index)
        except (TypeError, ValueError):
            return False
        if scalar_index < 0 or scalar_index >= ACTION_DIM:
            return False
    action = normalize_action(action_index)
    mask_array = np.asarray(mask, dtype=np.bool_)
    if mask_array.shape != (ACTION_DIM,):
        return False
    for local_index, offset, head_dim in zip(action, ACTION_HEAD_OFFSETS, ACTION_HEAD_DIMS):
        local = int(local_index)
        if local < 0 or local >= head_dim:
            return False
        if not bool(mask_array[offset + local]):
            return False
    return True


def sample_legal_action(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask_array = np.asarray(mask, dtype=np.bool_)
    action = np.zeros(len(ACTION_HEAD_DIMS), dtype=np.int64)
    for head_index, action_slice in enumerate(ACTION_HEAD_SLICES):
        legal = np.flatnonzero(mask_array[action_slice])
        action[head_index] = int(rng.choice(legal)) if len(legal) else 0
    return action


def mirror_action_index(action_index) -> np.ndarray:
    action = normalize_action(action_index)
    mirrored = np.array(action, dtype=np.int64, copy=True)

    move_local = int(action[1])
    if 0 < move_local <= len(MOVE_ACTION_SPECS):
        spec = MOVE_ACTION_SPECS[move_local - 1]
        mirrored[1] = _MOVE_LOCAL_INDEX_BY_SPEC.get(
            (mirrored_node_id(spec.source), mirrored_node_id(spec.destination), spec.intent),
            0,
        )

    build_local = int(action[2])
    if 0 < build_local <= len(BUILD_ACTION_SPECS):
        spec = BUILD_ACTION_SPECS[build_local - 1]
        mirrored[2] = _BUILD_LOCAL_INDEX_BY_SPEC.get(
            (spec.building_type, mirrored_node_id(spec.destination)),
            0,
        )

    return mirrored


def mirror_observation(obs: dict) -> dict:
    mirrored = dict(obs)
    mirrored.pop(_OBS_CACHE_KEY, None)
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
    if isinstance(obs.get("order"), dict):
        # Order values are board-agnostic labels (tempo/win_path/role) —
        # only the per-player keys need remapping under the mirror.
        mirrored["order"] = {
            mirrored_player_id(player_id): dict(entry)
            for player_id, entry in obs["order"].items()
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


def _observation_cache(obs: dict, player_id: str) -> dict:
    root = obs.setdefault(_OBS_CACHE_KEY, {})
    return root.setdefault(player_id, {})


def _mirror_node(node: dict) -> dict:
    mirrored = dict(node)
    mirrored["owner"] = mirrored_player_id(node.get("owner"))
    mirrored["buildings"] = [
        _mirror_building(building)
        for building in node.get("buildings", []) or []
    ]
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


def _mirror_building(building: dict) -> dict:
    mirrored = dict(building)
    mirrored["owner"] = mirrored_player_id(building.get("owner"))
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
    if isinstance(baron.get("allied_lord_path"), list):
        mirrored["allied_lord_path"] = [
            mirrored_node_id(node_id)
            for node_id in baron.get("allied_lord_path", [])
        ]
    if isinstance(baron.get("allied_lord_support_units"), dict):
        mirrored["allied_lord_support_units"] = dict(baron.get("allied_lord_support_units") or {})
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
    if len(mirrored) > 8:
        try:
            mirrored[8] = round(float(WORLD_SIZE[0]) - float(mirrored[8]), 2)
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


def _dead_core_node_obs(obs: dict, node_id: str) -> bool:
    if NODE_TYPES.get(node_id) != NodeType.CORE:
        return False
    player = (obs.get("players", {}) or {}).get(node_id)
    return bool(player) and (
        not player.get("alive", True)
        or float(player.get("core_hp", 0.0) or 0.0) <= 0.0
    )


def _is_transit_node(player_id: str, node_id: str, nodes: dict, baron: dict | None) -> bool:
    if _living_neutral_lord(node_id, nodes, baron):
        return False
    if NODE_TYPES.get(node_id) == NodeType.CORE and nodes.get(node_id, {}).get("owner") == "NEUTRAL":
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
    if NODE_TYPES.get(node_id) == NodeType.CORE and nodes.get(node_id, {}).get("owner") == "NEUTRAL":
        return False
    owner = nodes[node_id]["owner"]
    if owner == "NEUTRAL" or _friendly_owner(owner, player_id):
        return True
    return _is_transit_node(player_id, via_node, nodes, baron)


def find_route_obs(player_id: str, src: str, dst: str, obs: dict) -> Optional[list[str]]:
    nodes = obs["nodes"]
    baron = obs.get("baron", {})
    if src == dst or src not in nodes or dst not in nodes:
        return None
    routes = _observation_cache(obs, player_id).setdefault("routes", {})
    route_key = (src, dst)
    if route_key in routes:
        cached = routes[route_key]
        return list(cached) if cached else None

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
                    routes[route_key] = list(next_path)
                    return next_path
            if _is_transit_node(player_id, neighbor, nodes, baron):
                visited.add(neighbor)
                queue.append(next_path)
    routes[route_key] = None
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
    segment_speed = max(0.001, float(speed))
    total = 0
    for index, node_id in enumerate(path):
        if index == 0:
            total += _discrete_segment_ticks(_movement_ticks_for_distance(node_exit_length(node_id), segment_speed))
            continue

        prev_node = path[index - 1]
        total += _discrete_segment_ticks(
            _movement_ticks_for_distance(approach_edge_length(prev_node, node_id), segment_speed)
        )

        if index == len(path) - 1:
            total += _discrete_segment_ticks(_movement_ticks_for_distance(node_enter_length(node_id), segment_speed))
        else:
            total += _discrete_segment_ticks(
                _movement_ticks_for_distance(
                    node_transit_length(prev_node, node_id, path[index + 1]),
                    segment_speed,
                )
            )
    return float(total)


def _movement_ticks_for_distance(distance: float, speed: float) -> float:
    return float(distance) / (MOVEMENT_MAP_UNITS_PER_TICK * max(0.001, float(speed)))


def _discrete_segment_ticks(ticks: float) -> int:
    if ticks <= 0:
        return 0
    return max(1, math.ceil(float(ticks)))


def movable_sources(obs: dict, player_id: str) -> list[str]:
    cache = _observation_cache(obs, player_id)
    if "movable_sources" in cache:
        return list(cache["movable_sources"])
    tick = int(obs.get("tick", 0) or 0)
    sources = []
    for node_id in ALL_NODES:
        node = obs["nodes"][node_id]
        if int((node.get("unit_ready_tick", {}) or {}).get(player_id, 0) or 0) > tick:
            continue
        if unit_total(_source_units_for_player(node, player_id)) > 0:
            sources.append(node_id)
    cache["movable_sources"] = list(sources)
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
    metrics = _observation_cache(obs, player_id).setdefault("route_metrics", {})
    if destination in metrics:
        return tuple(metrics[destination])
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
        metrics[destination] = (0.0, 1.0, 1.0)
        return metrics[destination]

    candidates.sort()
    ticks, edges, _, _ = candidates[0]
    metrics[destination] = (
        1.0,
        min(1.0, edges / float(ROUTE_EDGE_NORM)),
        min(1.0, ticks / ROUTE_TICK_NORM),
    )
    return metrics[destination]


def special_lord_destination_node(
    obs: dict,
    player_id: str,
    destination: str | None,
    source: str | None = None,
) -> Optional[str]:
    if destination not in SPECIAL_LORD_DESTINATIONS:
        return destination if destination in ALL_NODES else None
    special_destinations = _observation_cache(obs, player_id).setdefault("special_lord_destinations", {})
    special_key = (destination, source)
    if special_key in special_destinations:
        return special_destinations[special_key]

    baron = obs.get("baron", {}) or {}
    if not baron.get("allied_lord_alive") or baron.get("allied_lord_expended"):
        special_destinations[special_key] = None
        return None
    owner = baron.get("allied_lord_owner")
    if not owner:
        special_destinations[special_key] = None
        return None
    same_team = team_of(owner) == team_of(player_id)
    if destination == SPECIAL_DEST_ALLY_LORD and not same_team:
        special_destinations[special_key] = None
        return None
    if destination == SPECIAL_DEST_ENEMY_LORD and same_team:
        special_destinations[special_key] = None
        return None

    battle_node = _special_lord_battle_node(obs, player_id, owner, source)
    if battle_node:
        special_destinations[special_key] = battle_node
        return battle_node

    group_node = _special_lord_group_node(obs, player_id, owner, source)
    if group_node:
        special_destinations[special_key] = group_node
        return group_node

    tick = int(obs.get("tick", 0) or 0)
    if int(baron.get("allied_lord_ready_tick", 0) or 0) > tick:
        special_destinations[special_key] = "Baron"
        return "Baron"

    for node_id, node in (obs.get("nodes", {}) or {}).items():
        if node_id not in ALL_NODES:
            continue
        if node.get("owner") == owner and int((node.get("units", {}) or {}).get(UnitType.LORD.value, 0) or 0) > 0:
            special_destinations[special_key] = node_id
            return node_id
        if int(((node.get("visiting_units", {}) or {}).get(owner, {}) or {}).get(UnitType.LORD.value, 0) or 0) > 0:
            special_destinations[special_key] = node_id
            return node_id
    special_destinations[special_key] = None
    return None


def _special_lord_battle_node(
    obs: dict,
    player_id: str,
    owner: str,
    source: str | None,
) -> Optional[str]:
    for battle in obs.get("active_battles", []) or []:
        side = (battle.get("sides", {}) or {}).get(owner, {})
        if int((side.get("units", {}) or {}).get(UnitType.LORD.value, 0) or 0) <= 0:
            continue
        if battle.get("location_kind") == "edge":
            endpoints = _edge_battle_endpoints(battle)
            return _best_accessible_target_obs(obs, player_id, source, endpoints)
        node_id = battle.get("location_id")
        return str(node_id) if node_id in ALL_NODES else None
    return None


def _special_lord_group_node(
    obs: dict,
    player_id: str,
    owner: str,
    source: str | None,
) -> Optional[str]:
    for group in obs.get("moving_groups", []) or []:
        if group.get("player") != owner:
            continue
        if int((group.get("units", {}) or {}).get(UnitType.LORD.value, 0) or 0) <= 0:
            continue
        phase = str(group.get("phase") or "")
        if phase in {"node_exit", "node_enter", "node_transit"}:
            node_id = group.get("segment_node") or group.get("dst") or group.get("src")
            return str(node_id) if node_id in ALL_NODES else None
        if phase == "edge":
            return _best_accessible_target_obs(
                obs,
                player_id,
                source,
                [group.get("dst"), group.get("src")],
            )
        node_id = group.get("dst")
        return str(node_id) if node_id in ALL_NODES else None
    return None


def _best_accessible_target_obs(
    obs: dict,
    player_id: str,
    source: str | None,
    candidates: list[str | None],
) -> Optional[str]:
    valid = [str(node_id) for node_id in candidates if node_id in ALL_NODES]
    if not valid:
        return None
    if not source or source not in ALL_NODES:
        return valid[0]
    scored = []
    for node_id in valid:
        route = find_route_obs(player_id, source, node_id, obs)
        if route:
            units = _source_units_for_player(obs["nodes"][source], player_id)
            scored.append((estimate_route_ticks(route, group_speed(units)), len(route), ALL_NODES.index(node_id), node_id))
    if scored:
        scored.sort()
        return scored[0][3]
    return valid[0]


def _intent_allowed(spec: ActionSpec, obs: dict, player_id: str) -> bool:
    if not _has_move(spec):
        return True
    destination = spec.destination
    if not destination:
        return False
    if destination in SPECIAL_LORD_DESTINATIONS:
        return True
    if _dead_core_node_obs(obs, destination):
        return False
    if spec.intent == "CAPTURE":
        return True
    nodes = obs["nodes"]
    if _living_neutral_lord(destination, nodes, obs.get("baron", {})):
        return False
    owner = nodes[destination]["owner"]
    return owner == "NEUTRAL" or _friendly_owner(owner, player_id)


def _has_move(spec: ActionSpec) -> bool:
    return spec.kind == "MOVE" and bool(spec.source and spec.destination)


def _move_allowed(
    spec: ActionSpec,
    obs: dict,
    player_id: str,
    movable_source_set: set[str] | None = None,
) -> bool:
    if not _has_move(spec):
        return True
    if not _intent_allowed(spec, obs, player_id):
        return False
    sources = movable_source_set if movable_source_set is not None else set(movable_sources(obs, player_id))
    if spec.source not in sources:
        return False
    destination = special_lord_destination_node(obs, player_id, spec.destination, spec.source)
    if destination is None:
        return False
    return find_route_obs(player_id, spec.source, destination, obs) is not None


def _completed_building_types_obs(obs: dict, player_id: str) -> set[str]:
    completed: set[str] = set()
    tick = int(obs.get("tick", 0) or 0)
    for node in (obs.get("nodes", {}) or {}).values():
        for building in node.get("buildings", []) or []:
            if building.get("owner") != player_id:
                continue
            complete = bool(building.get("complete"))
            complete_tick = building.get("complete_tick")
            if complete_tick is not None:
                complete = complete or tick >= int(complete_tick)
            if complete and float(building.get("hp", 0.0) or 0.0) > 0:
                completed.add(str(building.get("building_type")))
    return completed


def _produce_allowed_obs(obs: dict, player_id: str, unit_type: UnitType, resources: float) -> bool:
    if resources < float(UNIT_COST.get(unit_type, 10**9)):
        return False
    if UNIT_FACTION.get(unit_type) != obs.get("faction", "humans"):
        return False
    completed = _completed_building_types_obs(obs, player_id)
    return all(required.value in completed for required in unit_tech_requirements(unit_type))


def _build_allowed_obs(
    obs: dict,
    player_id: str,
    node_id: str,
    building_type: BuildingType,
    resources: float,
) -> bool:
    if resources < float(BUILDING_COST.get(building_type, 10**9)):
        return False
    unique_owner_faction = {
        building.value: faction
        for faction, building in FACTION_UNIQUE_BUILDINGS.items()
    }.get(building_type.value)
    if unique_owner_faction is not None and obs.get("faction", "humans") != unique_owner_faction:
        return False
    nodes = obs.get("nodes", {}) or {}
    node = nodes.get(node_id)
    if not node or node.get("owner") != player_id or int(node.get("neutral_hp", 0) or 0) > 0:
        return False
    if any(
        battle.get("location_kind") == "node" and battle.get("location_id") == node_id
        for battle in obs.get("active_battles", []) or []
    ):
        return False

    existing = [
        building
        for building in node.get("buildings", []) or []
        if building.get("owner") == player_id
    ]
    existing_types = {str(building.get("building_type")) for building in existing}
    node_type = NODE_TYPES[node_id]

    if node_type == NodeType.CONTROL_POINT:
        if building_type not in CONTROL_POINT_BUILDINGS:
            return False
        return len(existing) < MAX_CONTROL_POINT_BUILDINGS and building_type.value not in existing_types

    if node_type == NodeType.CORE and node_id == obs.get("core_node"):
        if building_type not in CORE_BUILDINGS:
            return False
        if len(existing) >= MAX_CORE_BUILDINGS or building_type.value in existing_types:
            return False
        if (
            building_type == BuildingType.TECH_2
            and BuildingType.TECH_1.value not in _completed_building_types_obs(obs, player_id)
        ):
            return False
        return True

    return False


def legal_action_mask(obs: dict, player_id: str) -> np.ndarray:
    policy_obs, policy_player = policy_observation(obs, player_id)
    return _legal_action_mask_absolute(policy_obs, policy_player)


def _legal_action_mask_absolute(obs: dict, player_id: str) -> np.ndarray:
    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    mask[PRODUCE_ACTION_START] = True
    mask[MOVE_ACTION_START] = True
    mask[BUILD_ACTION_START] = True

    player = (obs.get("players", {}) or {}).get(player_id)
    if player and (
        not player.get("alive", True)
        or float(player.get("core_hp", 0.0) or 0.0) <= 0.0
    ):
        return mask

    resources = float(obs.get("resources", 0.0) or 0.0)

    build_allowed: list[bool] = []
    build_costs: list[float] = []
    for spec in BUILD_ACTION_SPECS:
        allowed = bool(
            spec.building_type
            and spec.destination
            and _build_allowed_obs(
                obs,
                player_id,
                spec.destination,
                spec.building_type,
                resources,
            )
        )
        build_allowed.append(allowed)
        if allowed and spec.building_type:
            build_costs.append(float(BUILDING_COST[spec.building_type]))

    # Factorized heads are masked independently. To avoid legal
    # produce+build pairs that can overspend together, production reserves the
    # most expensive currently legal build. This is deliberately conservative
    # and keeps build decisions reliable for the policy.
    build_reserve = max(build_costs) if build_costs else 0.0
    produce_resources = max(0.0, resources - build_reserve)
    produce_allowed = [
        _produce_allowed_obs(obs, player_id, unit_type, produce_resources)
        for unit_type in UNIT_ORDER
    ]
    for unit_index, allowed in enumerate(produce_allowed):
        if allowed:
            mask[PRODUCE_ACTION_START + 1 + unit_index] = True

    movable_source_set = set(movable_sources(obs, player_id))
    move_allowed = [
        _move_allowed(spec, obs, player_id, movable_source_set)
        for spec in MOVE_ACTION_SPECS
    ]
    for move_index, allowed in enumerate(move_allowed):
        if not allowed:
            continue
        mask[MOVE_ACTION_START + 1 + move_index] = True
    for build_index, allowed in enumerate(build_allowed):
        if allowed:
            mask[BUILD_ACTION_START + 1 + build_index] = True
    return mask


def decode_action(action_index, obs: dict, player_id: str, assume_legal: bool = False) -> Action:
    policy_obs, policy_player = policy_observation(obs, player_id)
    return _decode_policy_action(action_index, policy_obs, policy_player, player_id, assume_legal=assume_legal)


def _decode_policy_action(
    action_index,
    policy_obs: dict,
    policy_player: str,
    actual_player: str,
    assume_legal: bool = False,
) -> Action:
    action = normalize_action(action_index)
    if not assume_legal:
        mask = _legal_action_mask_absolute(policy_obs, policy_player)
        if not action_is_legal(action, mask):
            return Action(ActionType.HOLD)

    produce_local, move_local, build_local = (int(value) for value in action)
    sub_actions: list[Action] = []

    if produce_local > 0:
        unit_index = produce_local - 1
        if unit_index < len(UNIT_ORDER):
            sub_actions.append(Action(ActionType.PRODUCE, unit_type=UNIT_ORDER[unit_index]))

    if move_local > 0:
        move_index = move_local - 1
        if move_index < len(MOVE_ACTION_SPECS):
            spec = MOVE_ACTION_SPECS[move_index]
            source = spec.source
            destination = spec.destination
            if uses_mirrored_policy_perspective(actual_player):
                source = mirrored_node_id(source)
                destination = mirrored_node_id(destination)
            sub_actions.append(Action(
                ActionType.MOVE,
                source=source,
                destination=destination,
                intent=spec.intent,
            ))

    if build_local > 0:
        build_index = build_local - 1
        if build_index < len(BUILD_ACTION_SPECS):
            spec = BUILD_ACTION_SPECS[build_index]
            destination = spec.destination
            if uses_mirrored_policy_perspective(actual_player):
                destination = mirrored_node_id(destination)
            sub_actions.append(Action(
                ActionType.BUILD,
                building_type=spec.building_type,
                destination=destination,
            ))

    if not sub_actions:
        return Action(ActionType.HOLD)
    if len(sub_actions) == 1:
        return sub_actions[0]
    return Action(ActionType.MULTI, sub_actions=sub_actions[:3])


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


def _lord_relation_features(obs: dict, player_id: str) -> list[float]:
    baron = obs.get("baron", {}) or {}
    if not baron.get("allied_lord_alive"):
        relation = "neutral"
    else:
        relation = owner_relation(baron.get("allied_lord_owner", "NEUTRAL"), player_id)
    return [1.0 if relation == item else 0.0 for item in RELATIONS]


def _lord_route_flags(obs: dict) -> tuple[set[str], set[str]]:
    baron = obs.get("baron", {}) or {}
    if not baron.get("allied_lord_alive"):
        return set(), set()
    path = {
        str(node_id)
        for node_id in (baron.get("allied_lord_path", []) or [])
        if str(node_id) in ALL_NODES
    }
    target = str(baron.get("allied_lord_target") or "")
    targets = {target} if target in ALL_NODES else set()
    return path, targets


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
        if battle.get("location_kind") not in {"node", "approach"}:
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


def _building_features(node: dict, player_id: str) -> list[float]:
    features = [0.0] * BUILDING_FEATURES
    for building in node.get("buildings", []) or []:
        try:
            building_type = BuildingType(str(building.get("building_type")))
        except ValueError:
            continue
        relation = owner_relation(building.get("owner", "NEUTRAL"), player_id)
        if relation not in RELATIONS:
            continue
        relation_index_ = RELATIONS.index(relation)
        building_index = BUILDING_ORDER.index(building_type)
        offset = (relation_index_ * len(BUILDING_ORDER) + building_index) * 3
        max_hp = max(1.0, float(building.get("max_hp", BUILDING_STATS[building_type]["hp"]) or 1.0))
        hp = max(0.0, float(building.get("hp", 0.0) or 0.0))
        progress = float(building.get("progress", 1.0 if building.get("complete") else 0.0) or 0.0)
        features[offset] += 1.0 / 2.0
        features[offset + 1] += min(1.0, hp / max_hp)
        features[offset + 2] += max(0.0, min(1.0, progress))
    return features


def _player_order(obs: dict, player_id: str) -> dict:
    """The manager order for one position, with neutral defaults.

    obs["order"] is keyed by player_id: {pid: {tempo, win_path, role}}.
    Missing entries / fields fall back to the neutral baseline so an un-ordered
    episode is well-defined (channel reads balanced/economy/support_flex).
    """
    order = obs.get("order") or {}
    entry = order.get(player_id, {}) if isinstance(order, dict) else {}
    return entry if isinstance(entry, dict) else {}


def _order_features(obs: dict, player_id: str) -> list[float]:
    entry = _player_order(obs, player_id)
    # Explicit "no order" -> all-zeros: the channel is masked off and the net plays
    # freely (distinct from any specific order, which is one-hot). Pass
    # {"absent": True} for a player to mask the order. Missing/empty entries keep
    # the neutral default (back-compat with un-ordered callers).
    if entry.get("absent"):
        return [0.0] * ORDER_FEATURES
    tempo = entry.get("tempo", ORDER_TEMPO_DEFAULT)
    win_path = entry.get("win_path", ORDER_WIN_PATH_DEFAULT)
    role = entry.get("role", ORDER_ROLE_DEFAULT)
    feats: list[float] = []
    feats.extend(1.0 if tempo == v else 0.0 for v in ORDER_TEMPO)
    feats.extend(1.0 if win_path == v else 0.0 for v in ORDER_WIN_PATH)
    feats.extend(1.0 if role == v else 0.0 for v in ORDER_ROLE)
    return feats


def encode_observation(obs: dict, player_id: str) -> np.ndarray:
    policy_obs, policy_player = policy_observation(obs, player_id)
    return _encode_observation_absolute(policy_obs, policy_player)


def _encode_observation_absolute(obs: dict, player_id: str) -> np.ndarray:
    own_resources, enemy_resources = _team_resource_totals(obs, player_id)
    own_alive, enemy_alive = _team_alive_counts(obs, player_id)
    baron = obs.get("baron", {}) or {}
    tick = int(obs.get("tick", 0) or 0)
    baron_hp = float(baron.get("hp", 0.0) or 0.0)
    allied_lord_hp = float(baron.get("allied_lord_hp", 0.0) or 0.0)
    allied_lord_ready_in = max(0, int(baron.get("allied_lord_ready_tick", 0) or 0) - tick)
    allied_lord_eta = max(0, int(baron.get("allied_lord_arrival_tick", 0) or 0) - tick)
    allied_lord_support_units = baron.get("allied_lord_support_units", {}) or {}

    features: list[float] = [
        float(tick) / float(TICK_LIMIT),
        max(0.0, float(TICK_LIMIT - tick)) / float(TICK_LIMIT),
        max(0.0, float(DEATH_TIME_START_TICK - tick)) / float(TICK_LIMIT),
        1.0 if tick >= DEATH_TIME_START_TICK else 0.0,
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
        1.0 if baron.get("allied_lord_alive") else 0.0,
        allied_lord_hp / float(BARON_HP),
        min(1.0, allied_lord_ready_in / float(TICK_LIMIT)),
        min(1.0, allied_lord_eta / float(TICK_LIMIT)),
        *_lord_relation_features(obs, player_id),
        *_normal_unit_counts(allied_lord_support_units),
        *(1.0 if obs.get("faction", "humans") == faction else 0.0 for faction in FACTIONS),
        *_order_features(obs, player_id),
    ]

    moving_totals = _moving_node_totals(obs, player_id)
    battle_totals = _battle_node_totals(obs, player_id)
    edge_battle_totals = _edge_battle_node_totals(obs, player_id)
    own_team = team_of(player_id)
    lord_path_nodes, lord_target_nodes = _lord_route_flags(obs)

    for node_id in ALL_NODES:
        node = obs["nodes"][node_id]
        node_type = NODE_TYPES[node_id]
        route_reachable, route_edges, route_ticks = route_metrics_for_node(obs, player_id, node_id)
        features.extend([
            1.0 if node_type == NodeType.CORE else 0.0,
            1.0 if node_type == NodeType.CONTROL_POINT else 0.0,
            1.0 if node_type == NodeType.BARON else 0.0,
        ])
        core_type = node.get("core_type") if node_type == NodeType.CORE else None
        features.extend(
            1.0 if core_type == item else 0.0
            for item in CORE_TYPES
        )
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
            1.0 if node_id in lord_path_nodes else 0.0,
            1.0 if node_id in lord_target_nodes else 0.0,
        ])
        features.extend(_building_features(node, player_id))
        features.extend(moving_totals[node_id])
        features.extend(battle_totals[node_id])
        features.extend(edge_battle_totals[node_id])

    vector = np.asarray(features, dtype=np.float32)
    if vector.shape != (OBSERVATION_DIM,):
        raise ValueError(f"Observation shape changed: {vector.shape}, expected {(OBSERVATION_DIM,)}")
    return vector
