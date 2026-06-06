from .constants import (
    BUILDING_BUILD_TICKS,
    BUILDING_COST,
    BUILDING_STATS,
    CORE_BUILDINGS,
    CONTROL_POINT_BUILDINGS,
    FACTION_UNIQUE_BUILDINGS,
    MAX_CONTROL_POINT_BUILDINGS,
    MAX_CORE_BUILDINGS,
    UNIT_COST,
    UNIT_FACTION,
    unit_tech_requirements,
    BuildingType,
    UnitType,
)
from .map import NODE_TYPES, NodeType
from .state import BuildingState, GameState, PlayerState


def _building_type(value: BuildingType | str) -> BuildingType:
    return value if isinstance(value, BuildingType) else BuildingType(str(value))


def completed_buildings(state: GameState, node_id: str, owner: str | None = None) -> list[BuildingState]:
    node = state.nodes[node_id]
    return [
        building
        for building in node.buildings
        if (
            building.is_complete(state.tick)
            and building.hp > 0
            and (owner is None or building.owner == owner)
        )
    ]


def player_has_completed_building(
    state: GameState,
    player_id: str,
    building_type: BuildingType | str,
) -> bool:
    target_type = _building_type(building_type).value
    for node in state.nodes.values():
        for building in node.buildings:
            if (
                building.owner == player_id
                and building.building_type == target_type
                and building.is_complete(state.tick)
                and building.hp > 0
            ):
                return True
    return False


def can_produce_unit(state: GameState, player_id: str, unit_type: UnitType) -> bool:
    player = state.players[player_id]
    if not player.alive:
        return False
    cost = UNIT_COST.get(unit_type)
    if cost is None or player.resources < cost:
        return False
    if UNIT_FACTION.get(unit_type) != player.faction:
        return False
    for required in unit_tech_requirements(unit_type):
        if not player_has_completed_building(state, player_id, required):
            return False
    return True


def apply_production(state: GameState, player_id: str, unit_type: UnitType) -> bool:
    """
    Attempt to produce one unit of unit_type for player_id at their core node.
    Returns True if production succeeded.
    """
    if not can_produce_unit(state, player_id, unit_type):
        return False

    player = state.players[player_id]
    cost = UNIT_COST[unit_type]
    player.resources -= cost
    node = state.nodes[player.core_node]
    unit_val = unit_type.value
    node.units[unit_val] = node.units.get(unit_val, 0) + 1
    return True


def can_build(state: GameState, player_id: str, node_id: str, building_type: BuildingType | str) -> bool:
    if player_id not in state.players or node_id not in state.nodes:
        return False
    player = state.players[player_id]
    if not player.alive:
        return False

    try:
        building_type = _building_type(building_type)
    except ValueError:
        return False

    cost = BUILDING_COST.get(building_type)
    if cost is None or player.resources < cost:
        return False
    unique_owner_faction = {
        building.value: faction
        for faction, building in FACTION_UNIQUE_BUILDINGS.items()
    }.get(building_type.value)
    if unique_owner_faction is not None and player.faction != unique_owner_faction:
        return False

    node = state.nodes[node_id]
    if node.owner != player_id or node.neutral_hp > 0:
        return False
    if f"node:{node_id}" in state.active_battles:
        return False

    node_type = NODE_TYPES[node_id]
    existing = [building for building in node.buildings if building.owner == player_id]
    existing_types = {building.building_type for building in existing}

    if node_type == NodeType.CONTROL_POINT:
        if building_type not in CONTROL_POINT_BUILDINGS:
            return False
        if len(existing) >= MAX_CONTROL_POINT_BUILDINGS:
            return False
        return building_type.value not in existing_types

    if node_type == NodeType.CORE and node_id == player.core_node:
        if building_type not in CORE_BUILDINGS:
            return False
        if len(existing) >= MAX_CORE_BUILDINGS:
            return False
        if building_type.value in existing_types:
            return False
        if (
            building_type == BuildingType.TECH_2
            and not player_has_completed_building(state, player_id, BuildingType.TECH_1)
        ):
            return False
        return True

    return False


def apply_build(state: GameState, player_id: str, node_id: str, building_type: BuildingType | str) -> bool:
    if not can_build(state, player_id, node_id, building_type):
        return False

    building_type = _building_type(building_type)
    player = state.players[player_id]
    player.resources -= BUILDING_COST[building_type]
    stats = BUILDING_STATS[building_type]
    state.nodes[node_id].buildings.append(
        BuildingState(
            building_type=building_type.value,
            owner=player_id,
            hp=float(stats["hp"]),
            max_hp=float(stats["hp"]),
            started_tick=state.tick,
            complete_tick=state.tick + int(BUILDING_BUILD_TICKS[building_type]),
            last_damage_tick=-1,
        )
    )
    return True


def collect_income(state: GameState) -> None:
    """Add passive income and CP-based income to each alive player."""
    from .constants import RESOURCE_PER_TICK

    for player in state.players.values():
        if not player.alive:
            continue
        player.resources += RESOURCE_PER_TICK

    # Per-player income from captured CPs
    for node_id, node in state.nodes.items():
        if node.owner == "NEUTRAL" or node.neutral_income == 0:
            continue
        if node.neutral_hp > 0:
            continue  # not captured yet
        owner = node.owner
        if owner in state.players and state.players[owner].alive:
            state.players[owner].resources += node.neutral_income

    for node in state.nodes.values():
        for building in node.buildings:
            if not building.is_complete(state.tick):
                continue
            if building.hp <= 0:
                continue
            if building.owner not in state.players or not state.players[building.owner].alive:
                continue
            stats = BUILDING_STATS.get(BuildingType(building.building_type), {})
            income = float(stats.get("income", 0.0) or 0.0)
            if income > 0:
                state.players[building.owner].resources += income
