from .constants import UnitType, UNIT_COST
from .state import GameState, PlayerState


def apply_production(state: GameState, player_id: str, unit_type: UnitType) -> bool:
    """
    Attempt to produce one unit of unit_type for player_id at their core node.
    Returns True if production succeeded.
    """
    player = state.players[player_id]
    if not player.alive:
        return False

    cost = UNIT_COST[unit_type]
    if player.resources < cost:
        return False

    player.resources -= cost
    node = state.nodes[player.core_node]
    unit_val = unit_type.value
    node.units[unit_val] = node.units.get(unit_val, 0) + 1
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
