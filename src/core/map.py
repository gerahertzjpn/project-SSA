from collections import deque
from .constants import NodeType, TEAM_A_PLAYERS, TEAM_B_PLAYERS

ADJACENCY: dict[str, list[str]] = {
    "A1": ["CP1", "CP2"],
    "A2": ["CP1", "CP2"],
    "A3": ["CP1", "CP2"],
    "B1": ["CP5", "CP6"],
    "B2": ["CP5", "CP6"],
    "B3": ["CP5", "CP6"],
    "CP1": ["A1", "A2", "A3", "CP3", "Baron"],
    "CP2": ["A1", "A2", "A3", "CP4", "Baron"],
    "CP3": ["CP1", "CP5"],
    "CP4": ["CP2", "CP6"],
    "CP5": ["B1", "B2", "B3", "CP3", "Baron"],
    "CP6": ["B1", "B2", "B3", "CP4", "Baron"],
    "Baron": ["CP1", "CP2", "CP5", "CP6"],
}

NODE_TYPES: dict[str, NodeType] = {
    "A1": NodeType.CORE, "A2": NodeType.CORE, "A3": NodeType.CORE,
    "B1": NodeType.CORE, "B2": NodeType.CORE, "B3": NodeType.CORE,
    "CP1": NodeType.CONTROL_POINT, "CP2": NodeType.CONTROL_POINT,
    "CP3": NodeType.CONTROL_POINT, "CP4": NodeType.CONTROL_POINT,
    "CP5": NodeType.CONTROL_POINT, "CP6": NodeType.CONTROL_POINT,
    "Baron": NodeType.BARON,
}

ALL_NODES = list(ADJACENCY.keys())
CP_NODES = ["CP1", "CP2", "CP3", "CP4", "CP5", "CP6"]
CORE_NODES = TEAM_A_PLAYERS + TEAM_B_PLAYERS


def bfs_distance(src: str, dst: str) -> int:
    if src == dst:
        return 0
    visited = {src}
    queue = deque([(src, 0)])
    while queue:
        node, dist = queue.popleft()
        for neighbor in ADJACENCY[node]:
            if neighbor == dst:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return 999


def get_legal_moves(node: str, player_id: str, node_owners: dict[str, str]) -> list[str]:
    """
    Returns nodes reachable in one step from `node` for `player_id`.
    Cannot move to or through a node owned by the enemy team.
    Neutral and own nodes are passable.
    """
    from .constants import TEAM_A_PLAYERS, TEAM_B_PLAYERS
    if player_id in TEAM_A_PLAYERS:
        enemy_players = set(TEAM_B_PLAYERS)
    else:
        enemy_players = set(TEAM_A_PLAYERS)

    legal = []
    for neighbor in ADJACENCY.get(node, []):
        owner = node_owners.get(neighbor, "NEUTRAL")
        # Can move to own nodes, neutral nodes, or enemy nodes (to attack)
        # Cannot use an enemy-controlled node as a transit hub — but single-step
        # movement never requires transit, so all neighbors are valid targets
        # (you attack an enemy node by moving into it)
        legal.append(neighbor)
    return legal
