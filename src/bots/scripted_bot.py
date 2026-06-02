"""
Simple heuristic bot:
- Produce VANGUARD if < threshold units at core
- Once enough units accumulated, send toward nearest neutral/enemy CP
- If all CPs taken, push toward enemy core
"""
from collections import deque

from ..core.engine import Action
from ..core.constants import ActionType, UnitType, UNIT_COST, TEAM_A_PLAYERS, TEAM_B_PLAYERS
from ..core.map import ADJACENCY
from .base import Bot

PRODUCE_THRESHOLD = 5  # accumulate this many units before sending


class ScriptedBot(Bot):
    def act(self, obs: dict) -> Action:
        resources = obs["resources"]
        core_node = obs["core_node"]
        nodes = obs["nodes"]
        baron = obs.get("baron", {})
        player_id = self.player_id

        core_units = sum(nodes[core_node]["units"].values())

        # Step 1: accumulate units
        if core_units < PRODUCE_THRESHOLD:
            cost = UNIT_COST[UnitType.VANGUARD]
            if resources >= cost:
                return Action(action_type=ActionType.PRODUCE, unit_type=UnitType.VANGUARD)
            return Action(action_type=ActionType.HOLD)

        # Step 2: find a target node to march toward
        target = self._find_target(player_id, core_node, nodes, baron)
        if target is None:
            # Keep producing while waiting
            if resources >= UNIT_COST[UnitType.VANGUARD]:
                return Action(action_type=ActionType.PRODUCE, unit_type=UnitType.VANGUARD)
            return Action(action_type=ActionType.HOLD)

        # Step 3: find best node to send from (own nodes with most units)
        best_src = self._find_best_src(player_id, core_node, target, nodes, baron)
        if best_src is None:
            return Action(action_type=ActionType.HOLD)

        return Action(action_type=ActionType.MOVE, source=best_src, destination=target)

    def _find_target(self, player_id: str, core_node: str, nodes: dict, baron: dict | None = None) -> str | None:
        """Find nearest neutral CP or enemy node. BFS from core."""
        for node in self._accessible_targets(player_id, core_node, nodes, baron):
            owner = nodes[node]["owner"]
            if node == "Baron" or owner == "NEUTRAL" or not self._is_friendly_owner(player_id, owner):
                return node
        return None

    def _find_best_src(
        self,
        player_id: str,
        core_node: str,
        target: str,
        nodes: dict,
        baron: dict | None = None,
    ) -> str | None:
        """Find the own node with most units that can reach target."""
        best_node = None
        best_count = 0

        for node_id, node_data in nodes.items():
            if node_data["owner"] == player_id:
                count = sum(node_data["units"].values())
                if count > best_count and target in self._accessible_targets(player_id, node_id, nodes, baron):
                    best_count = count
                    best_node = node_id

        return best_node if best_count > 0 else None

    def _accessible_targets(
        self,
        player_id: str,
        src: str,
        nodes: dict,
        baron: dict | None = None,
    ) -> list[str]:
        baron = baron or {}
        targets = []
        seen_targets = set()
        visited = {src}
        queue = deque([src])

        while queue:
            node_id = queue.popleft()
            for neighbor in ADJACENCY.get(node_id, []):
                if neighbor != src and self._is_accessible_destination(player_id, neighbor, node_id, nodes, baron):
                    if neighbor not in seen_targets:
                        seen_targets.add(neighbor)
                        targets.append(neighbor)
                if neighbor not in visited and self._is_transit_node(player_id, neighbor, nodes, baron):
                    visited.add(neighbor)
                    queue.append(neighbor)

        return targets

    def _is_accessible_destination(
        self,
        player_id: str,
        node_id: str,
        via_node: str,
        nodes: dict,
        baron: dict,
    ) -> bool:
        owner = nodes[node_id]["owner"]
        if node_id == "Baron":
            return True
        if owner == "NEUTRAL" or self._is_friendly_owner(player_id, owner):
            return True
        return self._is_transit_node(player_id, via_node, nodes, baron)

    def _is_transit_node(self, player_id: str, node_id: str, nodes: dict, baron: dict) -> bool:
        if (
            node_id == "Baron"
            and nodes[node_id]["owner"] == "NEUTRAL"
            and int(baron.get("hp", 0) or 0) > 0
        ):
            return False
        owner = nodes[node_id]["owner"]
        if owner == "NEUTRAL":
            return True
        return self._is_friendly_owner(player_id, owner)

    def _is_friendly_owner(self, player_id: str, owner: str) -> bool:
        return owner == player_id or owner in self._get_friendly_players(player_id)

    def _next_hop(self, src: str, target: str, nodes: dict, player_id: str) -> str | None:
        """BFS to find next hop from src toward target."""
        if src == target:
            return None
        visited = {src}
        queue = deque([(src, [src])])
        while queue:
            node, path = queue.popleft()
            for neighbor in ADJACENCY.get(node, []):
                if neighbor == target:
                    return path[1] if len(path) > 1 else neighbor
                if neighbor not in visited:
                    # Can pass through own or neutral nodes
                    owner = nodes[neighbor]["owner"]
                    friendly = self._get_friendly_players(player_id)
                    if owner == "NEUTRAL" or owner == player_id or owner in friendly:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        # Can't find path through own territory — return direct neighbor
        for neighbor in ADJACENCY.get(src, []):
            return neighbor
        return None

    def _get_friendly_players(self, player_id: str) -> set:
        if player_id in TEAM_A_PLAYERS:
            return set(TEAM_A_PLAYERS) - {player_id}
        return set(TEAM_B_PLAYERS) - {player_id}
