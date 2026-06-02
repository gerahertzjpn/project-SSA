import random
from collections import deque

from ..core.engine import Action
from ..core.constants import ActionType, UnitType, UNIT_COST
from ..core.map import ADJACENCY
from .base import Bot

UNIT_TYPES = list(UnitType)


class RandomBot(Bot):
    def __init__(self, player_id: str, team: str, seed: int = None):
        super().__init__(player_id, team)
        self.rng = random.Random(seed)

    def act(self, obs: dict) -> Action:
        resources = obs["resources"]
        core_node = obs["core_node"]
        nodes = obs["nodes"]
        baron = obs.get("baron", {})

        affordable = [ut for ut in UNIT_TYPES if UNIT_COST[ut] <= resources]

        choices = [ActionType.HOLD]
        if affordable:
            choices.append(ActionType.PRODUCE)

        # Check if we have units somewhere to move
        movable_nodes = []
        for node_id, node_data in nodes.items():
            if node_data["owner"] == self.player_id and sum(node_data["units"].values()) > 0:
                targets = self._accessible_targets(node_id, nodes, baron)
                if targets:
                    movable_nodes.append((node_id, targets))

        if movable_nodes:
            choices.append(ActionType.MOVE)

        action_type = self.rng.choice(choices)

        if action_type == ActionType.PRODUCE and affordable:
            return Action(action_type=ActionType.PRODUCE, unit_type=self.rng.choice(affordable))

        if action_type == ActionType.MOVE and movable_nodes:
            src, neighbors = self.rng.choice(movable_nodes)
            dst = self.rng.choice(neighbors)
            return Action(action_type=ActionType.MOVE, source=src, destination=dst)

        return Action(action_type=ActionType.HOLD)

    def _accessible_targets(self, src: str, nodes: dict, baron: dict | None = None) -> list[str]:
        baron = baron or {}

        def is_friendly(owner: str) -> bool:
            return owner.startswith(self.player_id[0])

        def is_living_lord(node_id: str) -> bool:
            return (
                node_id == "Baron"
                and nodes[node_id]["owner"] == "NEUTRAL"
                and int(baron.get("hp", 0) or 0) > 0
            )

        def is_transit(node_id: str) -> bool:
            if is_living_lord(node_id):
                return False
            owner = nodes[node_id]["owner"]
            if owner == "NEUTRAL":
                return int(nodes[node_id].get("neutral_hp", 0) or 0) <= 0
            return is_friendly(owner)

        def is_destination(node_id: str, via_node: str) -> bool:
            owner = nodes[node_id]["owner"]
            if node_id == "Baron":
                return True
            if owner == "NEUTRAL" or is_friendly(owner):
                return True
            return is_transit(via_node)

        targets = []
        seen_targets = set()
        visited = {src}
        queue = deque([src])

        while queue:
            node_id = queue.popleft()
            for neighbor in ADJACENCY.get(node_id, []):
                if neighbor != src and is_destination(neighbor, node_id) and neighbor not in seen_targets:
                    seen_targets.add(neighbor)
                    targets.append(neighbor)
                if neighbor not in visited and is_transit(neighbor):
                    visited.add(neighbor)
                    queue.append(neighbor)
        return targets
