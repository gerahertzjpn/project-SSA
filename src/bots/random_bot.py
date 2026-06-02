import random
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

        affordable = [ut for ut in UNIT_TYPES if UNIT_COST[ut] <= resources]

        choices = [ActionType.HOLD]
        if affordable:
            choices.append(ActionType.PRODUCE)

        # Check if we have units somewhere to move
        movable_nodes = []
        for node_id, node_data in nodes.items():
            if node_data["owner"] == self.player_id and sum(node_data["units"].values()) > 0:
                targets = self._accessible_targets(node_id, nodes)
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

    def _accessible_targets(self, src: str, nodes: dict) -> list[str]:
        def is_transit(node_id: str) -> bool:
            owner = nodes[node_id]["owner"]
            return owner == "NEUTRAL" or owner.startswith(self.player_id[0])

        targets = []
        for node_id, node_data in nodes.items():
            if node_id == src:
                continue
            owner = node_data["owner"]
            if node_id == "Baron" or owner == "NEUTRAL" or owner.startswith(self.player_id[0]):
                targets.append(node_id)
                continue
            if any(is_transit(neighbor) for neighbor in ADJACENCY.get(node_id, [])):
                targets.append(node_id)
        return targets
