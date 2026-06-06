import random
from collections import deque

from ..core.engine import Action
from ..core.constants import (
    ActionType,
    BuildingType,
    UnitType,
    BUILDING_COST,
    CORE_BUILDINGS,
    FACTION_UNIQUE_BUILDINGS,
    UNIT_COST,
    UNIT_FACTION,
    unit_tech_requirements,
)
from ..core.map import ADJACENCY, NODE_TYPES, NodeType
from .base import Bot

UNIT_TYPES = list(UNIT_COST)


class RandomBot(Bot):
    def __init__(self, player_id: str, team: str, seed: int = None):
        super().__init__(player_id, team)
        self.rng = random.Random(seed)

    def act(self, obs: dict) -> Action:
        resources = obs["resources"]
        core_node = obs["core_node"]
        nodes = obs["nodes"]
        baron = obs.get("baron", {})

        affordable = [ut for ut in UNIT_TYPES if UNIT_COST[ut] <= resources and self._unit_unlocked(obs, ut)]

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

        build_options = self._build_options(obs)
        if build_options:
            choices.append(ActionType.BUILD)

        action_type = self.rng.choice(choices)

        if action_type == ActionType.PRODUCE and affordable:
            return Action(action_type=ActionType.PRODUCE, unit_type=self.rng.choice(affordable))

        if action_type == ActionType.MOVE and movable_nodes:
            src, neighbors = self.rng.choice(movable_nodes)
            dst = self.rng.choice(neighbors)
            return Action(action_type=ActionType.MOVE, source=src, destination=dst)

        if action_type == ActionType.BUILD and build_options:
            node_id, building_type = self.rng.choice(build_options)
            return Action(action_type=ActionType.BUILD, destination=node_id, building_type=building_type)

        return Action(action_type=ActionType.HOLD)

    def _build_options(self, obs: dict) -> list[tuple[str, BuildingType]]:
        resources = float(obs.get("resources", 0.0) or 0.0)
        core_node = obs.get("core_node")
        options: list[tuple[str, BuildingType]] = []
        completed = {
            building.get("building_type")
            for node in (obs.get("nodes", {}) or {}).values()
            for building in node.get("buildings", []) or []
            if (
                building.get("owner") == self.player_id
                and building.get("complete")
                and float(building.get("hp", 0.0) or 0.0) > 0
            )
        }
        for node_id, node in (obs.get("nodes", {}) or {}).items():
            if node.get("owner") != self.player_id or int(node.get("neutral_hp", 0) or 0) > 0:
                continue
            existing = [
                building for building in node.get("buildings", []) or []
                if building.get("owner") == self.player_id
            ]
            existing_types = {building.get("building_type") for building in existing}
            node_type = NODE_TYPES[node_id]
            if node_type == NodeType.CONTROL_POINT and not existing:
                if resources >= BUILDING_COST[BuildingType.TURRET]:
                    options.append((node_id, BuildingType.TURRET))
            elif node_type == NodeType.CORE and node_id == core_node and len(existing) < 2:
                faction = obs.get("faction", "humans")
                unique_by_value = {
                    building.value: owner_faction
                    for owner_faction, building in FACTION_UNIQUE_BUILDINGS.items()
                }
                for building_type in CORE_BUILDINGS:
                    required_faction = unique_by_value.get(building_type.value)
                    if required_faction is not None and required_faction != faction:
                        continue
                    if building_type.value in existing_types:
                        continue
                    if building_type == BuildingType.TECH_2 and BuildingType.TECH_1.value not in completed:
                        continue
                    if resources >= BUILDING_COST[building_type]:
                        options.append((node_id, building_type))
        return options

    def _unit_unlocked(self, obs: dict, unit_type: UnitType) -> bool:
        if UNIT_FACTION.get(unit_type) != obs.get("faction", "humans"):
            return False
        completed = {
            building.get("building_type")
            for node in (obs.get("nodes", {}) or {}).values()
            for building in node.get("buildings", []) or []
            if (
                building.get("owner") == self.player_id
                and building.get("complete")
                and float(building.get("hp", 0.0) or 0.0) > 0
            )
        }
        return all(required.value in completed for required in unit_tech_requirements(unit_type))

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
                return True
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
