"""
Simple heuristic bot:
- Produce VANGUARD if < threshold units at core
- Once enough units accumulated, send toward nearest neutral/enemy CP
- If all CPs taken, push toward enemy core
"""
from collections import deque

from ..core.engine import Action
from ..core.constants import (
    ActionType,
    BASIC_UNITS_BY_FACTION,
    BuildingType,
    TECH1_UNITS_BY_FACTION,
    TECH2_UNITS_BY_FACTION,
    UnitType,
    BUILDING_COST,
    UNIT_COST,
    UNIT_FACTION,
    unit_tech_requirements,
    TEAM_A_PLAYERS,
    TEAM_B_PLAYERS,
)
from ..core.map import ADJACENCY, NODE_TYPES, NodeType
from .base import Bot

PRODUCE_THRESHOLD = 5  # accumulate this many units before sending


class ScriptedBot(Bot):
    def __init__(
        self,
        player_id: str,
        team: str,
        *,
        core_focus: bool = False,
        build_plan: tuple[BuildingType, ...] | None = None,
        production_priority: tuple[UnitType, ...] | None = None,
        cp_turrets: bool = True,
        save_for_first_building: bool = True,
        produce_threshold: int = PRODUCE_THRESHOLD,
    ):
        super().__init__(player_id, team)
        self.core_focus = core_focus
        self.build_plan = (
            (BuildingType.TECH_1, BuildingType.TECH_2)
            if build_plan is None else tuple(build_plan)
        )
        self.production_priority = tuple(production_priority or ())
        self.cp_turrets = cp_turrets
        self.save_for_first_building = save_for_first_building
        self.produce_threshold = int(produce_threshold)

    def act(self, obs: dict) -> Action:
        core_node = obs["core_node"]
        nodes = obs["nodes"]
        baron = obs.get("baron", {})
        player_id = self.player_id

        steps: list[Action] = []

        move_action = self._move_action(obs, player_id, core_node, nodes, baron)
        if move_action is not None:
            steps.append(move_action)

        defer_opening_build = self._defer_opening_build(obs)
        build_action = None if defer_opening_build else self._build_action(obs)
        build_reserve = self._action_cost(build_action)
        if build_action is not None:
            steps.append(build_action)

        if defer_opening_build or not self._saving_for_first_building(obs):
            produce_action = self._production_action(obs, reserve=build_reserve)
            if produce_action is not None:
                steps.append(produce_action)

        if not steps:
            return Action(action_type=ActionType.HOLD)
        if len(steps) == 1:
            return steps[0]
        return Action(action_type=ActionType.MULTI, sub_actions=steps[:3])

    def _move_action(
        self,
        obs: dict,
        player_id: str,
        core_node: str,
        nodes: dict,
        baron: dict | None = None,
    ) -> Action | None:
        target = self._find_target(player_id, core_node, nodes, baron)
        if target is None:
            return None

        best_src = self._find_best_src(player_id, core_node, target, nodes, baron)
        if best_src is None:
            return None

        source_units = sum(int(value or 0) for value in nodes[best_src].get("units", {}).values())
        if source_units < self._move_threshold(obs):
            return None

        return Action(action_type=ActionType.MOVE, source=best_src, destination=target)

    def _move_threshold(self, obs: dict) -> int:
        faction = obs.get("faction", "humans")
        if faction == "robots":
            return max(2, min(self.produce_threshold, 3))
        return self.produce_threshold

    def _defer_opening_build(self, obs: dict) -> bool:
        if obs.get("faction", "humans") != "robots" or not self.build_plan:
            return False
        return self._owned_unit_count(obs) < self._move_threshold(obs)

    def _owned_unit_count(self, obs: dict) -> int:
        total = 0
        for node in (obs.get("nodes", {}) or {}).values():
            if node.get("owner") == self.player_id:
                total += sum(int(value or 0) for value in (node.get("units", {}) or {}).values())
            total += sum(
                int(value or 0)
                for value in ((node.get("visiting_units", {}) or {}).get(self.player_id, {}) or {}).values()
            )
        for group in obs.get("moving_groups", []) or []:
            if group.get("player_id") == self.player_id:
                total += sum(int(value or 0) for value in (group.get("units", {}) or {}).values())
        return total

    def _action_cost(self, action: Action | None) -> float:
        if action is None:
            return 0.0
        if action.action_type == ActionType.BUILD and action.building_type is not None:
            return float(BUILDING_COST[action.building_type])
        if action.action_type == ActionType.PRODUCE and action.unit_type is not None:
            return float(UNIT_COST[action.unit_type])
        return 0.0

    def _find_target(self, player_id: str, core_node: str, nodes: dict, baron: dict | None = None) -> str | None:
        """Find nearest neutral CP or enemy node. BFS from core."""
        targets = self._accessible_targets(player_id, core_node, nodes, baron)
        if self.core_focus:
            for node in targets:
                owner = nodes[node]["owner"]
                if NODE_TYPES[node] == NodeType.CORE and owner != "NEUTRAL" and not self._is_friendly_owner(player_id, owner):
                    return node
            for node in targets:
                owner = nodes[node]["owner"]
                if owner != "NEUTRAL" and not self._is_friendly_owner(player_id, owner):
                    return node
        for node in targets:
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
        if NODE_TYPES[node_id] == NodeType.CORE and owner == "NEUTRAL":
            return False
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
        if NODE_TYPES[node_id] == NodeType.CORE and nodes[node_id]["owner"] == "NEUTRAL":
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

    def _build_action(self, obs: dict, reserve: float = 0.0) -> Action | None:
        resources = float(obs.get("resources", 0.0) or 0.0) - float(reserve or 0.0)
        core_node = obs.get("core_node")
        nodes = obs.get("nodes", {}) or {}

        completed = {
            building.get("building_type")
            for node in nodes.values()
            for building in node.get("buildings", []) or []
            if (
                building.get("owner") == self.player_id
                and building.get("complete")
                and float(building.get("hp", 0.0) or 0.0) > 0
            )
        }

        core = nodes.get(core_node, {})
        core_buildings = [
            building for building in core.get("buildings", []) or []
            if building.get("owner") == self.player_id
        ]
        core_types = {building.get("building_type") for building in core_buildings}
        if len(core_buildings) < 2:
            for building_type in self.build_plan:
                if building_type.value in core_types:
                    continue
                if building_type == BuildingType.TECH_2 and BuildingType.TECH_1.value not in completed:
                    continue
                if resources >= BUILDING_COST[building_type]:
                    return Action(
                        action_type=ActionType.BUILD,
                        destination=core_node,
                        building_type=building_type,
                    )

        if not self.cp_turrets:
            return None
        for node_id, node in nodes.items():
            if NODE_TYPES[node_id] != NodeType.CONTROL_POINT:
                continue
            if node.get("owner") != self.player_id or int(node.get("neutral_hp", 0) or 0) > 0:
                continue
            existing = [
                building for building in node.get("buildings", []) or []
                if building.get("owner") == self.player_id
            ]
            if not existing and resources >= BUILDING_COST[BuildingType.TURRET]:
                return Action(
                    action_type=ActionType.BUILD,
                    destination=node_id,
                    building_type=BuildingType.TURRET,
                )
        return None

    def _saving_for_first_building(self, obs: dict) -> bool:
        if not self.save_for_first_building or not self.build_plan:
            return False
        core_node = obs.get("core_node")
        core = (obs.get("nodes", {}) or {}).get(core_node, {})
        existing = [
            building for building in core.get("buildings", []) or []
            if building.get("owner") == self.player_id
        ]
        if existing:
            return False
        resources = float(obs.get("resources", 0.0) or 0.0)
        return resources < BUILDING_COST[self.build_plan[0]]

    def _production_action(self, obs: dict, reserve: float = 0.0) -> Action | None:
        resources = float(obs.get("resources", 0.0) or 0.0) - float(reserve or 0.0)
        if resources < min(UNIT_COST.values()):
            return None
        faction = obs.get("faction", "humans")
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
        priority = self.production_priority or self._dynamic_production_priority(obs, faction, completed)
        for unit_type in priority:
            if UNIT_FACTION.get(unit_type) != faction:
                continue
            if any(required.value not in completed for required in unit_tech_requirements(unit_type)):
                continue
            if resources >= UNIT_COST[unit_type]:
                return Action(action_type=ActionType.PRODUCE, unit_type=unit_type)
        return None

    def _dynamic_production_priority(self, obs: dict, faction: str, completed: set[str]) -> tuple[UnitType, ...]:
        priority: list[UnitType] = []
        for unit_type in TECH2_UNITS_BY_FACTION.get(faction, ()):
            if all(required.value in completed for required in unit_tech_requirements(unit_type)):
                priority.append(unit_type)
        for unit_type in TECH1_UNITS_BY_FACTION.get(faction, ()):
            if all(required.value in completed for required in unit_tech_requirements(unit_type)):
                priority.append(unit_type)

        core_node = obs.get("core_node")
        core_units = (obs.get("nodes", {}) or {}).get(core_node, {}).get("units", {}) or {}
        basics = tuple(BASIC_UNITS_BY_FACTION.get(faction, ()))
        priority.extend(sorted(
            basics,
            key=lambda unit: (
                int(core_units.get(unit.value, 0) or 0),
                int(UNIT_COST[unit]),
                unit.value,
            ),
        ))
        return tuple(priority)

    def _production_priority_for_faction(self, faction: str) -> tuple[UnitType, ...]:
        if self.production_priority:
            return self.production_priority
        if faction == "infected":
            return (
                UnitType.BEHEMOTH,
                UnitType.RAVAGER,
                UnitType.BRUTE,
                UnitType.SPITTER,
                UnitType.SWARMLING,
            )
        if faction == "robots":
            return (
                UnitType.SIEGE_PLATFORM,
                UnitType.GUARDIAN_MECH,
                UnitType.WALKER,
                UnitType.LASER,
                UnitType.DRONE,
            )
        return (UnitType.BMP, UnitType.RAIDER, UnitType.VANGUARD)


def make_scripted_bot_variant(model_id: str, player_id: str, team: str) -> ScriptedBot | None:
    if model_id == "scripted":
        return ScriptedBot(
            player_id,
            team,
            core_focus=True,
            cp_turrets=False,
            save_for_first_building=False,
            produce_threshold=6,
        )
    if model_id == "scripted_core":
        return ScriptedBot(
            player_id,
            team,
            core_focus=True,
            build_plan=(),
            cp_turrets=False,
            save_for_first_building=False,
            produce_threshold=7,
        )
    if model_id == "scripted_core_raider":
        return ScriptedBot(
            player_id,
            team,
            core_focus=True,
            build_plan=(BuildingType.TECH_1,),
            cp_turrets=False,
            save_for_first_building=False,
            produce_threshold=7,
        )
    if model_id == "scripted_core_bmp":
        return ScriptedBot(
            player_id,
            team,
            core_focus=True,
            build_plan=(BuildingType.TECH_1, BuildingType.TECH_2),
            cp_turrets=False,
            save_for_first_building=False,
            produce_threshold=5,
        )
    return None
