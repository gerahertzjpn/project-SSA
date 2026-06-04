"""
Main game engine. Runs one world tick at a time.

Tick order:
1. Existing persistent battles tick once.
2. Moving groups advance, collide on edges, join battles, or arrive at nodes.
3. Income is collected.
4. Bots choose actions.
5. Movement actions create MovingGroups.
6. Production is applied.
7. Victory is checked.
8. Tick snapshot is written to the replay buffer.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from .constants import (
    ActionType, UnitType, TICK_LIMIT, TRAVEL_TIME, EDGE_COLLISION_THRESHOLD,
    MAX_COMBAT_TICKS, TEAM_A_PLAYERS, TEAM_B_PLAYERS, CORE_HP,
    DEATH_TIME_START_TICK, NEUTRAL_GUARD_HP, death_time_core_damage_for_tick,
    ALLIED_LORD_PUSH_DELAY_TICKS, SPECIAL_DEST_ALLY_LORD, SPECIAL_DEST_ENEMY_LORD,
    SPECIAL_LORD_DESTINATIONS,
)
from .state import GameState, MovingGroup, NodeState, Battle, get_enemy_team, get_team
from .map import ADJACENCY, NODE_TYPES, NodeType
from .map2d import (
    approach_edge_length,
    canonical_lane_id,
    directed_approach_id,
    MOVEMENT_MAP_UNITS_PER_TICK,
    NODE_LAYOUT,
    node_enter_length,
    node_exit_length,
    node_transit_length,
    node_zone_progress_interval,
    threat_zone_entry_progress,
    threat_zone_ids_for_approach,
)
from .visual2d import moving_group_visual
from .combat import (
    build_player_units, build_neutral_units, build_core_unit, build_baron_unit,
    count_survivors_by_player,
    render_position,
)
from .combat2d import (
    place_player_units_2d,
    place_static_units_2d,
    tick_battle_units_2d,
    has_hostile_units_2d,
    unit_range_map,
)
from .production import apply_production, collect_income
from .units import UNIT_STATS


def unit_total(units: dict) -> int:
    return sum(int(count or 0) for count in (units or {}).values())


NODE_RANGED_THREAT_SCALE = 24.0
NODE_RANGED_MIN_RANGE = 1.0


@dataclass
class Action:
    action_type: ActionType
    unit_type: Optional[UnitType] = None   # for PRODUCE
    destination: Optional[str] = None      # for MOVE
    source: Optional[str] = None           # for MOVE (which node to send from)
    intent: str = "CAPTURE"               # CAPTURE/ATTACK engages neutrals, TRANSIT does not
    sub_actions: Optional[list] = None     # for MULTI: bounded action bundle


class GameEngine:
    def __init__(
        self,
        state: GameState,
        bots: dict,
        record_trace: bool = True,
        record_replay: bool = True,
    ):
        """
        bots: dict mapping player_id -> Bot instance
        record_trace: pass False during RL training to skip per-tick combat snapshots
        record_replay: pass False during RL training to avoid accumulating replay JSON
        """
        self.state = state
        self.bots = bots
        self.record_trace = record_trace
        self.record_replay = record_replay
        self.replay_ticks = []
        if self.record_replay:
            self._initialize_visual_state()

    # ------------------------------------------------------------------
    # Replay visual identity
    # ------------------------------------------------------------------

    def _initialize_visual_state(self) -> None:
        """Backfill stable visual ids for replay-only unit tracks."""

        for node_id, node in self.state.nodes.items():
            if node.owner in self.state.players and node.units:
                self._ensure_node_visual_units(node, node.owner, node.units)
            for visitor_id, units in list((node.visiting_units or {}).items()):
                self._ensure_node_visual_units(node, visitor_id, units, visiting=True)
            if node.neutral_hp > 0:
                self._ensure_node_visual_units(
                    node,
                    "NEUTRAL",
                    {"NEUTRAL": int(node.neutral_hp)},
                )
            if node_id == "Baron" and self.state.baron.hp > 0:
                self._ensure_node_visual_units(node, "BARON", {"BARON": 1})

    def _new_visual_id(self, owner: str, unit_type: str) -> str:
        counter = int(getattr(self.state, "visual_unit_counter", 0) or 0)
        self.state.visual_unit_counter = counter + 1
        return f"vu:{counter:06d}:{owner}:{self._unit_type_code(str(unit_type))}"

    def _unit_count_items(self, units: dict) -> list[tuple[str, int]]:
        items: list[tuple[str, int]] = []
        for unit_type, count in (units or {}).items():
            normalized = unit_type.value if isinstance(unit_type, UnitType) else str(unit_type)
            amount = int(count or 0)
            if amount > 0:
                items.append((normalized, amount))
        return items

    def _ensure_visual_bucket(self, bucket: dict, owner: str, units: dict) -> dict:
        if not self.record_replay:
            return {}
        expected = {unit_type: count for unit_type, count in self._unit_count_items(units)}
        for unit_type in list(bucket.keys()):
            if unit_type not in expected:
                bucket.pop(unit_type, None)
        for unit_type, count in expected.items():
            ids = bucket.setdefault(unit_type, [])
            while len(ids) < count:
                ids.append(self._new_visual_id(owner, unit_type))
            if len(ids) > count:
                del ids[count:]
        return bucket

    def _ensure_node_visual_units(
        self,
        node: NodeState,
        owner: str,
        units: dict,
        visiting: bool = False,
    ) -> dict:
        if not self.record_replay or not owner:
            return {}
        container = node.visiting_visual_units if visiting else node.visual_units
        bucket = container.setdefault(owner, {})
        self._ensure_visual_bucket(bucket, owner, units)
        if not bucket:
            container.pop(owner, None)
        return bucket

    def _take_visual_units_from_bucket(self, bucket: dict, owner: str, units: dict) -> dict:
        if not self.record_replay:
            return {}
        taken: dict[str, list[str]] = {}
        for unit_type, count in self._unit_count_items(units):
            ids = bucket.setdefault(unit_type, [])
            while len(ids) < count:
                ids.append(self._new_visual_id(owner, unit_type))
            taken[unit_type] = list(ids[:count])
            del ids[:count]
            if not ids:
                bucket.pop(unit_type, None)
        return {unit_type: ids for unit_type, ids in taken.items() if ids}

    def _take_node_visual_units(
        self,
        node: NodeState,
        owner: str,
        units: dict,
        visiting: bool = False,
    ) -> dict:
        if not self.record_replay or not owner:
            return {}
        container = node.visiting_visual_units if visiting else node.visual_units
        bucket = self._ensure_node_visual_units(node, owner, units, visiting=visiting)
        taken = self._take_visual_units_from_bucket(bucket, owner, units)
        if not bucket:
            container.pop(owner, None)
        return taken

    def _fresh_visual_units(self, owner: str, units: dict) -> dict:
        if not self.record_replay:
            return {}
        return {
            unit_type: [self._new_visual_id(owner, unit_type) for _ in range(count)]
            for unit_type, count in self._unit_count_items(units)
        }

    def _trim_visual_units_to_counts(self, visual_units: Optional[dict], owner: str, units: dict) -> dict:
        if not self.record_replay:
            return {}
        visual_units = {unit_type: list(ids) for unit_type, ids in (visual_units or {}).items()}
        trimmed: dict[str, list[str]] = {}
        for unit_type, count in self._unit_count_items(units):
            ids = list(visual_units.get(unit_type, []))
            while len(ids) < count:
                ids.append(self._new_visual_id(owner, unit_type))
            trimmed[unit_type] = ids[:count]
        return {unit_type: ids for unit_type, ids in trimmed.items() if ids}

    def _merge_visual_units_into_bucket(
        self,
        bucket: dict,
        owner: str,
        units: dict,
        visual_units: Optional[dict] = None,
    ) -> None:
        if not self.record_replay:
            return
        incoming = self._trim_visual_units_to_counts(visual_units, owner, units)
        for unit_type, ids in incoming.items():
            bucket.setdefault(unit_type, []).extend(ids)

    def _assign_visual_ids_to_units(
        self,
        units: list,
        owner: str,
        visual_units: Optional[dict],
    ) -> None:
        if not self.record_replay:
            return
        available = {
            unit_type: list(ids)
            for unit_type, ids in (visual_units or {}).items()
        }
        for unit in units:
            ids = available.setdefault(unit.unit_type, [])
            unit.visual_id = ids.pop(0) if ids else self._new_visual_id(owner, unit.unit_type)

    def _visual_survivors_by_owner(self, units: list) -> dict:
        survivors: dict[str, dict[str, list[str]]] = {}
        if not self.record_replay:
            return survivors
        for unit in units:
            if not unit.is_alive() or not getattr(unit, "visual_id", ""):
                continue
            owner = unit.team
            unit_type = unit.unit_type
            survivors.setdefault(owner, {}).setdefault(unit_type, []).append(unit.visual_id)
        return survivors

    def _visual_ids_from_battle_units(
        self,
        battle: Battle,
        owner: str,
        source_unit_type: str,
        target_unit_type: str,
        count: int = 1,
    ) -> dict:
        if not self.record_replay:
            return {}
        ids: list[str] = []
        for unit in battle.units:
            if (
                unit.team == owner
                and unit.unit_type == source_unit_type
                and getattr(unit, "visual_id", "")
            ):
                ids.append(unit.visual_id)
                if len(ids) >= count:
                    break
        return {target_unit_type: ids} if ids else {}

    def _remove_player_visual_units(self, player_id: str) -> None:
        if not self.record_replay:
            return
        for node in self.state.nodes.values():
            node.visual_units.pop(player_id, None)
            node.visiting_visual_units.pop(player_id, None)

    def run(self) -> str:
        """Run match to completion. Returns winner string."""
        while self.state.winner is None and self.state.tick < TICK_LIMIT:
            self.tick()

        if self.state.winner is None:
            self._finalize_death_time()
        if self.state.winner is None:
            self.state.winner = "DRAW"
            self.state.winner_type = "draw_no_core_damage"
        return self.state.winner

    def tick(self) -> None:
        state = self.state
        tick_events = []

        self._tick_active_battles(tick_events)
        lord_event = self._launch_ready_allied_lord_push()
        if lord_event:
            tick_events.append(lord_event)
        self._advance_moving_groups(tick_events)

        collect_income(state)

        actions: dict[str, Action] = {}
        for player_id, bot in self.bots.items():
            player = state.players[player_id]
            if not player.alive:
                continue
            actions[player_id] = bot.act(self._observe(player_id))

        for player_id, action in actions.items():
            for step in self._action_steps(action):
                if step.action_type == ActionType.MOVE:
                    event = self._apply_move(player_id, step)
                    if event:
                        tick_events.append(event)

        for player_id, action in actions.items():
            for step in self._action_steps(action):
                if step.action_type != ActionType.PRODUCE or not step.unit_type:
                    continue
                if apply_production(state, player_id, step.unit_type):
                    core_node = state.nodes[state.players[player_id].core_node]
                    if self.record_replay:
                        self._merge_visual_units_into_bucket(
                            core_node.visual_units.setdefault(player_id, {}),
                            player_id,
                            {step.unit_type.value: 1},
                        )
                    self._remember_node_entry_side(core_node, player_id, self._default_player_side(player_id))

        self._apply_death_time_damage(tick_events)
        self._check_victory()
        if self.record_replay:
            self.replay_ticks.append(self._snapshot_tick(state, tick_events))

        state.tick += 1

    def _action_steps(self, action: Optional[Action]) -> list[Action]:
        if action is None:
            return []
        if action.action_type != ActionType.MULTI:
            return [action]
        steps = [
            step for step in (action.sub_actions or [])
            if isinstance(step, Action) and step.action_type in (ActionType.PRODUCE, ActionType.MOVE)
        ]
        return steps[:2]

    def observe(self, player_id: str) -> dict:
        return self._observe(player_id)

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def _advance_moving_groups(self, tick_events: list) -> None:
        state = self.state
        if not state.moving_groups:
            return

        state.moving_groups = self._start_node_entry_battles(state.moving_groups, tick_events)
        state.moving_groups = self._start_node_transit_battles(state.moving_groups, tick_events)
        if not state.moving_groups:
            return

        for group in state.moving_groups:
            group.progress = min(1.0, group.progress + 1.0 / max(1, group.travel_ticks))

        state.moving_groups = self._join_reached_lord_push_groups(state.moving_groups, tick_events)
        if not state.moving_groups:
            return

        phased_groups: list[MovingGroup] = []
        for group in state.moving_groups:
            if group.progress >= 1.0 and group.phase != "edge":
                event, keep_moving = self._complete_group_segment(group)
                if event:
                    tick_events.append(event)
                if keep_moving:
                    phased_groups.append(group)
                continue
            phased_groups.append(group)

        remaining = self._start_reached_threat_zone_battles(phased_groups, tick_events)

        remaining = self._join_reached_edge_battles(remaining, tick_events)

        remaining = self._create_edge_collisions(remaining, tick_events)
        remaining = self._start_reached_threat_zone_battles(remaining, tick_events)
        remaining = self._join_reached_edge_battles(remaining, tick_events)

        still_moving: list[MovingGroup] = []
        for group in remaining:
            if group.progress >= 1.0:
                event, keep_moving = self._complete_group_segment(group)
                if event:
                    tick_events.append(event)
                if keep_moving:
                    still_moving.append(group)
            else:
                still_moving.append(group)

        still_moving = self._start_node_entry_battles(still_moving, tick_events)
        still_moving = self._start_node_transit_battles(still_moving, tick_events)
        state.moving_groups = still_moving

    def _join_reached_lord_push_groups(
        self,
        groups: list[MovingGroup],
        tick_events: list,
    ) -> list[MovingGroup]:
        lord_groups = [
            group for group in groups
            if group.intent == "LORD_PUSH" and group.units.get(UnitType.LORD.value)
        ]
        if not lord_groups:
            return groups

        remaining: list[MovingGroup] = []
        for group in groups:
            if group.intent != "JOIN_LORD" or group.units.get(UnitType.LORD.value):
                remaining.append(group)
                continue
            lord_group = next(
                (
                    candidate for candidate in lord_groups
                    if candidate.team == group.team
                    and self._groups_share_movement_segment(candidate, group)
                    and self._group_progress_on_canonical_segment(group) >= self._group_progress_on_canonical_segment(candidate) - EDGE_COLLISION_THRESHOLD
                ),
                None,
            )
            if lord_group is None:
                remaining.append(group)
                continue
            self._merge_group_into_lord_push(lord_group, group)
            tick_events.append({
                "type": "LORD_SUPPORT_JOINED",
                "lord_player": lord_group.player_id,
                "player": group.player_id,
                "units": dict(group.units),
                "path": list(lord_group.path),
                "arrives": lord_group.arrival_tick,
            })
        return remaining

    def _groups_share_movement_segment(self, first: MovingGroup, second: MovingGroup) -> bool:
        if first.phase != second.phase:
            return False
        first_segment = self._moving_group_segment_nodes(first)
        second_segment = self._moving_group_segment_nodes(second)
        return first_segment == second_segment

    def _group_progress_on_canonical_segment(self, group: MovingGroup) -> float:
        progress = max(0.0, min(1.0, float(group.progress or 0.0)))
        segment_src, segment_dst = self._moving_group_segment_nodes(group)
        if not segment_src or not segment_dst:
            return progress
        edge_start, _ = self._canonical_edge(segment_src, segment_dst)
        return progress if segment_src == edge_start else 1.0 - progress

    def _merge_group_into_lord_push(self, lord_group: MovingGroup, support_group: MovingGroup) -> None:
        for unit_type, count in (support_group.units or {}).items():
            amount = int(count or 0)
            if amount > 0:
                lord_group.units[unit_type] = lord_group.units.get(unit_type, 0) + amount
        if self.record_replay:
            for unit_type, ids in (support_group.visual_units or {}).items():
                lord_group.visual_units.setdefault(unit_type, []).extend(ids)
        lord_group.speed = self._group_speed(lord_group.units)
        self._set_group_segment(lord_group, lord_group.segment_index, progress=lord_group.progress)

    def _start_node_entry_battles(
        self,
        groups: list[MovingGroup],
        tick_events: list,
    ) -> list[MovingGroup]:
        remaining: list[MovingGroup] = []
        for group in groups:
            event = self._maybe_start_node_entry_battle(group)
            if event:
                tick_events.append(event)
            else:
                remaining.append(group)
        return remaining

    def _maybe_start_node_entry_battle(self, group: MovingGroup) -> Optional[dict]:
        if group.phase != "node_enter" or group.total_units() <= 0:
            return None
        node_id = group.dst
        if node_id not in self.state.nodes:
            return None

        if self._node_key(node_id) in self.state.active_battles:
            battle = self.state.active_battles[self._node_key(node_id)]
            return self._join_battle(battle, group, reinforced=True)

        node = self.state.nodes[node_id]
        if not self._node_has_hostile_occupants(node, group.player_id):
            return None

        include_visitors = self._has_enemy_visitors(node, group.player_id)
        include_owner_units = self._node_owner_should_join_entry_battle(node, group.player_id, include_visitors)
        if NODE_TYPES[node_id] == NodeType.CORE and not self._is_destroyed_core_node(node_id):
            return self._start_node_battle(
                group=group,
                node_id=node_id,
                objective="core",
                core_owner=node_id,
                include_owner_units=include_owner_units,
                include_visitors=True,
                include_core=True,
            )

        return self._start_node_battle(
            group=group,
            node_id=node_id,
            objective="pvp",
            include_owner_units=include_owner_units,
            include_visitors=include_visitors,
        )

    def _start_node_transit_battles(
        self,
        groups: list[MovingGroup],
        tick_events: list,
    ) -> list[MovingGroup]:
        remaining: list[MovingGroup] = []
        for group in groups:
            event = self._maybe_join_active_node_transit_battle(group)
            if event:
                tick_events.append(event)
            else:
                remaining.append(group)

        used: set[int] = set()
        for i, first in enumerate(remaining):
            if i in used or not self._group_node_phase_location(first):
                continue
            for j in range(i + 1, len(remaining)):
                if j in used:
                    continue
                second = remaining[j]
                if not self._node_transit_groups_collide(first, second):
                    continue
                node_id = str(self._group_node_phase_location(first))
                node = self.state.nodes[node_id]
                has_enemy_visitors = self._has_enemy_visitors(node, first.player_id)
                event = self._start_node_battle(
                    group=first,
                    node_id=node_id,
                    objective="pvp",
                    include_owner_units=self._node_owner_should_join_entry_battle(
                        node,
                        first.player_id,
                        has_enemy_visitors,
                    ),
                    include_visitors=has_enemy_visitors,
                )
                tick_events.append(event)
                battle = self.state.active_battles[self._node_key(node_id)]
                tick_events.append(self._join_battle(battle, second, reinforced=True))
                used.add(i)
                used.add(j)
                break

        return [group for index, group in enumerate(remaining) if index not in used]

    def _maybe_join_active_node_transit_battle(self, group: MovingGroup) -> Optional[dict]:
        node_id = self._group_node_phase_location(group)
        if not node_id:
            return None
        battle = self.state.active_battles.get(self._node_key(str(node_id)))
        if not battle:
            return None
        return self._join_battle(battle, group, reinforced=True)

    def _node_transit_groups_collide(self, first: MovingGroup, second: MovingGroup) -> bool:
        first_node = self._group_node_phase_location(first)
        second_node = self._group_node_phase_location(second)
        if not first_node or first_node != second_node:
            return False
        if first.team == second.team:
            return False
        return first.total_units() > 0 and second.total_units() > 0

    def _group_node_phase_location(self, group: MovingGroup) -> Optional[str]:
        if group.phase == "node_enter":
            return group.dst
        if group.phase == "node_exit":
            return group.src
        if group.phase == "node_transit" and group.segment_node:
            return str(group.segment_node)
        return None

    def _start_reached_threat_zone_battles(
        self,
        groups: list[MovingGroup],
        tick_events: list,
    ) -> list[MovingGroup]:
        remaining: list[MovingGroup] = []
        for group in groups:
            event = self._maybe_start_threat_zone_battle(group)
            if event:
                tick_events.append(event)
            else:
                remaining.append(group)
        return remaining

    def _maybe_start_threat_zone_battle(self, group: MovingGroup) -> Optional[dict]:
        if group.phase != "edge" or group.total_units() <= 0:
            return None
        try:
            approach_id = directed_approach_id(group.src, group.dst)
        except (KeyError, ValueError):
            return None

        threat_zone_ids = list(threat_zone_ids_for_approach(approach_id))
        if not threat_zone_ids:
            return self._start_node_ranged_approach_battle(group, approach_id)

        for zone_id in threat_zone_ids:
            if group.progress + 1e-9 < threat_zone_entry_progress(approach_id, zone_id):
                continue
            entry_progress = threat_zone_entry_progress(approach_id, zone_id)
            if zone_id == "Lord":
                return self._start_lord_approach_battle(group, approach_id, threat_zone_ids, entry_progress)
            if zone_id.startswith("core:"):
                core_owner = zone_id.split(":", 1)[1]
                return self._start_core_approach_battle(group, approach_id, core_owner, threat_zone_ids, entry_progress)
        return self._start_node_ranged_approach_battle(group, approach_id)

    def _start_lord_approach_battle(
        self,
        group: MovingGroup,
        approach_id: str,
        threat_zone_ids: list[str],
        edge_position: float,
    ) -> Optional[dict]:
        if group.dst != "Baron":
            return None
        if self.state.nodes["Baron"].owner != "NEUTRAL" or self.state.baron.hp <= 0:
            return None
        return self._start_node_battle(
            group=group,
            node_id="Baron",
            objective="baron",
            include_baron=True,
            include_visitors=True,
            location_kind="approach",
            approach_id=approach_id,
            threat_zone_ids=threat_zone_ids,
            edge_position=edge_position,
        )

    def _start_core_approach_battle(
        self,
        group: MovingGroup,
        approach_id: str,
        core_owner: str,
        threat_zone_ids: list[str],
        edge_position: float,
    ) -> Optional[dict]:
        if group.dst != core_owner:
            return None
        if self._is_destroyed_core_node(core_owner):
            return None
        if get_team(group.player_id) == get_team(core_owner):
            return None
        return self._start_node_battle(
            group=group,
            node_id=core_owner,
            objective="core",
            core_owner=core_owner,
            include_owner_units=True,
            include_visitors=True,
            include_core=True,
            location_kind="approach",
            approach_id=approach_id,
            threat_zone_ids=threat_zone_ids,
            edge_position=edge_position,
        )

    def _start_node_ranged_approach_battle(
        self,
        group: MovingGroup,
        approach_id: str,
    ) -> Optional[dict]:
        for node_id in (group.dst, group.src):
            if not node_id or node_id not in self.state.nodes:
                continue
            if self._is_destroyed_core_node(node_id):
                continue
            node = self.state.nodes[node_id]
            combat_range = self._hostile_ranged_range(node, group.player_id)
            if combat_range <= NODE_RANGED_MIN_RANGE:
                continue
            radius = NODE_LAYOUT[node_id].radius + combat_range * NODE_RANGED_THREAT_SCALE
            start, end = node_zone_progress_interval(approach_id, node_id, radius)
            if group.progress + 1e-9 < start or group.progress - 1e-9 > end:
                continue

            return self._start_node_battle(
                group=group,
                node_id=node_id,
                objective="pvp",
                include_owner_units=True,
                include_visitors=True,
                location_kind="approach",
                approach_id=approach_id,
                threat_zone_ids=[f"ranged:{node_id}"],
                edge_position=start,
            )
        return None

    def _join_reached_edge_battles(
        self,
        groups: list[MovingGroup],
        tick_events: list,
    ) -> list[MovingGroup]:
        remaining: list[MovingGroup] = []
        for group in groups:
            battle = self._edge_battle_for_group(group)
            if battle and self._group_reached_edge_battle(group, battle):
                tick_events.append(self._join_battle(battle, group, reinforced=True))
            else:
                remaining.append(group)
        return remaining

    def _create_edge_collisions(
        self,
        groups: list[MovingGroup],
        tick_events: list,
    ) -> list[MovingGroup]:
        used: set[int] = set()

        for i, first in enumerate(groups):
            if i in used:
                continue
            for j in range(i + 1, len(groups)):
                if j in used:
                    continue
                second = groups[j]
                if not self._groups_collide_on_edge(first, second):
                    continue

                battle = self._create_edge_battle(first, second)
                used.add(i)
                used.add(j)
                tick_events.append({
                    "type": "BATTLE_STARTED",
                    "battle_type": "edge_battle",
                    "location_kind": "edge",
                    "location_id": battle.location_id,
                    "battle_id": battle.battle_id,
                    "participants": [first.player_id, second.player_id],
                    "edge_position": battle.edge_position,
                })
                break

        return [group for idx, group in enumerate(groups) if idx not in used]

    def _groups_collide_on_edge(self, first: MovingGroup, second: MovingGroup) -> bool:
        if first.phase != "edge" or second.phase != "edge":
            return False
        if first.src != second.dst or first.dst != second.src:
            return False
        if first.team == second.team:
            return False
        if first.total_units() <= 0 or second.total_units() <= 0:
            return False
        return first.progress + second.progress >= 1.0 - EDGE_COLLISION_THRESHOLD

    def _create_edge_battle(self, first: MovingGroup, second: MovingGroup) -> Battle:
        key = self._edge_key(first.src, first.dst)
        edge_position = self._edge_collision_position(first, second)
        battle = Battle(
            battle_id=key,
            location_kind="edge",
            location_id=self._edge_id(first.src, first.dst),
            units=[],
            started_at_tick=self.state.tick,
            last_tick=self.state.tick,
            edge_endpoints=self._canonical_edge(first.src, first.dst),
            edge_position=edge_position,
            objective="pvp",
        )
        self._join_battle(battle, first, reinforced=False)
        self._join_battle(battle, second, reinforced=False)
        self.state.active_battles[key] = battle
        return battle

    def _edge_collision_position(self, first: MovingGroup, second: MovingGroup) -> float:
        first_progress, second_progress = self._edge_collision_contact_progress(first, second)
        if first.src == second.dst and first.dst == second.src:
            position = (first_progress + (1.0 - second_progress)) / 2.0
        else:
            position = first_progress
        position = max(0.0, min(1.0, position))
        edge_start, _ = self._canonical_edge(first.src, first.dst)
        return position if first.src == edge_start else 1.0 - position

    def _edge_collision_contact_progress(self, first: MovingGroup, second: MovingGroup) -> tuple[float, float]:
        first_step = 1.0 / max(1.0, float(first.travel_ticks))
        second_step = 1.0 / max(1.0, float(second.travel_ticks))
        first_before = max(0.0, float(first.progress) - first_step)
        second_before = max(0.0, float(second.progress) - second_step)
        first_after = float(first.progress)
        second_after = float(second.progress)
        threshold = 1.0 - EDGE_COLLISION_THRESHOLD
        before_total = first_before + second_before
        after_total = first_after + second_after
        delta_total = after_total - before_total
        if delta_total > 1e-9 and before_total < threshold <= after_total:
            t = (threshold - before_total) / delta_total
            return (
                first_before + (first_after - first_before) * t,
                second_before + (second_after - second_before) * t,
            )
        return first_after, second_after

    def _edge_battle_for_group(self, group: MovingGroup) -> Optional[Battle]:
        if group.phase != "edge":
            return None
        return self.state.active_battles.get(self._edge_key(group.src, group.dst))

    def _group_reached_edge_battle(self, group: MovingGroup, battle: Battle) -> bool:
        if battle.edge_endpoints and group.src == battle.edge_endpoints[1]:
            target_progress = 1.0 - battle.edge_position
        else:
            target_progress = battle.edge_position
        return group.progress >= target_progress - EDGE_COLLISION_THRESHOLD

    def _apply_move(self, player_id: str, action: Action) -> Optional[dict]:
        state = self.state
        src = action.source or state.players[player_id].core_node
        dst = action.destination

        if dst is None or src not in state.nodes:
            return None
        requested_dst = dst
        resolved_dst = self._resolve_special_lord_destination(player_id, requested_dst, src)
        if resolved_dst is None:
            return None
        path = self._find_route(player_id, src, resolved_dst)
        if not path:
            return None

        if (
            len(path) == 2
            and resolved_dst == "Baron"
            and action.intent.upper() == "TRANSIT"
            and state.nodes["Baron"].owner == "NEUTRAL"
            and state.baron.hp > 0
        ):
            return {
                "type": "BARON_BLOCKED",
                "player": player_id,
                "src": src,
                "dst": resolved_dst,
                "reason": "living_baron_blocks_transit",
            }

        src_node = state.nodes[src]
        units_to_send, visual_units_to_send = self._take_units_for_move(player_id, src_node)
        if not units_to_send:
            return None

        speed = self._group_speed(units_to_send)
        group = MovingGroup(
            player_id=player_id,
            team=get_team(player_id),
            units=units_to_send,
            src=src,
            dst=path[1],
            arrival_tick=self._estimate_arrival_tick(state.tick, path, speed),
            started_tick=state.tick,
            travel_ticks=1.0,
            progress=0.0,
            intent=action.intent.upper(),
            path=path,
            speed=speed,
            final_dst=requested_dst,
            visual_units=visual_units_to_send,
        )
        self._set_group_segment(group, 0, progress=0.0)
        state.moving_groups.append(group)

        return {
            "type": "MOVE",
            "player": player_id,
            "src": src,
            "dst": requested_dst,
            "resolved_dst": resolved_dst,
            "path": path,
            "units": units_to_send,
            "intent": group.intent,
            "speed": round(speed, 3),
            "arrives": group.arrival_tick,
        }

    def _take_units_for_move(self, player_id: str, node: NodeState) -> tuple[dict, dict]:
        if node.unit_ready_tick.get(player_id, 0) > self.state.tick:
            return {}, {}
        if node.owner == player_id and node.units:
            units = dict(node.units)
            visual_units = self._take_node_visual_units(node, player_id, units)
            node.units = {}
            node.unit_ready_tick.pop(player_id, None)
            node.unit_entry_side.pop(player_id, None)
            return units, visual_units
        if player_id in node.visiting_units and node.visiting_units[player_id]:
            node.unit_ready_tick.pop(player_id, None)
            node.unit_entry_side.pop(player_id, None)
            units = node.visiting_units.pop(player_id)
            visual_units = self._take_node_visual_units(node, player_id, units, visiting=True)
            return units, visual_units
        return {}, {}

    def _resolve_special_lord_destination(
        self,
        player_id: str,
        destination: Optional[str],
        source: Optional[str] = None,
    ) -> Optional[str]:
        if destination not in SPECIAL_LORD_DESTINATIONS:
            return destination if destination in self.state.nodes else None

        owner = self.state.baron.allied_lord_owner
        if not owner or not self.state.baron.allied_lord_alive or self.state.baron.allied_lord_expended:
            return None
        same_team = get_team(owner) == get_team(player_id)
        if destination == SPECIAL_DEST_ALLY_LORD and not same_team:
            return None
        if destination == SPECIAL_DEST_ENEMY_LORD and same_team:
            return None
        return self._current_lord_intercept_node(owner, source=source, hunter_id=player_id)

    def _current_lord_intercept_node(
        self,
        owner: str,
        source: Optional[str] = None,
        hunter_id: Optional[str] = None,
    ) -> Optional[str]:
        battle = self._active_lord_battle(owner)
        if battle:
            if battle.location_kind == "edge":
                endpoints = list(battle.edge_endpoints or ())
                if not endpoints and "|" in battle.location_id:
                    endpoints = battle.location_id.split("|", 1)
                return self._best_accessible_node(source, hunter_id or owner, endpoints)
            return battle.location_id

        lord_group = self._active_lord_group(owner)
        if lord_group:
            node_id = self._group_node_phase_location(lord_group)
            if node_id:
                return str(node_id)
            if lord_group.phase == "edge":
                return self._best_accessible_node(
                    source,
                    hunter_id or owner,
                    [lord_group.dst, lord_group.src],
                )
            return lord_group.dst

        if (
            int(self.state.baron.allied_lord_ready_tick or 0) > self.state.tick
            and float(self.state.baron.allied_lord_hp or 0.0) > 0.0
        ):
            return "Baron"

        for node_id, node in self.state.nodes.items():
            if node.units.get(UnitType.LORD.value) and node.owner == owner:
                return node_id
            if (node.visiting_units.get(owner) or {}).get(UnitType.LORD.value):
                return node_id
        return None

    def _best_accessible_node(
        self,
        source: Optional[str],
        player_id: str,
        candidates: list[str] | tuple[str, ...],
    ) -> Optional[str]:
        valid = [node_id for node_id in candidates if node_id in self.state.nodes]
        if not valid:
            return None
        if not source or source not in self.state.nodes:
            return valid[0]
        scored = []
        for node_id in valid:
            route = self._find_route(player_id, source, node_id)
            if route:
                scored.append((len(route), self._estimate_arrival_tick(self.state.tick, route, 1.0), node_id))
        if scored:
            scored.sort()
            return scored[0][2]
        return valid[0]

    def _active_lord_group(self, owner: str) -> Optional[MovingGroup]:
        for group in self.state.moving_groups:
            if group.player_id == owner and group.units.get(UnitType.LORD.value):
                return group
        return None

    def _active_lord_battle(self, owner: str) -> Optional[Battle]:
        for battle in self.state.active_battles.values():
            if any(
                unit.team == owner
                and unit.unit_type == UnitType.LORD.value
                and unit.is_alive()
                for unit in battle.units
            ):
                return battle
        return None

    def _find_route(self, player_id: str, src: str, dst: str) -> Optional[list[str]]:
        if dst not in self.state.nodes:
            return None
        if src == dst:
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
                if neighbor == dst and self._is_accessible_destination(player_id, neighbor, node_id):
                    return next_path
                if self._is_transit_node(player_id, neighbor):
                    visited.add(neighbor)
                    queue.append(next_path)

        return None

    def _find_unrestricted_route(self, src: str, dst: str) -> Optional[list[str]]:
        if src not in self.state.nodes or dst not in self.state.nodes or src == dst:
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
                if neighbor == dst:
                    return next_path
                visited.add(neighbor)
                queue.append(next_path)
        return None

    def _random_enemy_core(self, player_id: str) -> Optional[str]:
        enemy_team = get_enemy_team(get_team(player_id))
        cores = [
            player.player_id
            for player in self.state.players.values()
            if player.team == enemy_team and player.alive
        ]
        if not cores:
            return None
        index = int(self.state.rng.integers(0, len(cores)))
        return sorted(cores)[index]

    def _is_accessible_destination(self, player_id: str, node_id: str, via_node: str) -> bool:
        node = self.state.nodes[node_id]
        if node_id == "Baron":
            return True
        if node.owner == "NEUTRAL":
            return True
        if get_team(node.owner) == get_team(player_id):
            return True
        return self._is_transit_node(player_id, via_node)

    def _is_transit_node(self, player_id: str, node_id: str) -> bool:
        if node_id == "Baron" and self.state.nodes["Baron"].owner == "NEUTRAL" and self.state.baron.hp > 0:
            return False
        node = self.state.nodes[node_id]
        if node.owner == "NEUTRAL":
            return True
        return get_team(node.owner) == get_team(player_id)

    def _group_speed(self, units: dict) -> float:
        speeds = []
        for unit_type, count in (units or {}).items():
            if int(count or 0) > 0:
                unit_key = unit_type.value if isinstance(unit_type, UnitType) else unit_type
                unit_enum = UnitType.RANGER if unit_key == "ARCHER" else UnitType(unit_key)
                speeds.append(float(UNIT_STATS[unit_enum]["speed"]))
        return min(speeds) if speeds else 1.0

    def _movement_segments(self, path: list[str], speed: float) -> list[dict]:
        segments = []
        for index, node_id in enumerate(path):
            if index == 0:
                segments.append({
                    "phase": "node_exit",
                    "src": node_id,
                    "dst": path[index + 1],
                    "node": node_id,
                    "ticks": self._movement_ticks_for_distance(node_exit_length(node_id), speed),
                })
                continue

            prev_node = path[index - 1]
            segments.append({
                "phase": "edge",
                "src": prev_node,
                "dst": node_id,
                "node": None,
                "ticks": self._movement_ticks_for_distance(approach_edge_length(prev_node, node_id), speed),
            })

            if index == len(path) - 1:
                segments.append({
                    "phase": "node_enter",
                    "src": prev_node,
                    "dst": node_id,
                    "node": node_id,
                    "ticks": self._movement_ticks_for_distance(node_enter_length(node_id), speed),
                })
            else:
                segments.append({
                    "phase": "node_transit",
                    "src": prev_node,
                    "dst": path[index + 1],
                    "node": node_id,
                    "ticks": self._movement_ticks_for_distance(
                        node_transit_length(prev_node, node_id, path[index + 1]),
                        speed,
                    ),
                })
        return segments

    def _movement_ticks_for_distance(self, distance: float, speed: float) -> float:
        return max(0.001, float(distance) / (MOVEMENT_MAP_UNITS_PER_TICK * max(0.001, float(speed))))

    def _set_group_segment(self, group: MovingGroup, segment_index: int, progress: float = 0.0) -> None:
        segments = self._movement_segments(group.path, group.speed)
        segment = segments[segment_index]
        group.segment_index = segment_index
        group.phase = segment["phase"]
        group.src = segment["src"]
        group.dst = segment["dst"]
        group.segment_node = segment["node"]
        group.travel_ticks = max(0.001, segment["ticks"])
        group.progress = progress
        group.arrival_tick = self._estimate_arrival_tick(self.state.tick, group.path, group.speed, group.segment_index, group.progress)

    def _estimate_arrival_tick(
        self,
        current_tick: int,
        path: list[str],
        speed: float,
        segment_index: int = 0,
        progress: float = 0.0,
    ) -> int:
        segments = self._movement_segments(path, speed)
        remaining = 0
        for index, segment in enumerate(segments[segment_index:], start=segment_index):
            ticks = float(segment["ticks"])
            remaining_ticks = ticks * (1.0 - progress if index == segment_index else 1.0)
            remaining += self._discrete_segment_ticks(remaining_ticks)
        return current_tick + max(1, remaining)

    @staticmethod
    def _discrete_segment_ticks(ticks: float) -> int:
        if ticks <= 0:
            return 0
        return max(1, math.ceil(float(ticks)))

    def _complete_group_segment(self, group: MovingGroup) -> tuple[Optional[dict], bool]:
        if group.phase == "node_enter":
            return self._resolve_arrival(group), False

        next_index = group.segment_index + 1
        segments = self._movement_segments(group.path, group.speed)
        if next_index >= len(segments):
            return self._resolve_arrival(group), False

        if group.phase == "edge" and segments[next_index]["phase"] == "node_transit":
            event = self._handle_transit_checkpoint(group)
            if event:
                return event, False

        self._set_group_segment(group, next_index, progress=0.0)
        return None, True

    def _handle_transit_checkpoint(self, group: MovingGroup) -> Optional[dict]:
        node_id = group.dst
        if self._node_key(node_id) in self.state.active_battles:
            battle = self.state.active_battles[self._node_key(node_id)]
            return self._join_battle(battle, group, reinforced=True)

        node_type = NODE_TYPES[node_id]
        if node_type == NodeType.BARON and self.state.nodes["Baron"].owner == "NEUTRAL" and self.state.baron.hp > 0:
            self._return_group_to_source(group)
            return {
                "type": "BARON_BLOCKED",
                "player": group.player_id,
                "src": group.src,
                "dst": node_id,
                "reason": "living_lord_blocks_transit",
            }

        node = self.state.nodes[node_id]
        if node.neutral_hp > 0:
            if self._has_enemy_visitors(node, group.player_id):
                return self._start_node_battle(
                    group=group,
                    node_id=node_id,
                    objective="pvp",
                    include_visitors=True,
                )
            return None

        if node.owner == "NEUTRAL":
            if self._has_enemy_visitors(node, group.player_id):
                return self._start_node_battle(
                    group=group,
                    node_id=node_id,
                    objective="pvp",
                    include_visitors=True,
                )
            return None

        if get_team(node.owner) == get_team(group.player_id):
            if self._has_enemy_visitors(node, group.player_id):
                return self._start_node_battle(
                    group=group,
                    node_id=node_id,
                    objective="pvp",
                    include_owner_units=True,
                    include_visitors=True,
                )
            return None

        return self._start_node_battle(
            group=group,
            node_id=node_id,
            objective="pvp",
            include_owner_units=True,
            include_visitors=True,
        )

    # ------------------------------------------------------------------
    # Arrivals and node battles
    # ------------------------------------------------------------------

    def _resolve_arrival(self, group: MovingGroup) -> Optional[dict]:
        dst = group.dst
        if self._node_key(dst) in self.state.active_battles:
            battle = self.state.active_battles[self._node_key(dst)]
            return self._join_battle(battle, group, reinforced=True)

        special_event = self._handle_lord_special_arrival(group)
        if special_event:
            return special_event

        node_type = NODE_TYPES[dst]
        if node_type == NodeType.BARON:
            return self._handle_baron_arrival(group)
        if node_type == NodeType.CORE:
            return self._handle_core_arrival(group)
        return self._handle_control_point_arrival(group)

    def _handle_lord_special_arrival(self, group: MovingGroup) -> Optional[dict]:
        intent = group.intent.upper()
        if intent not in {"JOIN_LORD", "HUNT_LORD"}:
            return None

        special_destination = (
            SPECIAL_DEST_ALLY_LORD if intent == "JOIN_LORD" else SPECIAL_DEST_ENEMY_LORD
        )
        resolved = self._resolve_special_lord_destination(
            group.player_id,
            special_destination,
            source=group.dst,
        )
        if resolved and resolved != group.dst:
            return self._retarget_special_lord_group(group, resolved)

        if intent == "JOIN_LORD" and group.dst == "Baron" and self._delayed_allied_lord_waiting_at_baron():
            owner = self.state.baron.allied_lord_owner
            if owner and get_team(owner) == get_team(group.player_id):
                self._store_lord_support(group)
                return {
                    "type": "LORD_SUPPORT_WAITING",
                    "player": group.player_id,
                    "lord_player": owner,
                    "node": "Baron",
                    "units": dict(group.units),
                    "ready_tick": self.state.baron.allied_lord_ready_tick,
                }

        if intent == "HUNT_LORD" and group.dst == "Baron" and self._delayed_allied_lord_waiting_at_baron():
            owner = self.state.baron.allied_lord_owner
            if owner and get_team(owner) != get_team(group.player_id):
                return self._start_node_battle(
                    group=group,
                    node_id="Baron",
                    objective="pvp",
                    include_owner_units=True,
                    include_visitors=True,
                    include_allied_lord=True,
                )
        return None

    def _store_lord_support(self, group: MovingGroup) -> None:
        for unit_type, count in (group.units or {}).items():
            amount = int(count or 0)
            if amount > 0:
                current = int(self.state.baron.allied_lord_support_units.get(unit_type, 0) or 0)
                self.state.baron.allied_lord_support_units[unit_type] = current + amount
        if self.record_replay:
            for unit_type, ids in (group.visual_units or {}).items():
                self.state.baron.allied_lord_support_visual_units.setdefault(unit_type, []).extend(ids)

    def _retarget_special_lord_group(self, group: MovingGroup, target_node: str) -> Optional[dict]:
        path = self._find_route(group.player_id, group.dst, target_node)
        if not path:
            return None
        speed = self._group_speed(group.units)
        new_group = MovingGroup(
            player_id=group.player_id,
            team=group.team,
            units=dict(group.units),
            src=group.dst,
            dst=path[1],
            arrival_tick=self._estimate_arrival_tick(self.state.tick, path, speed),
            started_tick=self.state.tick,
            travel_ticks=1.0,
            progress=0.0,
            intent=group.intent,
            path=path,
            speed=speed,
            final_dst=group.final_dst,
            visual_units=group.visual_units,
        )
        self._set_group_segment(new_group, 0, progress=0.0)
        self.state.moving_groups.append(new_group)
        return {
            "type": "LORD_SPECIAL_RETARGETED",
            "player": group.player_id,
            "intent": group.intent,
            "src": group.dst,
            "target": target_node,
            "path": path,
            "units": dict(group.units),
            "arrives": new_group.arrival_tick,
        }

    def _handle_core_arrival(self, group: MovingGroup) -> Optional[dict]:
        node = self.state.nodes[group.dst]
        core_owner = group.dst

        if self._is_destroyed_core_node(core_owner):
            if group.intent == "LORD_PUSH" and group.units.get(UnitType.LORD.value):
                lord_event = self._retarget_lord_push_from_node(
                    player_id=group.player_id,
                    node_id=core_owner,
                    visual_units=group.visual_units,
                    reason="target_core_destroyed_before_arrival",
                )
                if lord_event:
                    return lord_event
            self._neutralize_destroyed_core_node(core_owner)
            if self._has_enemy_visitors(node, group.player_id):
                return self._start_node_battle(
                    group=group,
                    node_id=group.dst,
                    objective="pvp",
                    include_visitors=True,
                )
            self._merge_visiting_units(
                node,
                group.player_id,
                group.units,
                entry_side=self._entry_side(group.player_id, group.src, group.dst),
            )
            self._lock_node_player(node, group.player_id)
            return {
                "type": "MOVE_IN",
                "node": group.dst,
                "player": group.player_id,
                "core_destroyed": True,
            }

        if node.owner == core_owner and get_team(group.player_id) == get_team(core_owner):
            self._merge_units(
                node,
                group.units,
                player_id=group.player_id,
                entry_side=self._entry_side(group.player_id, group.src, group.dst),
            )
            self._lock_node_player(node, group.player_id)
            return {"type": "REINFORCE", "node": group.dst, "player": group.player_id}

        return self._start_node_battle(
            group=group,
            node_id=group.dst,
            objective="core",
            core_owner=core_owner,
            include_owner_units=True,
            include_visitors=True,
            include_core=True,
        )

    def _handle_control_point_arrival(self, group: MovingGroup) -> Optional[dict]:
        node = self.state.nodes[group.dst]
        intent = group.intent.upper()

        if node.neutral_hp > 0:
            if intent in {"CAPTURE", "ATTACK"}:
                return self._start_node_battle(
                    group=group,
                    node_id=group.dst,
                    objective="neutral",
                    include_neutral=True,
                    include_visitors=True,
                )
            if self._has_enemy_visitors(node, group.player_id):
                return self._start_node_battle(
                    group=group,
                    node_id=group.dst,
                    objective="pvp",
                    include_visitors=True,
                )
            self._merge_visiting_units(
                node,
                group.player_id,
                group.units,
                entry_side=self._entry_side(group.player_id, group.src, group.dst),
            )
            self._lock_node_player(node, group.player_id)
            return {
                "type": "TRANSIT",
                "node": group.dst,
                "player": group.player_id,
                "neutral_hp": node.neutral_hp,
            }

        if node.owner == "NEUTRAL":
            if self._has_enemy_visitors(node, group.player_id):
                return self._start_node_battle(
                    group=group,
                    node_id=group.dst,
                    objective="pvp",
                    include_visitors=True,
                )
            node.owner = group.player_id
            self._merge_units(
                node,
                group.units,
                player_id=group.player_id,
                entry_side=self._entry_side(group.player_id, group.src, group.dst),
            )
            self._lock_node_player(node, group.player_id)
            return {"type": "CAPTURE", "node": group.dst, "player": group.player_id}

        if get_team(node.owner) == get_team(group.player_id):
            if self._has_enemy_visitors(node, group.player_id):
                return self._start_node_battle(
                    group=group,
                    node_id=group.dst,
                    objective="pvp",
                    include_owner_units=True,
                    include_visitors=True,
                )
            self._merge_units(
                node,
                group.units,
                player_id=group.player_id,
                entry_side=self._entry_side(group.player_id, group.src, group.dst),
            )
            self._lock_node_player(node, group.player_id)
            return {"type": "REINFORCE", "node": group.dst, "player": group.player_id}

        return self._start_node_battle(
            group=group,
            node_id=group.dst,
            objective="pvp",
            include_owner_units=True,
            include_visitors=True,
        )

    def _handle_baron_arrival(self, group: MovingGroup) -> Optional[dict]:
        state = self.state
        baron_node = state.nodes["Baron"]

        if self._delayed_allied_lord_waiting_at_baron():
            owner = state.baron.allied_lord_owner
            if owner and get_team(owner) != get_team(group.player_id):
                return self._start_node_battle(
                    group=group,
                    node_id="Baron",
                    objective="pvp",
                    include_owner_units=True,
                    include_visitors=True,
                    include_allied_lord=True,
                )

        if baron_node.owner == "NEUTRAL" and state.baron.hp > 0:
            if group.intent.upper() == "TRANSIT":
                self._return_group_to_source(group)
                return {
                    "type": "BARON_BLOCKED",
                    "player": group.player_id,
                    "src": group.src,
                    "dst": "Baron",
                    "reason": "living_baron_blocks_transit",
                }
            return self._start_node_battle(
                group=group,
                node_id="Baron",
                objective="baron",
                include_baron=True,
                include_visitors=True,
            )

        if baron_node.owner != "NEUTRAL":
            if get_team(baron_node.owner) == get_team(group.player_id):
                self._merge_units(
                    baron_node,
                    group.units,
                    player_id=group.player_id,
                    entry_side=self._entry_side(group.player_id, group.src, group.dst),
                )
                self._lock_node_player(baron_node, group.player_id)
                return {"type": "REINFORCE", "node": "Baron", "player": group.player_id}
            return self._start_node_battle(
                group=group,
                node_id="Baron",
                objective="pvp",
                include_owner_units=True,
                include_visitors=True,
            )

        baron_node.owner = group.player_id
        self._merge_units(
            baron_node,
            group.units,
            player_id=group.player_id,
            entry_side=self._entry_side(group.player_id, group.src, group.dst),
        )
        self._lock_node_player(baron_node, group.player_id)
        return {"type": "MOVE_IN", "node": "Baron", "player": group.player_id}

    def _delayed_allied_lord_waiting_at_baron(self) -> bool:
        return (
            bool(self.state.baron.allied_lord_alive)
            and bool(self.state.baron.allied_lord_owner)
            and not bool(self.state.baron.allied_lord_expended)
            and int(self.state.baron.allied_lord_ready_tick or 0) > self.state.tick
            and float(self.state.baron.allied_lord_hp or 0.0) > 0.0
        )

    def _spawn_lord_push(
        self,
        player_id: str,
        node_id: str = "Baron",
        visual_units: Optional[dict] = None,
    ) -> Optional[dict]:
        target = self._random_enemy_core(player_id)
        if target is None:
            return None

        units = {UnitType.LORD.value: 1}
        path = self._find_unrestricted_route(node_id, target)
        if not path or len(path) < 2:
            self._kill_allied_lord(expended=False)
            return {
                "type": "LORD_STOPPED",
                "player": player_id,
                "src": node_id,
                "target": target,
                "reason": "no_route_to_living_enemy_core",
            }

        speed = self._group_speed(units)
        ready_tick = self.state.tick + ALLIED_LORD_PUSH_DELAY_TICKS
        arrival_tick = self._estimate_arrival_tick(ready_tick, path, speed)
        self.state.baron.allied_lord_owner = player_id
        self.state.baron.allied_lord_hp = float(self.state.baron.max_hp)
        self.state.baron.allied_lord_target = target
        self.state.baron.allied_lord_path = list(path)
        self.state.baron.allied_lord_ready_tick = ready_tick
        self.state.baron.allied_lord_arrival_tick = arrival_tick
        self.state.baron.allied_lord_visual_units = (
            self._trim_visual_units_to_counts(visual_units, player_id, units)
            if visual_units
            else self._fresh_visual_units(player_id, units)
        )
        self.state.baron.allied_lord_alive = True
        self.state.baron.allied_lord_expended = False
        return {
            "type": "LORD_SPAWNED",
            "player": player_id,
            "team": get_team(player_id),
            "src": node_id,
            "target": target,
            "path": list(path),
            "units": units,
            "speed": round(speed, 3),
            "ready_tick": ready_tick,
            "arrives": arrival_tick,
            "delay_ticks": ALLIED_LORD_PUSH_DELAY_TICKS,
        }

    def _launch_ready_allied_lord_push(self) -> Optional[dict]:
        baron = self.state.baron
        if (
            not baron.allied_lord_alive
            or baron.allied_lord_expended
            or not baron.allied_lord_owner
            or not baron.allied_lord_ready_tick
            or self.state.tick < int(baron.allied_lord_ready_tick)
        ):
            return None

        owner = baron.allied_lord_owner
        target = self._living_lord_target(owner)
        if target is None:
            self._kill_allied_lord(expended=False)
            return {
                "type": "LORD_STOPPED",
                "player": owner,
                "src": "Baron",
                "reason": "no_living_enemy_core",
            }

        push_units, push_visual_units = self._ready_lord_push_payload(owner)
        event = self._create_lord_push_group(
            player_id=owner,
            node_id="Baron",
            target=target,
            units=push_units,
            visual_units=push_visual_units,
            event_type="LORD_PUSH_STARTED",
            reason="spawn_delay_elapsed",
        )
        baron.allied_lord_ready_tick = 0
        return event

    def _ready_lord_push_payload(self, owner: str) -> tuple[dict, dict]:
        units = {UnitType.LORD.value: 1}
        visual_units = self._trim_visual_units_to_counts(
            self.state.baron.allied_lord_visual_units,
            owner,
            {UnitType.LORD.value: 1},
        )
        for unit_type, count in (self.state.baron.allied_lord_support_units or {}).items():
            amount = int(count or 0)
            if amount > 0:
                units[unit_type] = units.get(unit_type, 0) + amount
        for unit_type, ids in (self.state.baron.allied_lord_support_visual_units or {}).items():
            if ids:
                visual_units.setdefault(unit_type, []).extend(ids)
        self.state.baron.allied_lord_support_units = {}
        self.state.baron.allied_lord_support_visual_units = {}
        return units, visual_units

    def _living_lord_target(self, player_id: str) -> Optional[str]:
        target = self.state.baron.allied_lord_target
        if (
            target in self.state.players
            and self.state.players[target].alive
            and not self._is_destroyed_core_node(target)
        ):
            return target
        target = self._random_enemy_core(player_id)
        self.state.baron.allied_lord_target = target
        return target

    def _retarget_lord_push_from_node(
        self,
        player_id: str,
        node_id: str,
        visual_units: Optional[dict] = None,
        reason: str = "target_unavailable",
    ) -> Optional[dict]:
        if not self.state.baron.allied_lord_alive:
            return None
        target = self._living_lord_target(player_id)
        if target is None:
            self._kill_allied_lord(expended=False)
            return {
                "type": "LORD_STOPPED",
                "player": player_id,
                "src": node_id,
                "reason": "no_living_enemy_core",
            }
        if target == node_id:
            return None
        return self._create_lord_push_group(
            player_id=player_id,
            node_id=node_id,
            target=target,
            units={UnitType.LORD.value: 1},
            visual_units=visual_units,
            event_type="LORD_RETARGETED",
            reason=reason,
        )

    def _create_lord_push_group(
        self,
        player_id: str,
        node_id: str,
        target: str,
        units: dict,
        visual_units: Optional[dict] = None,
        event_type: str = "LORD_PUSH",
        reason: Optional[str] = None,
    ) -> Optional[dict]:
        path = self._find_unrestricted_route(node_id, target)
        if not path or len(path) < 2:
            self._kill_allied_lord(expended=False)
            return {
                "type": "LORD_STOPPED",
                "player": player_id,
                "src": node_id,
                "target": target,
                "reason": "no_route_to_living_enemy_core",
            }

        speed = self._group_speed(units)
        group = MovingGroup(
            player_id=player_id,
            team=get_team(player_id),
            units=units,
            src=node_id,
            dst=path[1],
            arrival_tick=self._estimate_arrival_tick(self.state.tick, path, speed),
            started_tick=self.state.tick,
            travel_ticks=1.0,
            progress=0.0,
            intent="LORD_PUSH",
            path=path,
            speed=speed,
            final_dst=target,
            visual_units=(
                self._trim_visual_units_to_counts(visual_units, player_id, units)
                if visual_units
                else self._fresh_visual_units(player_id, units)
            ),
        )
        self._set_group_segment(group, 0, progress=0.0)
        self.state.moving_groups.append(group)
        self.state.baron.allied_lord_target = target
        self.state.baron.allied_lord_path = list(path)
        self.state.baron.allied_lord_arrival_tick = group.arrival_tick
        self.state.baron.allied_lord_visual_units = {}
        event = {
            "type": event_type,
            "player": player_id,
            "team": get_team(player_id),
            "src": node_id,
            "target": target,
            "path": path,
            "units": units,
            "speed": round(speed, 3),
            "ready_tick": self.state.tick,
            "arrives": group.arrival_tick,
        }
        if reason:
            event["reason"] = reason
        return event

    def _continue_lord_push(self, battle: Battle, player_id: Optional[str]) -> Optional[dict]:
        if not player_id:
            return None
        if battle.participant_intents.get(player_id) != "LORD_PUSH":
            return None
        if not self.state.baron.allied_lord_alive:
            return None

        node = self.state.nodes[battle.location_id]
        lord_count = int((node.units or {}).get(UnitType.LORD.value, 0) or 0)
        if lord_count <= 0:
            return None

        final_dst = battle.participant_final_dsts.get(player_id)
        living_target = self._living_lord_target(player_id)
        if living_target:
            final_dst = living_target
        if not final_dst:
            self._kill_allied_lord(expended=False)
            return {
                "type": "LORD_STOPPED",
                "player": player_id,
                "src": battle.location_id,
                "reason": "no_living_enemy_core",
            }
        if final_dst == battle.location_id:
            return None

        path = self._find_unrestricted_route(battle.location_id, final_dst)
        if not path or len(path) < 2:
            self._kill_allied_lord(expended=False)
            return {
                "type": "LORD_STOPPED",
                "player": player_id,
                "src": battle.location_id,
                "target": final_dst,
                "reason": "no_route_to_living_enemy_core",
            }

        lord_visual_units = self._take_node_visual_units(
            node,
            player_id,
            node.units,
        )
        units = dict(node.units)
        node.units = {}
        node.unit_ready_tick.pop(player_id, None)
        node.unit_entry_side.pop(player_id, None)
        speed = self._group_speed(units)
        group = MovingGroup(
            player_id=player_id,
            team=get_team(player_id),
            units=units,
            src=battle.location_id,
            dst=path[1],
            arrival_tick=self._estimate_arrival_tick(self.state.tick, path, speed),
            started_tick=self.state.tick,
            travel_ticks=1.0,
            progress=0.0,
            intent="LORD_PUSH",
            path=path,
            speed=speed,
            final_dst=final_dst,
            visual_units=lord_visual_units,
        )
        self._set_group_segment(group, 0, progress=0.0)
        self.state.moving_groups.append(group)
        self.state.baron.allied_lord_ready_tick = 0
        return {
            "player": player_id,
            "src": battle.location_id,
            "target": final_dst,
            "path": path,
            "units": units,
            "arrives": group.arrival_tick,
        }

    def _start_node_battle(
        self,
        group: MovingGroup,
        node_id: str,
        objective: str,
        core_owner: Optional[str] = None,
        include_owner_units: bool = False,
        include_core: bool = False,
        include_neutral: bool = False,
        include_baron: bool = False,
        include_allied_lord: bool = False,
        include_visitors: bool = False,
        location_kind: str = "node",
        approach_id: Optional[str] = None,
        threat_zone_ids: Optional[list[str]] = None,
        edge_position: Optional[float] = None,
    ) -> dict:
        key = self._node_key(node_id)
        existing = self.state.active_battles.get(key)
        if existing:
            if objective == "neutral":
                self._aggro_neutrals(existing, group.player_id, node_id)
            return self._join_battle(existing, group, reinforced=True)

        node = self.state.nodes[node_id]
        battle = Battle(
            battle_id=key,
            location_kind=location_kind,
            location_id=node_id,
            units=[],
            started_at_tick=self.state.tick,
            last_tick=self.state.tick,
            objective=objective,
            core_owner=core_owner,
            approach_id=approach_id,
            threat_zone_ids=list(threat_zone_ids or []),
            edge_position=float(edge_position) if edge_position is not None else 0.5,
        )

        if include_owner_units and node.owner in self.state.players and node.units:
            owner_entry_side = node.unit_entry_side.get(node.owner)
            owner_visual_units = self._take_node_visual_units(node, node.owner, node.units)
            self._add_player_units_to_battle(
                battle,
                player_id=node.owner,
                units=node.units,
                src=node_id,
                dst=node_id,
                intent="DEFEND",
                entry_side=owner_entry_side,
                visual_units=owner_visual_units,
            )
            node.units = {}
            node.unit_entry_side.pop(node.owner, None)

        if include_visitors and node.visiting_units:
            for visitor_id, units in list(node.visiting_units.items()):
                visitor_entry_side = node.unit_entry_side.get(visitor_id)
                visitor_visual_units = self._take_node_visual_units(
                    node,
                    visitor_id,
                    units,
                    visiting=True,
                )
                self._add_player_units_to_battle(
                    battle,
                    player_id=visitor_id,
                    units=units,
                    src=node_id,
                    dst=node_id,
                    intent="VISIT",
                    entry_side=visitor_entry_side,
                    visual_units=visitor_visual_units,
                )
                node.unit_entry_side.pop(visitor_id, None)
            node.visiting_units.clear()
            node.visiting_visual_units.clear()

        if include_core and core_owner:
            core_unit = build_core_unit(core_owner, hp=self.state.players[core_owner].core_hp)
            self._assign_visual_ids_to_units(
                [core_unit],
                core_owner,
                {"CORE": [f"core:{core_owner}"]},
            )
            place_static_units_2d([core_unit], node_id, role="core", owner=core_owner)
            battle.units.append(core_unit)

        if include_neutral and node.neutral_hp > 0:
            neutral_units = build_neutral_units(node.neutral_hp)
            self._assign_visual_ids_to_units(
                neutral_units,
                "NEUTRAL",
                self._take_node_visual_units(
                    node,
                    "NEUTRAL",
                    {"NEUTRAL": int(node.neutral_hp)},
                ),
            )
            place_static_units_2d(neutral_units, node_id, role="neutral", owner="NEUTRAL")
            battle.units.extend(neutral_units)
            self._aggro_neutrals(battle, group.player_id, node_id)

        if include_baron and self.state.baron.hp > 0:
            baron_unit = build_baron_unit(self.state.baron.hp)
            self._assign_visual_ids_to_units(
                [baron_unit],
                "BARON",
                self._take_node_visual_units(node, "BARON", {"BARON": 1}),
            )
            place_static_units_2d([baron_unit], node_id, role="baron", owner="BARON")
            battle.units.append(baron_unit)

        if include_allied_lord and self._delayed_allied_lord_waiting_at_baron():
            owner = self.state.baron.allied_lord_owner
            if owner:
                self._add_player_units_to_battle(
                    battle,
                    player_id=owner,
                    units={UnitType.LORD.value: 1},
                    src=node_id,
                    dst=node_id,
                    intent="LORD_PUSH",
                    visual_units=self.state.baron.allied_lord_visual_units,
                )

        self._join_battle(battle, group, reinforced=False)
        self.state.active_battles[key] = battle

        return {
            "type": "BATTLE_STARTED",
            "battle_type": f"{battle.location_kind}_battle",
            "location_kind": battle.location_kind,
            "location_id": node_id,
            "battle_id": battle.battle_id,
            "objective": objective,
            "approach_id": battle.approach_id,
            "threat_zone_ids": list(battle.threat_zone_ids),
            "participants": sorted(self._battle_player_ids(battle)),
            "neutral_aggro_teams": sorted(battle.neutral_aggro_teams),
        }

    def _join_battle(self, battle: Battle, group: MovingGroup, reinforced: bool) -> dict:
        self._current_group = group
        try:
            self._add_player_units_to_battle(
                battle,
                player_id=group.player_id,
                units=group.units,
                src=group.src,
                dst=group.dst,
                intent=group.intent,
                visual_units=group.visual_units,
            )
        finally:
            self._current_group = None
        if battle.objective == "neutral" and group.intent.upper() in {"CAPTURE", "ATTACK"}:
            self._aggro_neutrals(battle, group.player_id, battle.location_id)

        return {
            "type": "BATTLE_REINFORCEMENT" if reinforced else "BATTLE_JOIN",
            "battle_type": f"{battle.location_kind}_battle",
            "location_kind": battle.location_kind,
            "location_id": battle.location_id,
            "battle_id": battle.battle_id,
            "approach_id": battle.approach_id,
            "threat_zone_ids": list(battle.threat_zone_ids),
            "player": group.player_id,
            "units": dict(group.units),
            "intent": group.intent,
        }

    def _add_player_units_to_battle(
        self,
        battle: Battle,
        player_id: str,
        units: dict,
        src: str,
        dst: str,
        intent: str,
        entry_side: Optional[str] = None,
        visual_units: Optional[dict] = None,
    ) -> None:
        if not units:
            return
        entry_side = entry_side or self._entry_side(player_id, src, dst)
        built_units = build_player_units(units, player_id, entry_side)
        self._apply_allied_lord_hp_to_units(player_id, built_units)
        self._assign_visual_ids_to_units(built_units, player_id, visual_units)
        current_group = getattr(self, "_current_group", None)
        group_progress = (
            getattr(current_group, "progress", None)
            if current_group is not None and getattr(current_group, "player_id", None) == player_id
            else None
        )
        if (
            current_group is not None
            and getattr(current_group, "player_id", None) == player_id
            and battle.location_kind == "approach"
        ):
            if getattr(current_group, "phase", None) == "node_enter":
                group_progress = 1.0
            elif getattr(current_group, "phase", None) == "node_exit":
                group_progress = 0.0
        place_src, place_dst = self._battle_placement_route_for_group(
            battle,
            current_group,
            player_id,
            src,
            dst,
        )
        place_player_units_2d(
            battle,
            built_units,
            player_id=player_id,
            src=place_src,
            dst=place_dst,
            entry_side=entry_side,
            progress=group_progress,
        )
        battle.units.extend(built_units)
        battle.participant_routes[player_id] = (src, dst)
        battle.participant_intents[player_id] = intent
        battle.participant_entry_sides[player_id] = entry_side
        if current_group is not None and getattr(current_group, "player_id", None) == player_id:
            battle.participant_paths[player_id] = list(getattr(current_group, "path", []) or [src, dst])
            battle.participant_segment_indices[player_id] = getattr(current_group, "segment_index", 0)
            battle.participant_speeds[player_id] = getattr(current_group, "speed", 1.0)
            battle.participant_final_dsts[player_id] = getattr(current_group, "final_dst", dst) or dst

    def _battle_placement_route_for_group(
        self,
        battle: Battle,
        group: Optional[MovingGroup],
        player_id: str,
        src: str,
        dst: str,
    ) -> tuple[str, str]:
        if (
            group is None
            or getattr(group, "player_id", None) != player_id
            or getattr(group, "phase", None) != "node_transit"
            or not getattr(group, "segment_node", None)
            or str(getattr(group, "segment_node")) != battle.location_id
        ):
            return src, dst

        segment_node = str(group.segment_node)
        if float(getattr(group, "progress", 0.0) or 0.0) <= 0.5:
            return str(group.src), segment_node
        return segment_node, str(group.dst)

    def _apply_allied_lord_hp_to_units(self, player_id: str, units: list) -> None:
        if (
            self.state.baron.allied_lord_owner != player_id
            or not self.state.baron.allied_lord_alive
        ):
            return
        for unit in units:
            if unit.unit_type == UnitType.LORD.value:
                unit.hp = max(0.0, float(self.state.baron.allied_lord_hp))
                unit.max_hp = float(self.state.baron.max_hp)

    def _sync_allied_lord_after_battle(self, battle: Battle) -> None:
        owner = self.state.baron.allied_lord_owner
        if not owner or not self.state.baron.allied_lord_alive:
            return
        lord_units = [
            unit for unit in battle.units
            if unit.team == owner and unit.unit_type == UnitType.LORD.value
        ]
        if not lord_units:
            return
        alive_hp = [max(0.0, unit.hp) for unit in lord_units if unit.is_alive()]
        if alive_hp:
            self.state.baron.allied_lord_hp = max(alive_hp)
        else:
            self._kill_allied_lord(expended=False)

    def _refresh_active_lord_target(self, battle: Battle) -> Optional[dict]:
        owner = self.state.baron.allied_lord_owner
        if not owner or not self.state.baron.allied_lord_alive:
            return None

        lord_units = [
            unit for unit in battle.units
            if unit.team == owner and unit.unit_type == UnitType.LORD.value and unit.is_alive()
        ]
        if not lord_units:
            return None

        previous_target = self.state.baron.allied_lord_target
        target = self._living_lord_target(owner)
        if target is None:
            for unit in lord_units:
                unit.hp = 0.0
            self._kill_allied_lord(expended=False)
            return {
                "type": "LORD_STOPPED",
                "player": owner,
                "src": battle.location_id,
                "reason": "no_living_enemy_core",
            }
        if target == previous_target:
            return None

        battle.participant_final_dsts[owner] = target
        src, dst = battle.participant_routes.get(owner, (None, None))
        if src and dst:
            path = [src, dst]
            if dst != target:
                tail = self._find_unrestricted_route(dst, target)
                if tail and len(tail) >= 2:
                    path = [src, dst] + tail[1:]
            battle.participant_paths[owner] = path

        return {
            "type": "LORD_RETARGETED",
            "player": owner,
            "src": battle.location_id,
            "target": target,
            "previous_target": previous_target,
            "path": list(battle.participant_paths.get(owner, [])),
            "reason": "active_target_destroyed",
        }

    def _kill_allied_lord(self, expended: bool) -> None:
        self.state.baron.allied_lord_hp = 0.0
        self.state.baron.allied_lord_alive = False
        self.state.baron.allied_lord_expended = expended
        self.state.baron.allied_lord_target = None
        self.state.baron.allied_lord_path = []
        self.state.baron.allied_lord_ready_tick = 0
        self.state.baron.allied_lord_arrival_tick = 0
        self.state.baron.allied_lord_visual_units = {}
        self.state.baron.allied_lord_support_units = {}
        self.state.baron.allied_lord_support_visual_units = {}
        owner = self.state.baron.allied_lord_owner
        if not owner:
            return
        for node in self.state.nodes.values():
            if node.units.get(UnitType.LORD.value):
                node.units.pop(UnitType.LORD.value, None)
                node.visual_units.get(owner, {}).pop(UnitType.LORD.value, None)
            visitors = node.visiting_units.get(owner)
            if visitors and visitors.get(UnitType.LORD.value):
                visitors.pop(UnitType.LORD.value, None)
                node.visiting_visual_units.get(owner, {}).pop(UnitType.LORD.value, None)
        for group in list(self.state.moving_groups):
            if group.player_id == owner and group.units.get(UnitType.LORD.value):
                group.units.pop(UnitType.LORD.value, None)
                group.visual_units.pop(UnitType.LORD.value, None)
        self.state.moving_groups = [group for group in self.state.moving_groups if unit_total(group.units) > 0]

    def _aggro_neutrals(self, battle: Battle, player_id: str, node_id: str) -> None:
        team = get_team(player_id)
        battle.neutral_aggro_teams.add(team)
        if node_id in self.state.nodes:
            self.state.nodes[node_id].neutral_aggro_teams.add(team)

    # ------------------------------------------------------------------
    # Battle ticking and resolution
    # ------------------------------------------------------------------

    def _tick_active_battles(self, tick_events: list) -> None:
        for key, battle in list(self.state.active_battles.items()):
            lord_event = self._refresh_active_lord_target(battle)
            if lord_event:
                tick_events.append(lord_event)

            before = self._alive_unit_index(battle.units) if self.record_trace else {}
            frame = tick_battle_units_2d(
                battle.units,
                record_trace=self.record_trace,
                neutral_aggro_teams=battle.neutral_aggro_teams,
                battle=battle,
            )
            self._sync_allied_lord_after_battle(battle)
            battle.last_tick = self.state.tick

            if self.record_trace:
                tick_events.append({
                    "type": "BATTLE_TICK",
                    "battle_type": f"{battle.location_kind}_battle",
                    "location_kind": battle.location_kind,
                    "location_id": battle.location_id,
                    "battle_id": battle.battle_id,
                    "approach_id": battle.approach_id,
                    "threat_zone_ids": list(battle.threat_zone_ids),
                    "summary": self._battle_tick_summary(battle, frame, before),
                })

            timed_out = self.state.tick - battle.started_at_tick >= MAX_COMBAT_TICKS
            if timed_out:
                self._force_timeout_resolution(battle)

            if timed_out or not has_hostile_units_2d(battle.units, battle.neutral_aggro_teams):
                resolution = (
                    self._finish_edge_battle(battle)
                    if battle.location_kind == "edge"
                    else self._finish_node_battle(battle)
                )
                tick_events.append(resolution)
                del self.state.active_battles[key]

    def _finish_node_battle(self, battle: Battle) -> dict:
        node = self.state.nodes[battle.location_id]
        self._sync_allied_lord_after_battle(battle)
        survivors_by_player = count_survivors_by_player(battle.units)
        visual_survivors_by_owner = self._visual_survivors_by_owner(battle.units)
        winner_player = self._strongest_survivor(survivors_by_player)

        if battle.objective == "core":
            core_alive = any(u.is_alive() and u.unit_type == "CORE" for u in battle.units)
            core_owner = battle.core_owner or battle.location_id
            previous_core_hp = float(self.state.players[core_owner].core_hp)
            remaining_core_hp = sum(
                max(0.0, unit.hp)
                for unit in battle.units
                if unit.unit_type == "CORE" and unit.is_alive()
            )
            self.state.players[core_owner].core_hp = max(0.0, remaining_core_hp)
            self._record_core_combat_damage(
                core_owner,
                previous_core_hp - self.state.players[core_owner].core_hp,
            )
            lord_expended = False
            if not core_alive:
                survivor_units = (
                    dict(survivors_by_player.get(winner_player, {}))
                    if winner_player
                    else {}
                )
                if winner_player and survivor_units.pop(UnitType.LORD.value, 0) > 0:
                    lord_expended = True
                    self._kill_allied_lord(expended=True)
                if winner_player:
                    survivors_by_player[winner_player] = survivor_units
                    visual_survivors_by_owner[winner_player] = self._trim_visual_units_to_counts(
                        visual_survivors_by_owner.get(winner_player, {}),
                        winner_player,
                        survivor_units,
                    )
                self.state.players[core_owner].alive = False
                self.state.players[core_owner].core_hp = 0.0
                self._remove_player_units(core_owner)
                self._neutralize_destroyed_core_node(core_owner)
                if winner_player and winner_player != core_owner and unit_total(survivor_units) > 0:
                    self._merge_visiting_units(
                        node,
                        winner_player,
                        survivor_units,
                        entry_side=self._battle_entry_side(battle, winner_player),
                        visual_units=visual_survivors_by_owner.get(winner_player, {}),
                    )
                    self._lock_node_player(node, winner_player)
            else:
                node.owner = core_owner
                self._set_node_units(
                    node,
                    core_owner,
                    survivors_by_player.get(core_owner, {}),
                    self._battle_entry_side(battle, core_owner),
                    visual_units=visual_survivors_by_owner.get(core_owner, {}),
                )
                self._lock_node_player(node, core_owner)
            return self._resolution_event(
                battle,
                winner_player,
                {
                    "core_destroyed": not core_alive,
                    "core_owner": core_owner,
                    "core_hp": round(self.state.players[core_owner].core_hp, 3),
                    "lord_expended": lord_expended,
                },
            )

        if battle.objective == "neutral":
            remaining_neutral_hp = sum(
                1
                for unit in battle.units
                if unit.unit_type == "NEUTRAL" and unit.is_alive()
            )
            node.neutral_aggro_teams = set(battle.neutral_aggro_teams)
            if remaining_neutral_hp > 0 or not winner_player:
                node.owner = "NEUTRAL"
                node.units = {}
                if self.record_replay:
                    node.visual_units.clear()
                    neutral_visual_units = self._trim_visual_units_to_counts(
                        visual_survivors_by_owner.get("NEUTRAL", {}),
                        "NEUTRAL",
                        {"NEUTRAL": remaining_neutral_hp},
                    )
                    if neutral_visual_units:
                        node.visual_units["NEUTRAL"] = neutral_visual_units
                self._forget_battle_entry_sides(node, battle)
                node.neutral_hp = max(0, remaining_neutral_hp)
                return self._resolution_event(
                    battle,
                    "NEUTRAL",
                    {"result": "REPELLED", "remaining_neutral_hp": node.neutral_hp},
                )

            node.owner = winner_player
            node.neutral_hp = 0
            self._set_node_units(
                node,
                winner_player,
                survivors_by_player.get(winner_player, {}),
                self._battle_entry_side(battle, winner_player),
                visual_units=visual_survivors_by_owner.get(winner_player, {}),
            )
            self._lock_node_player(node, winner_player)
            return self._resolution_event(battle, winner_player, {"result": "CAPTURED"})

        if battle.objective == "baron":
            baron_alive = any(u.is_alive() and u.unit_type == "BARON" for u in battle.units)
            remaining_baron_hp = int(sum(
                max(0.0, unit.hp)
                for unit in battle.units
                if unit.unit_type == "BARON" and unit.is_alive()
            ))
            if not baron_alive and winner_player:
                self.state.baron.controlled_by = winner_player
                self.state.baron.hp = 0
                self.state.baron.ticks_remaining = 30
                node.owner = winner_player
                self._set_node_units(
                    node,
                    winner_player,
                    survivors_by_player.get(winner_player, {}),
                    self._battle_entry_side(battle, winner_player),
                    visual_units=visual_survivors_by_owner.get(winner_player, {}),
                )
                self._lock_node_player(node, winner_player)
                lord_visual_units = self._visual_ids_from_battle_units(
                    battle,
                    "BARON",
                    "BARON",
                    UnitType.LORD.value,
                )
                lord_event = self._spawn_lord_push(
                    winner_player,
                    "Baron",
                    visual_units=lord_visual_units,
                )
            else:
                self.state.baron.hp = max(0, remaining_baron_hp)
                node.owner = "NEUTRAL"
                node.units = {}
                if self.record_replay:
                    node.visual_units.clear()
                    if remaining_baron_hp > 0:
                        baron_visual_units = self._trim_visual_units_to_counts(
                            visual_survivors_by_owner.get("BARON", {}),
                            "BARON",
                            {"BARON": 1},
                        )
                        if baron_visual_units:
                            node.visual_units["BARON"] = baron_visual_units
                self._forget_battle_entry_sides(node, battle)
                lord_event = None
            extra = {"baron_killed": not baron_alive}
            if lord_event:
                extra["lord_spawned"] = lord_event
            return self._resolution_event(battle, winner_player or "BARON", extra)

        if self._is_destroyed_core_node(battle.location_id):
            self._neutralize_destroyed_core_node(battle.location_id)
            lord_continue = None
            if winner_player:
                survivor_units = dict(survivors_by_player.get(winner_player, {}))
                winner_visual_units = visual_survivors_by_owner.get(winner_player, {})
                lord_count = int(survivor_units.pop(UnitType.LORD.value, 0) or 0)
                if lord_count > 0 and battle.participant_intents.get(winner_player) == "LORD_PUSH":
                    lord_visual_units = self._trim_visual_units_to_counts(
                        winner_visual_units,
                        winner_player,
                        {UnitType.LORD.value: 1},
                    )
                    lord_continue = self._retarget_lord_push_from_node(
                        player_id=winner_player,
                        node_id=battle.location_id,
                        visual_units=lord_visual_units,
                        reason="dead_core_pvp_resolved",
                    )
                    visual_survivors_by_owner[winner_player] = self._trim_visual_units_to_counts(
                        winner_visual_units,
                        winner_player,
                        survivor_units,
                    )
                if unit_total(survivor_units) > 0:
                    self._merge_visiting_units(
                        node,
                        winner_player,
                        survivor_units,
                        entry_side=self._battle_entry_side(battle, winner_player),
                        visual_units=visual_survivors_by_owner.get(winner_player, {}),
                    )
                    self._lock_node_player(node, winner_player)
            else:
                node.units = {}
                node.visual_units.clear()
                self._forget_battle_entry_sides(node, battle)
            return self._resolution_event(
                battle,
                winner_player,
                {
                    "result": "DEAD_CORE_PVP_RESOLVED",
                    **({"lord_continued": lord_continue} if lord_continue else {}),
                },
            )

        if winner_player:
            if node.owner == "NEUTRAL" and node.neutral_hp > 0:
                self._merge_visiting_units(
                    node,
                    winner_player,
                    survivors_by_player.get(winner_player, {}),
                    entry_side=self._battle_entry_side(battle, winner_player),
                    visual_units=visual_survivors_by_owner.get(winner_player, {}),
                )
                self._lock_node_player(node, winner_player)
            else:
                node.owner = winner_player
                self._set_node_units(
                    node,
                    winner_player,
                    survivors_by_player.get(winner_player, {}),
                    self._battle_entry_side(battle, winner_player),
                    visual_units=visual_survivors_by_owner.get(winner_player, {}),
                )
                self._lock_node_player(node, winner_player)
            lord_continue = self._continue_lord_push(battle, winner_player)
        else:
            node.units = {}
            node.visual_units.clear()
            self._forget_battle_entry_sides(node, battle)
            lord_continue = None

        extra = {"result": "PVP_RESOLVED"}
        if lord_continue:
            extra["lord_continued"] = lord_continue
        return self._resolution_event(battle, winner_player, extra)

    def _finish_edge_battle(self, battle: Battle) -> dict:
        self._sync_allied_lord_after_battle(battle)
        survivors_by_player = count_survivors_by_player(battle.units)
        visual_survivors_by_owner = self._visual_survivors_by_owner(battle.units)
        continuing = []

        for player_id, units in survivors_by_player.items():
            src, dst = battle.participant_routes.get(player_id, (None, None))
            if src is None or dst is None:
                continue
            path = list(battle.participant_paths.get(player_id, [src, dst]))
            speed = float(battle.participant_speeds.get(player_id, self._group_speed(units)))
            segment_index = int(battle.participant_segment_indices.get(player_id, 0))
            continue_progress = self._edge_progress_for_participant(battle, src, dst)
            group = MovingGroup(
                player_id=player_id,
                team=get_team(player_id),
                units=units,
                src=src,
                dst=dst,
                arrival_tick=self._estimate_arrival_tick(
                    self.state.tick,
                    path,
                    speed,
                    segment_index,
                    continue_progress,
                ),
                started_tick=self.state.tick,
                travel_ticks=float(TRAVEL_TIME) / speed,
                progress=continue_progress,
                intent=battle.participant_intents.get(player_id, "CAPTURE"),
                path=path,
                segment_index=segment_index,
                phase="edge",
                segment_node=None,
                speed=speed,
                final_dst=battle.participant_final_dsts.get(player_id, dst),
                visual_units=self._trim_visual_units_to_counts(
                    visual_survivors_by_owner.get(player_id, {}),
                    player_id,
                    units,
                ),
            )
            self._set_group_segment(group, segment_index, progress=continue_progress)
            self.state.moving_groups.append(group)
            continuing.append({"player": player_id, "src": src, "dst": dst, "path": path, "units": units})

        return self._resolution_event(battle, self._strongest_survivor(survivors_by_player), {
            "result": "EDGE_RESOLVED",
            "continuing_groups": continuing,
        })

    def _edge_progress_for_participant(self, battle: Battle, src: str, dst: str) -> float:
        progress = max(0.0, min(1.0, float(battle.edge_position or 0.0)))
        endpoints = battle.edge_endpoints
        if not endpoints:
            return progress
        edge_start, edge_end = endpoints
        if src == edge_start and dst == edge_end:
            return progress
        if src == edge_end and dst == edge_start:
            return 1.0 - progress
        return progress

    def _force_timeout_resolution(self, battle: Battle) -> None:
        hp_by_side: dict[str, float] = {}
        for unit in battle.units:
            if not unit.is_alive():
                continue
            side = self._unit_resolution_side(unit)
            hp_by_side[side] = hp_by_side.get(side, 0.0) + unit.hp

        if not hp_by_side:
            return
        winner_side = max(hp_by_side, key=hp_by_side.get)
        for unit in battle.units:
            if unit.is_alive() and self._unit_resolution_side(unit) != winner_side:
                unit.hp = 0

    def _unit_resolution_side(self, unit) -> str:
        if unit.unit_type == "NEUTRAL":
            return "NEUTRAL"
        if unit.unit_type == "BARON":
            return "BARON"
        return get_team(unit.team)

    def _resolution_event(self, battle: Battle, winner, extra: dict) -> dict:
        event = {
            "type": "BATTLE_RESOLVED",
            "battle_type": f"{battle.location_kind}_battle",
            "location_kind": battle.location_kind,
            "location_id": battle.location_id,
            "battle_id": battle.battle_id,
            "approach_id": battle.approach_id,
            "threat_zone_ids": list(battle.threat_zone_ids),
            "winner": winner,
            "participants": sorted(self._battle_player_ids(battle)),
        }
        event.update(extra)
        return event

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _remove_player_units(self, player_id: str) -> None:
        state = self.state

        for node in state.nodes.values():
            if node.owner == player_id:
                node.owner = "NEUTRAL"
                node.units = {}
                node.visual_units.pop(player_id, None)
                node.unit_ready_tick.pop(player_id, None)
                node.unit_entry_side.pop(player_id, None)
            node.visiting_units.pop(player_id, None)
            node.visiting_visual_units.pop(player_id, None)
            node.unit_ready_tick.pop(player_id, None)
            node.unit_entry_side.pop(player_id, None)

        state.moving_groups = [
            group for group in state.moving_groups if group.player_id != player_id
        ]
        self._remove_player_visual_units(player_id)

        for battle in state.active_battles.values():
            for unit in battle.units:
                if unit.team == player_id:
                    unit.hp = 0

    def _is_destroyed_core_node(self, node_id: str) -> bool:
        if NODE_TYPES.get(node_id) != NodeType.CORE:
            return False
        player = self.state.players.get(node_id)
        return player is None or not player.alive or player.core_hp <= 0

    def _neutralize_destroyed_core_node(self, node_id: str) -> None:
        if not self._is_destroyed_core_node(node_id):
            return

        node = self.state.nodes[node_id]
        previous_owner = node.owner
        previous_units = dict(node.units or {})
        previous_entry_side = node.unit_entry_side.get(previous_owner)
        previous_visual_units = self._take_node_visual_units(
            node,
            previous_owner,
            previous_units,
        )

        node.owner = "NEUTRAL"
        node.units = {}
        node.visual_units.pop(previous_owner, None)
        node.unit_ready_tick.pop(node_id, None)
        node.unit_entry_side.pop(node_id, None)

        if (
            previous_owner in self.state.players
            and previous_owner != node_id
            and unit_total(previous_units) > 0
        ):
            self._merge_visiting_units(
                node,
                previous_owner,
                previous_units,
                entry_side=previous_entry_side,
                visual_units=previous_visual_units,
            )

    def _return_group_to_source(self, group: MovingGroup) -> None:
        src_node = self.state.nodes[group.src]
        entry_side = self._entry_side(group.player_id, group.dst, group.src)
        if src_node.owner == group.player_id:
            self._merge_units(
                src_node,
                group.units,
                player_id=group.player_id,
                entry_side=entry_side,
                visual_units=group.visual_units,
            )
        else:
            self._merge_visiting_units(
                src_node,
                group.player_id,
                group.units,
                entry_side=entry_side,
                visual_units=group.visual_units,
            )
        self._lock_node_player(src_node, group.player_id)

    def _set_node_units(
        self,
        node: NodeState,
        player_id: Optional[str],
        units: dict,
        entry_side: Optional[str] = None,
        visual_units: Optional[dict] = None,
    ) -> None:
        node.units = dict(units or {})
        if self.record_replay:
            node.visual_units.clear()
            if player_id and unit_total(node.units) > 0:
                bucket = node.visual_units.setdefault(player_id, {})
                self._merge_visual_units_into_bucket(bucket, player_id, node.units, visual_units)
        if unit_total(node.units) > 0:
            self._remember_node_entry_side(node, player_id, entry_side)
        elif player_id:
            node.unit_entry_side.pop(player_id, None)

    def _merge_units(
        self,
        node: NodeState,
        units: dict,
        player_id: Optional[str] = None,
        entry_side: Optional[str] = None,
        visual_units: Optional[dict] = None,
    ) -> None:
        for unit_type, count in units.items():
            node.units[unit_type] = node.units.get(unit_type, 0) + count
        if self.record_replay and unit_total(units) > 0:
            visual_owner = node.owner if node.owner in self.state.players else player_id
            if visual_owner:
                bucket = node.visual_units.setdefault(visual_owner, {})
                self._merge_visual_units_into_bucket(bucket, visual_owner, units, visual_units)
        if unit_total(units) > 0:
            self._remember_node_entry_side(node, player_id or node.owner, entry_side)

    def _merge_visiting_units(
        self,
        node: NodeState,
        player_id: str,
        units: dict,
        entry_side: Optional[str] = None,
        visual_units: Optional[dict] = None,
    ) -> None:
        bucket = node.visiting_units.setdefault(player_id, {})
        for unit_type, count in units.items():
            bucket[unit_type] = bucket.get(unit_type, 0) + count
        if self.record_replay and unit_total(units) > 0:
            visual_bucket = node.visiting_visual_units.setdefault(player_id, {})
            self._merge_visual_units_into_bucket(visual_bucket, player_id, units, visual_units)
        if unit_total(units) > 0:
            self._remember_node_entry_side(node, player_id, entry_side)

    def _remember_node_entry_side(
        self,
        node: NodeState,
        player_id: Optional[str],
        entry_side: Optional[str],
    ) -> None:
        if not player_id or player_id not in self.state.players:
            return
        node.unit_entry_side[player_id] = entry_side or self._default_player_side(player_id)

    def _forget_battle_entry_sides(self, node: NodeState, battle: Battle) -> None:
        for player_id in battle.participant_entry_sides:
            node.unit_entry_side.pop(player_id, None)

    def _battle_entry_side(self, battle: Battle, player_id: Optional[str]) -> Optional[str]:
        if not player_id:
            return None
        return battle.participant_entry_sides.get(player_id, self._default_player_side(player_id))

    def _lock_node_player(self, node: NodeState, player_id: Optional[str]) -> None:
        if not player_id or player_id not in self.state.players:
            return
        node.unit_ready_tick[player_id] = self.state.tick + 1

    def _has_enemy_visitors(self, node: NodeState, player_id: str) -> bool:
        player_team = get_team(player_id)
        return any(get_team(visitor_id) != player_team for visitor_id in node.visiting_units)

    def _node_owner_is_hostile(self, node: NodeState, player_id: str) -> bool:
        if node.owner not in self.state.players or not node.units:
            return False
        return get_team(node.owner) != get_team(player_id)

    def _node_owner_should_join_entry_battle(
        self,
        node: NodeState,
        player_id: str,
        has_enemy_visitors: bool,
    ) -> bool:
        if node.owner not in self.state.players or not node.units:
            return False
        return has_enemy_visitors or get_team(node.owner) != get_team(player_id)

    def _node_has_hostile_occupants(self, node: NodeState, player_id: str) -> bool:
        return self._node_owner_is_hostile(node, player_id) or self._has_enemy_visitors(node, player_id)

    def _hostile_ranged_range(self, node: NodeState, player_id: str) -> float:
        player_team = get_team(player_id)
        max_range = 0.0
        if node.owner in self.state.players and get_team(node.owner) != player_team:
            max_range = max(max_range, self._max_ranged_range(node.units))
        for visitor_id, units in (node.visiting_units or {}).items():
            if get_team(visitor_id) != player_team:
                max_range = max(max_range, self._max_ranged_range(units))
        return max_range

    def _max_ranged_range(self, units: dict) -> float:
        max_range = 0.0
        for unit_type, count in (units or {}).items():
            if int(count or 0) <= 0:
                continue
            unit_key = unit_type.value if isinstance(unit_type, UnitType) else unit_type
            try:
                unit_enum = UnitType.RANGER if unit_key == "ARCHER" else UnitType(unit_key)
            except ValueError:
                continue
            unit_range = float(UNIT_STATS[unit_enum]["range"])
            if unit_range > NODE_RANGED_MIN_RANGE:
                max_range = max(max_range, unit_range)
        return max_range

    def _battle_player_ids(self, battle: Battle) -> set[str]:
        return {
            unit.team
            for unit in battle.units
            if unit.unit_type not in {"NEUTRAL", "CORE", "BARON"}
        }

    def _strongest_survivor(self, survivors_by_player: dict[str, dict]) -> Optional[str]:
        if not survivors_by_player:
            return None
        return max(survivors_by_player, key=lambda pid: sum(survivors_by_player[pid].values()))

    def _entry_side(self, player_id: str, src: str, dst: str) -> str:
        if src == dst:
            return self._default_player_side(player_id)
        return "attacker" if self._node_order(src) <= self._node_order(dst) else "defender"

    def _default_player_side(self, player_id: str) -> str:
        return "attacker" if get_team(player_id) == "TEAM_A" else "defender"

    def _node_order(self, node_id: str) -> int:
        order = {
            "A1": 0, "A2": 0, "A3": 0,
            "CP1": 1, "CP2": 1,
            "CP3": 2, "CP4": 2, "Baron": 2,
            "CP5": 3, "CP6": 3,
            "B1": 4, "B2": 4, "B3": 4,
        }
        return order.get(node_id, 0)

    def _node_key(self, node_id: str) -> str:
        return f"node:{node_id}"

    def _canonical_edge(self, a: str, b: str) -> tuple[str, str]:
        return tuple(sorted((a, b)))

    def _edge_id(self, a: str, b: str) -> str:
        left, right = self._canonical_edge(a, b)
        return f"{left}|{right}"

    def _edge_key(self, a: str, b: str) -> str:
        return f"edge:{self._edge_id(a, b)}"

    def _record_core_combat_damage(self, core_owner: str, amount: float) -> None:
        if amount <= 0:
            return
        damaged_team = get_team(core_owner)
        source_team = get_enemy_team(damaged_team)
        self.state.core_combat_damage += float(amount)
        self.state.core_damage_by_team[source_team] = (
            self.state.core_damage_by_team.get(source_team, 0.0) + float(amount)
        )

    def _apply_death_time_damage(self, tick_events: list) -> None:
        if self.state.tick < DEATH_TIME_START_TICK or self.state.tick >= TICK_LIMIT:
            return
        for player_id, player in list(self.state.players.items()):
            if not player.alive or player.core_hp <= 0:
                continue
            before = float(player.core_hp)
            player.core_hp = max(0.0, before - death_time_core_damage_for_tick(self.state.tick))
            damage = before - player.core_hp
            if damage <= 0:
                continue
            event = {
                "type": "DEATH_TIME_CORE_DAMAGE",
                "player": player_id,
                "team": player.team,
                "damage": round(damage, 3),
                "core_hp": round(player.core_hp, 3),
            }
            if player.core_hp <= 0:
                self._destroy_core_by_environment(player_id)
                event["core_destroyed"] = True
            tick_events.append(event)

    def _destroy_core_by_environment(self, player_id: str) -> None:
        player = self.state.players[player_id]
        if not player.alive:
            return
        player.alive = False
        player.core_hp = 0.0
        self._remove_player_units(player_id)

    def _team_core_hp(self, team: str) -> float:
        return sum(
            max(0.0, float(player.core_hp))
            for player in self.state.players.values()
            if player.team == team
        )

    def _alive_core_count(self, team: str) -> int:
        return sum(
            1
            for player in self.state.players.values()
            if player.team == team and player.alive and player.core_hp > 0
        )

    def _death_time_winner_from_scores(
        self,
        final_deaths: Optional[dict[str, int]] = None,
        hp_before: Optional[dict[str, float]] = None,
    ) -> str:
        final_deaths = final_deaths or {
            "TEAM_A": self._alive_core_count("TEAM_A"),
            "TEAM_B": self._alive_core_count("TEAM_B"),
        }
        if final_deaths.get("TEAM_A", 0) > final_deaths.get("TEAM_B", 0):
            return "TEAM_A"
        if final_deaths.get("TEAM_B", 0) > final_deaths.get("TEAM_A", 0):
            return "TEAM_B"

        hp_before = hp_before or {
            "TEAM_A": self._team_core_hp("TEAM_A"),
            "TEAM_B": self._team_core_hp("TEAM_B"),
        }
        if hp_before["TEAM_A"] > hp_before["TEAM_B"]:
            return "TEAM_A"
        if hp_before["TEAM_B"] > hp_before["TEAM_A"]:
            return "TEAM_B"

        dealt = self.state.core_damage_by_team
        if dealt.get("TEAM_A", 0.0) > dealt.get("TEAM_B", 0.0):
            return "TEAM_A"
        if dealt.get("TEAM_B", 0.0) > dealt.get("TEAM_A", 0.0):
            return "TEAM_B"

        return "TEAM_A"

    def _finalize_death_time(self) -> None:
        hp_before = {
            "TEAM_A": self._team_core_hp("TEAM_A"),
            "TEAM_B": self._team_core_hp("TEAM_B"),
        }
        final_deaths = {"TEAM_A": 0, "TEAM_B": 0}
        for player_id, player in list(self.state.players.items()):
            if player.alive and player.core_hp > 0:
                final_deaths[player.team] += 1
                self._destroy_core_by_environment(player_id)
        self.state.death_time_final_core_deaths = final_deaths

        if self.state.core_combat_damage <= 0:
            self.state.winner = "DRAW"
            self.state.winner_type = "draw_no_core_damage"
            return

        self.state.winner = self._death_time_winner_from_scores(final_deaths, hp_before)
        self.state.winner_type = "death_time"

    def _check_victory(self) -> None:
        state = self.state
        a_alive = any(p.alive for p in state.players.values() if p.team == "TEAM_A")
        b_alive = any(p.alive for p in state.players.values() if p.team == "TEAM_B")

        if not a_alive and not b_alive:
            if state.tick >= DEATH_TIME_START_TICK and state.core_combat_damage > 0:
                state.winner = self._death_time_winner_from_scores()
                state.winner_type = "death_time"
            else:
                state.winner = "DRAW"
                state.winner_type = "draw_no_core_damage"
        elif not a_alive:
            state.winner = "TEAM_B"
            state.winner_type = "death_time" if state.tick >= DEATH_TIME_START_TICK else "normal"
        elif not b_alive:
            state.winner = "TEAM_A"
            state.winner_type = "death_time" if state.tick >= DEATH_TIME_START_TICK else "normal"

    # ------------------------------------------------------------------
    # Observation and replay
    # ------------------------------------------------------------------

    def _visual_unit_snapshots(self, state: GameState) -> list[dict]:
        if not self.record_replay:
            return []

        rows: list[dict] = []
        active_node_locations = {
            battle.location_id
            for battle in state.active_battles.values()
            if battle.location_kind in {"node", "approach"}
        }

        for node_id, node in state.nodes.items():
            if node_id in active_node_locations:
                continue
            anchor = NODE_LAYOUT[node_id].center
            heading = self._visual_heading_for_owner(node.owner)

            if node.owner in state.players and node.units:
                bucket = self._ensure_node_visual_units(node, node.owner, node.units)
                rows.extend(self._visual_rows_from_bucket(
                    bucket,
                    owner=node.owner,
                    anchor=anchor,
                    heading=heading,
                    layer="stationed",
                    node_id=node_id,
                ))

            for visitor_id, units in sorted((node.visiting_units or {}).items()):
                bucket = self._ensure_node_visual_units(node, visitor_id, units, visiting=True)
                rows.extend(self._visual_rows_from_bucket(
                    bucket,
                    owner=visitor_id,
                    anchor=anchor,
                    heading=self._visual_heading_for_owner(visitor_id),
                    layer="visiting",
                    node_id=node_id,
                ))

            if node.neutral_hp > 0:
                bucket = self._ensure_node_visual_units(
                    node,
                    "NEUTRAL",
                    {"NEUTRAL": int(node.neutral_hp)},
                )
                rows.extend(self._visual_rows_from_bucket(
                    bucket,
                    owner="NEUTRAL",
                    anchor=anchor,
                    heading=(1.0, 0.0),
                    layer="neutral",
                    node_id=node_id,
                ))

            if node_id == "Baron" and state.baron.hp > 0 and node.owner == "NEUTRAL":
                bucket = self._ensure_node_visual_units(node, "BARON", {"BARON": 1})
                rows.extend(self._visual_rows_from_bucket(
                    bucket,
                    owner="BARON",
                    anchor=anchor,
                    heading=(1.0, 0.0),
                    layer="neutral",
                    node_id=node_id,
                    hp_override=float(state.baron.hp),
                    max_hp_override=float(state.baron.max_hp),
                ))

            if (
                node_id == "Baron"
                and state.baron.allied_lord_alive
                and state.baron.allied_lord_owner
                and int(state.baron.allied_lord_ready_tick or 0) > state.tick
                and float(state.baron.allied_lord_hp or 0.0) > 0.0
            ):
                owner = state.baron.allied_lord_owner
                self._ensure_visual_bucket(
                    state.baron.allied_lord_visual_units,
                    owner,
                    {UnitType.LORD.value: 1},
                )
                rows.extend(self._visual_rows_from_bucket(
                    state.baron.allied_lord_visual_units,
                    owner=owner,
                    anchor=anchor,
                    heading=self._visual_heading_for_owner(owner),
                    layer="lord_delay",
                    node_id=node_id,
                    hp_override=float(state.baron.allied_lord_hp),
                    max_hp_override=float(state.baron.max_hp),
                ))

        for group in state.moving_groups:
            if unit_total(group.units) <= 0:
                continue
            self._ensure_visual_bucket(group.visual_units, group.player_id, group.units)
            try:
                visual = moving_group_visual(group)
                anchor = (float(visual["x"]), float(visual["y"]))
                heading = tuple(visual.get("heading") or [1.0, 0.0])
            except (KeyError, ValueError):
                continue
            rows.extend(self._visual_rows_from_bucket(
                group.visual_units,
                owner=group.player_id,
                anchor=anchor,
                heading=(float(heading[0]), float(heading[1])),
                layer="moving",
                group=group,
            ))

        for battle in state.active_battles.values():
            for unit in battle.units:
                if not unit.is_alive():
                    continue
                if not getattr(unit, "visual_id", ""):
                    unit.visual_id = self._new_visual_id(unit.team, unit.unit_type)
                rows.append({
                    "id": unit.visual_id,
                    "owner": unit.team,
                    "team": unit.team,
                    "unit_type": unit.unit_type,
                    "type": self._unit_type_code(unit.unit_type),
                    "x": round(float(unit.x), 3),
                    "y": round(float(unit.y), 3),
                    "hp": round(max(0.0, float(unit.hp)), 3),
                    "max_hp": round(max(0.0, float(unit.max_hp)), 3),
                    "hp_ratio": self._visual_hp_ratio(unit.hp, unit.max_hp),
                    "state": "fighting",
                    "layer": "battle",
                    "battle_id": battle.battle_id,
                    "location_kind": battle.location_kind,
                    "location_id": battle.location_id,
                    "uid": unit.uid,
                })
            if battle.objective == "baron":
                dead_baron = next(
                    (
                        unit for unit in battle.units
                        if unit.unit_type == "BARON"
                        and not unit.is_alive()
                        and getattr(unit, "visual_id", "")
                    ),
                    None,
                )
                if dead_baron is not None:
                    rows.append({
                        "id": dead_baron.visual_id,
                        "owner": "BARON",
                        "team": "BARON",
                        "unit_type": "BARON",
                        "type": self._unit_type_code("BARON"),
                        "x": round(float(dead_baron.x), 3),
                        "y": round(float(dead_baron.y), 3),
                        "hp": 0.0,
                        "max_hp": round(float(dead_baron.max_hp), 3),
                        "hp_ratio": 0.0,
                        "state": "fallen",
                        "layer": "battle",
                        "battle_id": battle.battle_id,
                        "location_kind": battle.location_kind,
                        "location_id": battle.location_id,
                        "uid": dead_baron.uid,
                    })

        rows.sort(key=lambda item: (item.get("layer", ""), item.get("owner", ""), item.get("id", "")))
        return rows

    def _visual_rows_from_bucket(
        self,
        bucket: dict,
        owner: str,
        anchor: tuple[float, float],
        heading: tuple[float, float],
        layer: str,
        node_id: Optional[str] = None,
        group: Optional[MovingGroup] = None,
        hp_override: Optional[float] = None,
        max_hp_override: Optional[float] = None,
    ) -> list[dict]:
        entries: list[tuple[str, str]] = []
        for unit_type, ids in sorted((bucket or {}).items()):
            for visual_id in ids:
                entries.append((unit_type, visual_id))
        if not entries:
            return []

        hx, hy = self._normalize_vector(heading)
        px, py = -hy, hx
        rows: list[dict] = []
        for index, (unit_type, visual_id) in enumerate(entries):
            along, lateral = self._visual_cluster_offset(index, len(entries), layer)
            lateral += self._visual_type_lane(unit_type, layer)
            if layer == "moving":
                # Keep marching offsets in world space. Rotating the whole
                # aggregate formation on every segment turn looks like units
                # orbiting the group anchor rather than walking.
                x = anchor[0] + along
                y = anchor[1] + lateral
            else:
                x = anchor[0] + hx * along + px * lateral
                y = anchor[1] + hy * along + py * lateral
            max_hp = float(max_hp_override if max_hp_override is not None else self._visual_max_hp(unit_type))
            hp = float(hp_override if hp_override is not None else max_hp)
            row = {
                "id": visual_id,
                "owner": owner,
                "team": owner,
                "unit_type": unit_type,
                "type": self._unit_type_code(unit_type),
                "x": round(x, 3),
                "y": round(y, 3),
                "hp": round(max(0.0, hp), 3),
                "max_hp": round(max(0.0, max_hp), 3),
                "hp_ratio": self._visual_hp_ratio(hp, max_hp),
                "state": layer,
                "layer": layer,
            }
            if node_id:
                row["node"] = node_id
            if group is not None:
                row.update({
                    "src": group.src,
                    "dst": group.dst,
                    "final_dst": group.final_dst or group.dst,
                    "path": list(group.path or [group.src, group.dst]),
                    "phase": group.phase,
                    "progress": round(group.progress, 3),
                    "intent": group.intent,
                })
            rows.append(row)
        return rows

    def _visual_heading_for_owner(self, owner: str) -> tuple[float, float]:
        if owner in {"NEUTRAL", "BARON"}:
            return (1.0, 0.0)
        return (1.0, 0.0) if get_team(owner) == "TEAM_A" else (-1.0, 0.0)

    def _normalize_vector(self, vector: tuple[float, float]) -> tuple[float, float]:
        x = float(vector[0] if vector else 1.0)
        y = float(vector[1] if vector else 0.0)
        length = math.hypot(x, y)
        if length <= 0.0:
            return (1.0, 0.0)
        return (x / length, y / length)

    def _visual_cluster_offset(self, index: int, count: int, layer: str) -> tuple[float, float]:
        if count <= 1:
            return (0.0, 0.0)
        spacing = 8.0 if layer in {"stationed", "visiting", "neutral"} else 6.0
        columns = max(1, math.ceil(math.sqrt(count)))
        row = index // columns
        column = index % columns
        rows = math.ceil(count / columns)
        return (
            (column - (columns - 1) / 2.0) * spacing,
            (row - (rows - 1) / 2.0) * spacing,
        )

    def _visual_type_lane(self, unit_type: str, layer: str) -> float:
        scale = 0.7 if layer == "moving" else 1.0
        return {
            UnitType.RANGER.value: -8.0 * scale,
            "ARCHER": -8.0 * scale,
            UnitType.SKIRMISHER.value: 0.0,
            UnitType.VANGUARD.value: 8.0 * scale,
            UnitType.LORD.value: 0.0,
            "NEUTRAL": 0.0,
            "BARON": 0.0,
            "CORE": 0.0,
        }.get(unit_type, 0.0)

    def _visual_max_hp(self, unit_type: str) -> float:
        if unit_type == "NEUTRAL":
            return float(NEUTRAL_GUARD_HP)
        if unit_type == "BARON":
            return float(self.state.baron.max_hp)
        if unit_type == "CORE":
            return float(CORE_HP)
        try:
            enum_value = UnitType.RANGER if unit_type == "ARCHER" else UnitType(unit_type)
            return float(UNIT_STATS[enum_value]["hp"])
        except (KeyError, ValueError):
            return 1.0

    def _visual_hp_ratio(self, hp: float, max_hp: float) -> float:
        max_hp = max(0.001, float(max_hp or 0.0))
        return round(max(0.0, min(1.0, float(hp or 0.0) / max_hp)), 3)

    def _observe(self, player_id: str) -> dict:
        state = self.state
        player = state.players[player_id]

        nodes_obs = {}
        for node_id, node in state.nodes.items():
            nodes_obs[node_id] = {
                "owner": node.owner,
                "units": dict(node.units),
                "core_hp": (
                    round(self.state.players[node_id].core_hp, 3)
                    if node_id in self.state.players
                    else None
                ),
                "core_type": (
                    self.state.players[node_id].core_type
                    if node_id in self.state.players
                    else None
                ),
                "neutral_hp": node.neutral_hp,
                "neutral_income": node.neutral_income,
                "neutral_aggro_teams": sorted(node.neutral_aggro_teams),
                "visiting_units": {
                    pid: dict(units) for pid, units in node.visiting_units.items()
                },
                "unit_ready_tick": dict(node.unit_ready_tick),
                "unit_entry_side": dict(node.unit_entry_side),
            }

        return {
            "player_id": player_id,
            "team": player.team,
            "tick": state.tick,
            "winner_type": state.winner_type,
            "resources": player.resources,
            "alive": player.alive,
            "core_hp": player.core_hp,
            "core_node": player.core_node,
            "nodes": nodes_obs,
            "moving_groups": [
                self._moving_group_snapshot(group, include_visual=False)
                for group in state.moving_groups
            ],
            "active_battles": self._battle_snapshots(),
            "baron": {
                "hp": state.baron.hp,
                "controlled_by": state.baron.controlled_by,
                "allied_lord_owner": state.baron.allied_lord_owner,
                "allied_lord_hp": round(state.baron.allied_lord_hp, 3),
                "allied_lord_target": state.baron.allied_lord_target,
                "allied_lord_path": list(state.baron.allied_lord_path),
                "allied_lord_ready_tick": state.baron.allied_lord_ready_tick,
                "allied_lord_arrival_tick": state.baron.allied_lord_arrival_tick,
                "allied_lord_support_units": dict(state.baron.allied_lord_support_units),
                "allied_lord_ready_in": max(0, int(state.baron.allied_lord_ready_tick or 0) - state.tick),
                "allied_lord_eta": max(0, int(state.baron.allied_lord_arrival_tick or 0) - state.tick),
                "allied_lord_alive": state.baron.allied_lord_alive,
                "allied_lord_expended": state.baron.allied_lord_expended,
            },
            "players": {
                pid: {
                    "alive": p.alive,
                    "resources": p.resources,
                    "core_hp": round(p.core_hp, 3),
                    "core_type": p.core_type,
                }
                for pid, p in state.players.items()
            },
        }

    def _snapshot_tick(self, state: GameState, events: list) -> dict:
        return {
            "tick": state.tick,
            "winner_type": state.winner_type,
            "nodes": {
                nid: {
                    "owner": node.owner,
                    "units": dict(node.units),
                    "core_hp": (
                        round(state.players[nid].core_hp, 3)
                        if nid in state.players
                        else None
                    ),
                    "core_type": (
                        state.players[nid].core_type
                        if nid in state.players
                        else None
                    ),
                    "neutral_hp": node.neutral_hp,
                    "neutral_aggro_teams": sorted(node.neutral_aggro_teams),
                    "visiting_units": {
                        pid: dict(units) for pid, units in node.visiting_units.items()
                    },
                    "unit_ready_tick": dict(node.unit_ready_tick),
                    "unit_entry_side": dict(node.unit_entry_side),
                }
                for nid, node in state.nodes.items()
            },
            "moving_groups": [
                self._moving_group_snapshot(group, include_visual=True)
                for group in state.moving_groups
            ],
            "active_battles": self._battle_snapshots(),
            "visual_units": self._visual_unit_snapshots(state),
            "players": {
                pid: {
                    "resources": round(player.resources, 2),
                    "alive": player.alive,
                    "core_hp": round(player.core_hp, 3),
                    "core_type": player.core_type,
                }
                for pid, player in state.players.items()
            },
            "baron": {
                "hp": state.baron.hp,
                "controlled_by": state.baron.controlled_by,
                "allied_lord_owner": state.baron.allied_lord_owner,
                "allied_lord_hp": round(state.baron.allied_lord_hp, 3),
                "allied_lord_target": state.baron.allied_lord_target,
                "allied_lord_path": list(state.baron.allied_lord_path),
                "allied_lord_ready_tick": state.baron.allied_lord_ready_tick,
                "allied_lord_arrival_tick": state.baron.allied_lord_arrival_tick,
                "allied_lord_support_units": dict(state.baron.allied_lord_support_units),
                "allied_lord_ready_in": max(0, int(state.baron.allied_lord_ready_tick or 0) - state.tick),
                "allied_lord_eta": max(0, int(state.baron.allied_lord_arrival_tick or 0) - state.tick),
                "allied_lord_alive": state.baron.allied_lord_alive,
                "allied_lord_expended": state.baron.allied_lord_expended,
            },
            "core_damage_by_team": {
                team: round(damage, 3)
                for team, damage in sorted(state.core_damage_by_team.items())
            },
            "death_time_final_core_deaths": dict(state.death_time_final_core_deaths),
            "events": events,
        }

    def _moving_group_snapshot(self, group: MovingGroup, include_visual: bool = False) -> dict:
        segment_src, segment_dst = self._moving_group_segment_nodes(group)
        entry_src, entry_dst = self._moving_group_entry_segment_nodes(group)
        lane_id = None
        approach_id = None
        if segment_src and segment_dst:
            try:
                lane_id = canonical_lane_id(segment_src, segment_dst)
                approach_id = directed_approach_id(segment_src, segment_dst)
            except (KeyError, ValueError):
                lane_id = None
                approach_id = None
        threat_zone_ids = threat_zone_ids_for_approach(approach_id) if approach_id else ()
        entry_approach_id = None
        if entry_src and entry_dst:
            try:
                entry_approach_id = directed_approach_id(entry_src, entry_dst)
            except (KeyError, ValueError):
                entry_approach_id = None

        snapshot = {
            "player": group.player_id,
            "team": group.team,
            "units": dict(group.units),
            "src": group.src,
            "dst": group.dst,
            "final_dst": group.final_dst or group.dst,
            "path": list(group.path or [group.src, group.dst]),
            "phase": group.phase,
            "segment_node": group.segment_node,
            "segment_index": group.segment_index,
            "segment_src": segment_src,
            "segment_dst": segment_dst,
            "lane_id": lane_id,
            "approach_id": approach_id,
            "threat_zone_ids": list(threat_zone_ids),
            "entry_approach_id": entry_approach_id,
            "started": group.started_tick,
            "speed": round(group.speed, 3),
            "travel_ticks": round(group.travel_ticks, 3),
            "arrives": group.arrival_tick,
            "progress": round(group.progress, 3),
            "intent": group.intent,
        }
        if include_visual and lane_id and approach_id:
            snapshot["visual"] = moving_group_visual(group)
        if include_visual and getattr(group, "visual_units", None):
            snapshot["visual_units"] = {
                unit_type: list(ids)
                for unit_type, ids in sorted(group.visual_units.items())
            }
        return snapshot

    def _moving_group_segment_nodes(self, group: MovingGroup) -> tuple[Optional[str], Optional[str]]:
        if group.phase == "node_transit" and group.segment_node:
            return group.segment_node, group.dst
        if not group.src or not group.dst:
            return None, None
        return group.src, group.dst

    def _moving_group_entry_segment_nodes(self, group: MovingGroup) -> tuple[Optional[str], Optional[str]]:
        if group.phase == "node_transit" and group.segment_node:
            return group.src, group.segment_node
        if group.phase == "node_enter":
            return group.src, group.dst
        return None, None

    def _battle_snapshots(self) -> list[dict]:
        snapshots = []
        for battle in self.state.active_battles.values():
            snapshots.append({
                "battle_id": battle.battle_id,
                "battle_type": f"{battle.location_kind}_battle",
                "location_kind": battle.location_kind,
                "location_id": battle.location_id,
                "started_at_tick": battle.started_at_tick,
                "last_tick": battle.last_tick,
                "participants": sorted(self._battle_player_ids(battle)),
                "entry_sides": dict(battle.participant_entry_sides),
                "neutral_aggro_teams": sorted(battle.neutral_aggro_teams),
                "edge_position": (
                    battle.edge_position
                    if battle.location_kind in {"edge", "approach"}
                    else None
                ),
                "approach_id": battle.approach_id,
                "threat_zone_ids": list(battle.threat_zone_ids),
                "age_ticks": self.state.tick - battle.started_at_tick,
                "sides": self._battle_side_summary(battle.units),
                "formation": self._battle_formation(battle.units),
            })
        return snapshots

    def _battle_formation(self, units: list) -> list:
        formation = []
        for unit in units:
            if not unit.is_alive():
                continue
            formation.append([
                unit.uid,
                unit.team,
                self._unit_type_code(unit.unit_type),
                round(max(0.0, unit.hp), 1),
                round(max(0.0, unit.max_hp), 1),
                round(unit.pos, 2),
                round(unit.range, 2),
                round(render_position(unit), 2),
                round(unit.x, 2),
                round(unit.y, 2),
                round(unit_range_map(unit), 2),
            ])
        return formation

    def _unit_type_code(self, unit_type: str) -> str:
        return {
            "VANGUARD": "V",
            "RANGER": "R",
            "ARCHER": "R",
            "SKIRMISHER": "S",
            "NEUTRAL": "N",
            "CORE": "C",
            "BARON": "L",
            "LORD": "L",
        }.get(unit_type, unit_type[:1])

    def _alive_unit_index(self, units: list) -> dict[int, tuple[str, str, str]]:
        return {
            unit.uid: (unit.team, unit.unit_type, getattr(unit, "visual_id", ""))
            for unit in units
            if unit.is_alive()
        }

    def _battle_tick_summary(self, battle: Battle, frame: list, before: dict) -> dict:
        after_alive = {unit.uid for unit in battle.units if unit.is_alive()}
        deaths: dict[str, dict[str, int]] = {}
        death_units: list[dict] = []

        for uid, (team, unit_type, visual_id) in before.items():
            if uid in after_alive:
                continue
            deaths.setdefault(team, {})
            deaths[team][unit_type] = deaths[team].get(unit_type, 0) + 1
            if visual_id:
                death_units.append({
                    "id": visual_id,
                    "uid": uid,
                    "owner": team,
                    "type": unit_type,
                })

        damage_by_team: dict[str, float] = {}
        attacks = []
        frame_by_uid = {item.get("uid"): item for item in frame}
        for item in frame:
            damage = float(item.get("dmg", 0.0))
            if damage <= 0:
                continue
            team = item["team"]
            damage_by_team[team] = damage_by_team.get(team, 0.0) + damage
            shot_log = item.get("attacks") or []
            if not shot_log and item.get("target") is not None:
                shot_log = [{"target": item.get("target"), "damage": damage}]
            for shot in shot_log:
                target_uid = shot.get("target")
                target = frame_by_uid.get(target_uid)
                if target:
                    attacks.append([
                        item["uid"],
                        target_uid,
                        round(float(shot.get("damage", 0.0)), 2),
                        item["team"],
                        self._unit_type_code(item["type"]),
                        target["team"],
                        self._unit_type_code(target["type"]),
                    ])

        return {
            "alive": self._battle_side_summary(battle.units),
            "damage_by_team": {
                team: round(damage, 2)
                for team, damage in sorted(damage_by_team.items())
            },
            "attacks": attacks,
            "deaths": deaths,
            "death_units": death_units,
        }

    def _battle_side_summary(self, units: list) -> dict:
        sides: dict[str, dict] = {}
        for unit in units:
            if not unit.is_alive():
                continue
            side = sides.setdefault(unit.team, {"units": {}, "hp": 0.0})
            side["units"][unit.unit_type] = side["units"].get(unit.unit_type, 0) + 1
            side["hp"] += max(0.0, unit.hp)

        for side in sides.values():
            side["hp"] = round(side["hp"], 2)
            side["units"] = dict(sorted(side["units"].items()))
        return dict(sorted(sides.items()))
