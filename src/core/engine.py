"""
Main game engine. Runs one world tick at a time.

Tick order:
1. Existing persistent battles tick once.
2. Moving groups advance, collide on edges, join battles, or arrive at nodes.
3. Income is collected.
4. Bots choose actions.
5. Production is applied.
6. Movement actions create MovingGroups.
7. Victory is checked.
8. Tick snapshot is written to the replay buffer.
"""

from dataclasses import dataclass
from typing import Optional

from .constants import (
    ActionType, UnitType, TICK_LIMIT, TRAVEL_TIME, EDGE_COLLISION_THRESHOLD,
    MAX_COMBAT_TICKS, TEAM_A_PLAYERS, TEAM_B_PLAYERS,
)
from .state import GameState, MovingGroup, NodeState, Battle, get_team
from .map import ADJACENCY, NODE_TYPES, NodeType
from .combat import (
    build_player_units, build_neutral_units, build_core_unit, build_baron_unit,
    tick_battle_units, has_hostile_units, count_survivors_by_player,
)
from .production import apply_production, collect_income


def unit_total(units: dict) -> int:
    return sum(int(count or 0) for count in (units or {}).values())


@dataclass
class Action:
    action_type: ActionType
    unit_type: Optional[UnitType] = None   # for PRODUCE
    destination: Optional[str] = None      # for MOVE
    source: Optional[str] = None           # for MOVE (which node to send from)
    intent: str = "CAPTURE"               # CAPTURE/ATTACK engages neutrals, TRANSIT does not


class GameEngine:
    def __init__(self, state: GameState, bots: dict, record_trace: bool = True):
        """
        bots: dict mapping player_id -> Bot instance
        record_trace: pass False during RL training to skip per-tick combat snapshots
        """
        self.state = state
        self.bots = bots
        self.record_trace = record_trace
        self.replay_ticks = []

    def run(self) -> str:
        """Run match to completion. Returns winner string."""
        while self.state.winner is None and self.state.tick < TICK_LIMIT:
            self.tick()

        if self.state.winner is None:
            self.state.winner = "DRAW"
        return self.state.winner

    def tick(self) -> None:
        state = self.state
        tick_events = []

        self._tick_active_battles(tick_events)
        self._advance_moving_groups(tick_events)

        collect_income(state)

        actions: dict[str, Action] = {}
        for player_id, bot in self.bots.items():
            player = state.players[player_id]
            if not player.alive:
                continue
            actions[player_id] = bot.act(self._observe(player_id))

        for player_id, action in actions.items():
            if action.action_type == ActionType.PRODUCE and action.unit_type:
                if apply_production(state, player_id, action.unit_type):
                    core_node = state.nodes[state.players[player_id].core_node]
                    self._remember_node_entry_side(core_node, player_id, self._default_player_side(player_id))

        for player_id, action in actions.items():
            if action.action_type == ActionType.MOVE:
                event = self._apply_move(player_id, action)
                if event:
                    tick_events.append(event)

        self._check_victory()
        self.replay_ticks.append(self._snapshot_tick(state, tick_events))

        state.tick += 1

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def _advance_moving_groups(self, tick_events: list) -> None:
        state = self.state
        if not state.moving_groups:
            return

        for group in state.moving_groups:
            group.progress = min(1.0, group.progress + 1.0 / max(1, group.travel_ticks))

        remaining: list[MovingGroup] = []
        for group in state.moving_groups:
            battle = self._edge_battle_for_group(group)
            if battle and self._group_reached_edge_battle(group, battle):
                tick_events.append(self._join_battle(battle, group, reinforced=True))
            else:
                remaining.append(group)

        remaining = self._create_edge_collisions(remaining, tick_events)

        still_moving: list[MovingGroup] = []
        for group in remaining:
            if group.progress >= 1.0:
                event = self._resolve_arrival(group)
                if event:
                    tick_events.append(event)
            else:
                still_moving.append(group)

        state.moving_groups = still_moving

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
        if first.src != second.dst or first.dst != second.src:
            return False
        if first.team == second.team:
            return False
        if first.total_units() <= 0 or second.total_units() <= 0:
            return False
        return first.progress + second.progress >= 1.0 - EDGE_COLLISION_THRESHOLD

    def _create_edge_battle(self, first: MovingGroup, second: MovingGroup) -> Battle:
        key = self._edge_key(first.src, first.dst)
        battle = Battle(
            battle_id=key,
            location_kind="edge",
            location_id=self._edge_id(first.src, first.dst),
            units=[],
            started_at_tick=self.state.tick,
            last_tick=self.state.tick,
            edge_endpoints=self._canonical_edge(first.src, first.dst),
            edge_position=0.5,
            objective="pvp",
        )
        self._join_battle(battle, first, reinforced=False)
        self._join_battle(battle, second, reinforced=False)
        self.state.active_battles[key] = battle
        return battle

    def _edge_battle_for_group(self, group: MovingGroup) -> Optional[Battle]:
        return self.state.active_battles.get(self._edge_key(group.src, group.dst))

    def _group_reached_edge_battle(self, group: MovingGroup, battle: Battle) -> bool:
        return group.progress >= battle.edge_position - EDGE_COLLISION_THRESHOLD

    def _apply_move(self, player_id: str, action: Action) -> Optional[dict]:
        state = self.state
        src = action.source or state.players[player_id].core_node
        dst = action.destination

        if dst is None or src not in state.nodes:
            return None
        if dst not in ADJACENCY.get(src, []):
            return None

        if (
            dst == "Baron"
            and action.intent.upper() == "TRANSIT"
            and state.nodes["Baron"].owner == "NEUTRAL"
            and state.baron.hp > 0
        ):
            return {
                "type": "BARON_BLOCKED",
                "player": player_id,
                "src": src,
                "dst": dst,
                "reason": "living_baron_blocks_transit",
            }

        src_node = state.nodes[src]
        units_to_send = self._take_units_for_move(player_id, src_node)
        if not units_to_send:
            return None

        group = MovingGroup(
            player_id=player_id,
            team=get_team(player_id),
            units=units_to_send,
            src=src,
            dst=dst,
            arrival_tick=state.tick + TRAVEL_TIME,
            started_tick=state.tick,
            travel_ticks=TRAVEL_TIME,
            progress=0.0,
            intent=action.intent.upper(),
        )
        state.moving_groups.append(group)

        return {
            "type": "MOVE",
            "player": player_id,
            "src": src,
            "dst": dst,
            "units": units_to_send,
            "intent": group.intent,
        }

    def _take_units_for_move(self, player_id: str, node: NodeState) -> dict:
        if node.unit_ready_tick.get(player_id, 0) > self.state.tick:
            return {}
        if node.owner == player_id and node.units:
            units = dict(node.units)
            node.units = {}
            node.unit_ready_tick.pop(player_id, None)
            node.unit_entry_side.pop(player_id, None)
            return units
        if player_id in node.visiting_units and node.visiting_units[player_id]:
            node.unit_ready_tick.pop(player_id, None)
            node.unit_entry_side.pop(player_id, None)
            return node.visiting_units.pop(player_id)
        return {}

    # ------------------------------------------------------------------
    # Arrivals and node battles
    # ------------------------------------------------------------------

    def _resolve_arrival(self, group: MovingGroup) -> Optional[dict]:
        dst = group.dst
        if self._node_key(dst) in self.state.active_battles:
            battle = self.state.active_battles[self._node_key(dst)]
            return self._join_battle(battle, group, reinforced=True)

        node_type = NODE_TYPES[dst]
        if node_type == NodeType.BARON:
            return self._handle_baron_arrival(group)
        if node_type == NodeType.CORE:
            return self._handle_core_arrival(group)
        return self._handle_control_point_arrival(group)

    def _handle_core_arrival(self, group: MovingGroup) -> Optional[dict]:
        node = self.state.nodes[group.dst]
        core_owner = group.dst

        if get_team(group.player_id) == get_team(core_owner):
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
        include_visitors: bool = False,
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
            location_kind="node",
            location_id=node_id,
            units=[],
            started_at_tick=self.state.tick,
            last_tick=self.state.tick,
            objective=objective,
            core_owner=core_owner,
        )

        if include_owner_units and node.owner in self.state.players and node.units:
            owner_entry_side = node.unit_entry_side.get(node.owner)
            self._add_player_units_to_battle(
                battle,
                player_id=node.owner,
                units=node.units,
                src=node_id,
                dst=node_id,
                intent="DEFEND",
                entry_side=owner_entry_side,
            )
            node.units = {}
            node.unit_entry_side.pop(node.owner, None)

        if include_visitors and node.visiting_units:
            for visitor_id, units in list(node.visiting_units.items()):
                visitor_entry_side = node.unit_entry_side.get(visitor_id)
                self._add_player_units_to_battle(
                    battle,
                    player_id=visitor_id,
                    units=units,
                    src=node_id,
                    dst=node_id,
                    intent="VISIT",
                    entry_side=visitor_entry_side,
                )
                node.unit_entry_side.pop(visitor_id, None)
            node.visiting_units.clear()

        if include_core and core_owner:
            battle.units.append(build_core_unit(core_owner))

        if include_neutral and node.neutral_hp > 0:
            battle.units.extend(build_neutral_units(node.neutral_hp))
            self._aggro_neutrals(battle, group.player_id, node_id)

        if include_baron and self.state.baron.hp > 0:
            battle.units.append(build_baron_unit(self.state.baron.hp))

        self._join_battle(battle, group, reinforced=False)
        self.state.active_battles[key] = battle

        return {
            "type": "BATTLE_STARTED",
            "battle_type": "node_battle",
            "location_kind": "node",
            "location_id": node_id,
            "battle_id": battle.battle_id,
            "objective": objective,
            "participants": sorted(self._battle_player_ids(battle)),
            "neutral_aggro_teams": sorted(battle.neutral_aggro_teams),
        }

    def _join_battle(self, battle: Battle, group: MovingGroup, reinforced: bool) -> dict:
        self._add_player_units_to_battle(
            battle,
            player_id=group.player_id,
            units=group.units,
            src=group.src,
            dst=group.dst,
            intent=group.intent,
        )
        if battle.objective == "neutral" and group.intent.upper() in {"CAPTURE", "ATTACK"}:
            self._aggro_neutrals(battle, group.player_id, battle.location_id)

        return {
            "type": "BATTLE_REINFORCEMENT" if reinforced else "BATTLE_JOIN",
            "battle_type": f"{battle.location_kind}_battle",
            "location_kind": battle.location_kind,
            "location_id": battle.location_id,
            "battle_id": battle.battle_id,
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
    ) -> None:
        if not units:
            return
        entry_side = entry_side or self._entry_side(player_id, src, dst)
        battle.units.extend(build_player_units(units, player_id, entry_side))
        battle.participant_routes[player_id] = (src, dst)
        battle.participant_intents[player_id] = intent
        battle.participant_entry_sides[player_id] = entry_side

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
            before = self._alive_unit_index(battle.units) if self.record_trace else {}
            frame = tick_battle_units(
                battle.units,
                record_trace=self.record_trace,
                neutral_aggro_teams=battle.neutral_aggro_teams,
            )
            battle.last_tick = self.state.tick

            if self.record_trace:
                tick_events.append({
                    "type": "BATTLE_TICK",
                    "battle_type": f"{battle.location_kind}_battle",
                    "location_kind": battle.location_kind,
                    "location_id": battle.location_id,
                    "battle_id": battle.battle_id,
                    "summary": self._battle_tick_summary(battle, frame, before),
                })

            timed_out = self.state.tick - battle.started_at_tick >= MAX_COMBAT_TICKS
            if timed_out:
                self._force_timeout_resolution(battle)

            if timed_out or not has_hostile_units(battle.units, battle.neutral_aggro_teams):
                resolution = (
                    self._finish_edge_battle(battle)
                    if battle.location_kind == "edge"
                    else self._finish_node_battle(battle)
                )
                tick_events.append(resolution)
                del self.state.active_battles[key]

    def _finish_node_battle(self, battle: Battle) -> dict:
        node = self.state.nodes[battle.location_id]
        survivors_by_player = count_survivors_by_player(battle.units)
        winner_player = self._strongest_survivor(survivors_by_player)

        if battle.objective == "core":
            core_alive = any(u.is_alive() and u.unit_type == "CORE" for u in battle.units)
            core_owner = battle.core_owner or battle.location_id
            if not core_alive and winner_player:
                self.state.players[core_owner].alive = False
                node.owner = winner_player
                self._set_node_units(
                    node,
                    winner_player,
                    survivors_by_player.get(winner_player, {}),
                    self._battle_entry_side(battle, winner_player),
                )
                self._lock_node_player(node, winner_player)
                self._remove_player_units(core_owner)
            else:
                node.owner = core_owner
                self._set_node_units(
                    node,
                    core_owner,
                    survivors_by_player.get(core_owner, {}),
                    self._battle_entry_side(battle, core_owner),
                )
                self._lock_node_player(node, core_owner)
            return self._resolution_event(battle, winner_player, {"core_destroyed": not core_alive})

        if battle.objective == "neutral":
            remaining_neutral_hp = int(sum(
                max(0.0, unit.hp)
                for unit in battle.units
                if unit.unit_type == "NEUTRAL" and unit.is_alive()
            ))
            node.neutral_aggro_teams = set(battle.neutral_aggro_teams)
            if remaining_neutral_hp > 0 or not winner_player:
                node.owner = "NEUTRAL"
                node.units = {}
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
                )
                self._lock_node_player(node, winner_player)
            else:
                self.state.baron.hp = max(0, remaining_baron_hp)
                node.owner = "NEUTRAL"
                node.units = {}
                self._forget_battle_entry_sides(node, battle)
            return self._resolution_event(battle, winner_player or "BARON", {"baron_killed": not baron_alive})

        if winner_player:
            if node.owner == "NEUTRAL" and node.neutral_hp > 0:
                self._merge_visiting_units(
                    node,
                    winner_player,
                    survivors_by_player.get(winner_player, {}),
                    entry_side=self._battle_entry_side(battle, winner_player),
                )
                self._lock_node_player(node, winner_player)
            else:
                node.owner = winner_player
                self._set_node_units(
                    node,
                    winner_player,
                    survivors_by_player.get(winner_player, {}),
                    self._battle_entry_side(battle, winner_player),
                )
                self._lock_node_player(node, winner_player)
        else:
            node.units = {}
            self._forget_battle_entry_sides(node, battle)

        return self._resolution_event(battle, winner_player, {"result": "PVP_RESOLVED"})

    def _finish_edge_battle(self, battle: Battle) -> dict:
        survivors_by_player = count_survivors_by_player(battle.units)
        continuing = []

        for player_id, units in survivors_by_player.items():
            src, dst = battle.participant_routes.get(player_id, (None, None))
            if src is None or dst is None:
                continue
            group = MovingGroup(
                player_id=player_id,
                team=get_team(player_id),
                units=units,
                src=src,
                dst=dst,
                arrival_tick=self.state.tick + 2,
                started_tick=self.state.tick,
                travel_ticks=TRAVEL_TIME,
                progress=battle.edge_position,
                intent=battle.participant_intents.get(player_id, "CAPTURE"),
            )
            self.state.moving_groups.append(group)
            continuing.append({"player": player_id, "src": src, "dst": dst, "units": units})

        return self._resolution_event(battle, self._strongest_survivor(survivors_by_player), {
            "result": "EDGE_RESOLVED",
            "continuing_groups": continuing,
        })

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
                node.unit_ready_tick.pop(player_id, None)
                node.unit_entry_side.pop(player_id, None)
            node.visiting_units.pop(player_id, None)
            node.unit_ready_tick.pop(player_id, None)
            node.unit_entry_side.pop(player_id, None)

        state.moving_groups = [
            group for group in state.moving_groups if group.player_id != player_id
        ]

        for battle in state.active_battles.values():
            for unit in battle.units:
                if unit.team == player_id:
                    unit.hp = 0

    def _return_group_to_source(self, group: MovingGroup) -> None:
        src_node = self.state.nodes[group.src]
        entry_side = self._entry_side(group.player_id, group.dst, group.src)
        if src_node.owner == group.player_id:
            self._merge_units(src_node, group.units, player_id=group.player_id, entry_side=entry_side)
        else:
            self._merge_visiting_units(src_node, group.player_id, group.units, entry_side=entry_side)
        self._lock_node_player(src_node, group.player_id)

    def _set_node_units(
        self,
        node: NodeState,
        player_id: Optional[str],
        units: dict,
        entry_side: Optional[str] = None,
    ) -> None:
        node.units = dict(units or {})
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
    ) -> None:
        for unit_type, count in units.items():
            node.units[unit_type] = node.units.get(unit_type, 0) + count
        if unit_total(units) > 0:
            self._remember_node_entry_side(node, player_id or node.owner, entry_side)

    def _merge_visiting_units(
        self,
        node: NodeState,
        player_id: str,
        units: dict,
        entry_side: Optional[str] = None,
    ) -> None:
        bucket = node.visiting_units.setdefault(player_id, {})
        for unit_type, count in units.items():
            bucket[unit_type] = bucket.get(unit_type, 0) + count
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

    def _check_victory(self) -> None:
        state = self.state
        a_alive = any(p.alive for p in state.players.values() if p.team == "TEAM_A")
        b_alive = any(p.alive for p in state.players.values() if p.team == "TEAM_B")

        if not a_alive and not b_alive:
            state.winner = "DRAW"
        elif not a_alive:
            state.winner = "TEAM_B"
        elif not b_alive:
            state.winner = "TEAM_A"

    # ------------------------------------------------------------------
    # Observation and replay
    # ------------------------------------------------------------------

    def _observe(self, player_id: str) -> dict:
        state = self.state
        player = state.players[player_id]

        nodes_obs = {}
        for node_id, node in state.nodes.items():
            nodes_obs[node_id] = {
                "owner": node.owner,
                "units": dict(node.units),
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
            "resources": player.resources,
            "alive": player.alive,
            "core_node": player.core_node,
            "nodes": nodes_obs,
            "moving_groups": [
                {
                    "player": group.player_id,
                    "team": group.team,
                    "units": dict(group.units),
                    "src": group.src,
                    "dst": group.dst,
                    "arrives": group.arrival_tick,
                    "progress": round(group.progress, 3),
                    "intent": group.intent,
                }
                for group in state.moving_groups
            ],
            "active_battles": self._battle_snapshots(),
            "baron": {
                "hp": state.baron.hp,
                "controlled_by": state.baron.controlled_by,
            },
            "players": {
                pid: {"alive": p.alive, "resources": p.resources}
                for pid, p in state.players.items()
            },
        }

    def _snapshot_tick(self, state: GameState, events: list) -> dict:
        return {
            "tick": state.tick,
            "nodes": {
                nid: {
                    "owner": node.owner,
                    "units": dict(node.units),
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
                {
                    "player": group.player_id,
                    "units": dict(group.units),
                    "src": group.src,
                    "dst": group.dst,
                    "arrives": group.arrival_tick,
                    "progress": round(group.progress, 3),
                    "intent": group.intent,
                }
                for group in state.moving_groups
            ],
            "active_battles": self._battle_snapshots(),
            "players": {
                pid: {"resources": round(player.resources, 2), "alive": player.alive}
                for pid, player in state.players.items()
            },
            "baron": {
                "hp": state.baron.hp,
                "controlled_by": state.baron.controlled_by,
            },
            "events": events,
        }

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
                "edge_position": battle.edge_position if battle.location_kind == "edge" else None,
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

    def _alive_unit_index(self, units: list) -> dict[int, tuple[str, str]]:
        return {
            unit.uid: (unit.team, unit.unit_type)
            for unit in units
            if unit.is_alive()
        }

    def _battle_tick_summary(self, battle: Battle, frame: list, before: dict) -> dict:
        after_alive = {unit.uid for unit in battle.units if unit.is_alive()}
        deaths: dict[str, dict[str, int]] = {}

        for uid, (team, unit_type) in before.items():
            if uid in after_alive:
                continue
            deaths.setdefault(team, {})
            deaths[team][unit_type] = deaths[team].get(unit_type, 0) + 1

        damage_by_team: dict[str, float] = {}
        attacks = []
        frame_by_uid = {item.get("uid"): item for item in frame}
        for item in frame:
            damage = float(item.get("dmg", 0.0))
            if damage <= 0:
                continue
            team = item["team"]
            damage_by_team[team] = damage_by_team.get(team, 0.0) + damage
            target_uid = item.get("target")
            target = frame_by_uid.get(target_uid)
            if target:
                attacks.append([
                    item["uid"],
                    target_uid,
                    round(damage, 2),
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
