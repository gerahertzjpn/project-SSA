from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .constants import (
    UnitType, CORE_HP, BARON_HP, RESOURCE_PER_TICK,
    TEAM_A_PLAYERS, TEAM_B_PLAYERS, PLAYER_IDS,
)
from .map import CP_NODES as MAP_CP_NODES
from .units import UnitInstance


@dataclass
class NodeState:
    node_id: str
    owner: str              # 'NEUTRAL' | player_id
    units: dict             # {UnitType.value: count}
    neutral_hp: int         # remaining neutral guards (0 = cleared)
    neutral_income: int     # income per tick when captured (scaled from original neutral strength)
    neutral_aggro_teams: set[str] = field(default_factory=set)
    visiting_units: dict = field(default_factory=dict)  # player_id -> {UnitType.value: count}
    unit_ready_tick: dict = field(default_factory=dict)  # player_id -> first tick units may move again
    unit_entry_side: dict = field(default_factory=dict)  # player_id -> attacker/defender side last used in node combat

    def total_units(self) -> int:
        visitors = sum(sum(units.values()) for units in self.visiting_units.values())
        return sum(self.units.values()) + visitors

    def is_neutral(self) -> bool:
        return self.owner == "NEUTRAL"


@dataclass
class MovingGroup:
    player_id: str
    team: str
    units: dict             # {UnitType.value: count}
    src: str
    dst: str
    arrival_tick: int
    started_tick: int = 0
    travel_ticks: float = 1.0
    progress: float = 0.0
    intent: str = "CAPTURE"  # CAPTURE/ATTACK engages neutrals; TRANSIT passes passive neutrals
    path: list[str] = field(default_factory=list)
    segment_index: int = 0
    phase: str = "edge"  # node_exit | edge | node_transit | node_enter
    segment_node: Optional[str] = None
    speed: float = 1.0
    final_dst: Optional[str] = None

    def total_units(self) -> int:
        return sum(self.units.values())


@dataclass
class PlayerState:
    player_id: str
    team: str
    core_node: str
    resources: float
    alive: bool
    core_hp: float


@dataclass
class BaronState:
    hp: int
    max_hp: int
    controlled_by: Optional[str]   # None or player_id of killer
    ticks_remaining: int            # ticks until baron is expended/respawns
    allied_lord_owner: Optional[str] = None
    allied_lord_hp: float = 0.0
    allied_lord_target: Optional[str] = None
    allied_lord_alive: bool = False
    allied_lord_expended: bool = False


@dataclass
class Battle:
    battle_id: str
    location_kind: str              # 'node' | 'edge'
    location_id: str                # node id or canonical edge id
    units: list[UnitInstance]
    started_at_tick: int
    last_tick: int
    participant_routes: dict = field(default_factory=dict)  # player_id -> (src, dst)
    participant_intents: dict = field(default_factory=dict)  # player_id -> movement intent
    participant_entry_sides: dict = field(default_factory=dict)  # player_id -> attacker/defender
    participant_paths: dict = field(default_factory=dict)  # player_id -> full movement path
    participant_segment_indices: dict = field(default_factory=dict)  # player_id -> segment index at battle join
    participant_speeds: dict = field(default_factory=dict)  # player_id -> movement speed
    participant_final_dsts: dict = field(default_factory=dict)  # player_id -> requested destination
    edge_endpoints: Optional[tuple[str, str]] = None
    edge_position: float = 0.5
    neutral_aggro_teams: set[str] = field(default_factory=set)
    objective: Optional[str] = None  # neutral | baron | core | pvp
    core_owner: Optional[str] = None


@dataclass
class GameState:
    tick: int
    nodes: dict                         # node_id -> NodeState
    moving_groups: list                 # list[MovingGroup]
    players: dict                       # player_id -> PlayerState
    baron: BaronState
    winner: Optional[str]               # None | 'TEAM_A' | 'TEAM_B' | 'DRAW'
    winner_type: Optional[str]          # normal | death_time | draw_no_core_damage | truncated
    core_combat_damage: float           # non-death-time core damage dealt by combat/lord
    core_damage_by_team: dict           # team -> non-death-time core damage dealt
    death_time_final_core_deaths: dict  # team -> cores destroyed by final death-time hit
    rng: object                         # numpy rng for reproducibility

    # ongoing persistent battles: "node:<id>" or "edge:<a>|<b>" -> Battle
    active_battles: dict = field(default_factory=dict)


def make_initial_state(seed: int = 42) -> GameState:
    rng = np.random.default_rng(seed)

    neutral_strengths = _roll_neutrals(rng)

    nodes = {}

    for pid in TEAM_A_PLAYERS:
        nodes[pid] = NodeState(
            node_id=pid,
            owner=pid,
            units={UnitType.VANGUARD.value: 2},
            neutral_hp=0,
            neutral_income=0,
            unit_entry_side={pid: "attacker"},
        )
    for pid in TEAM_B_PLAYERS:
        nodes[pid] = NodeState(
            node_id=pid,
            owner=pid,
            units={UnitType.VANGUARD.value: 2},
            neutral_hp=0,
            neutral_income=0,
            unit_entry_side={pid: "defender"},
        )

    for cp in MAP_CP_NODES:
        strength = neutral_strengths[cp]
        nodes[cp] = NodeState(
            node_id=cp,
            owner="NEUTRAL",
            units={},
            neutral_hp=strength,
            neutral_income=_neutral_income(strength),
        )

    nodes["Baron"] = NodeState(
        node_id="Baron",
        owner="NEUTRAL",
        units={},
        neutral_hp=0,
        neutral_income=0,
    )

    start_resources = float(10 * rng.uniform(0.8, 1.2))

    players = {}
    for pid in PLAYER_IDS:
        team = "TEAM_A" if pid in TEAM_A_PLAYERS else "TEAM_B"
        players[pid] = PlayerState(
            player_id=pid,
            team=team,
            core_node=pid,
            resources=start_resources,
            alive=True,
            core_hp=float(CORE_HP),
        )

    baron = BaronState(
        hp=BARON_HP,
        max_hp=BARON_HP,
        controlled_by=None,
        ticks_remaining=0,
    )

    return GameState(
        tick=0,
        nodes=nodes,
        moving_groups=[],
        players=players,
        baron=baron,
        winner=None,
        winner_type=None,
        core_combat_damage=0.0,
        core_damage_by_team={"TEAM_A": 0.0, "TEAM_B": 0.0},
        death_time_final_core_deaths={"TEAM_A": 0, "TEAM_B": 0},
        rng=rng,
    )


def _roll_neutrals(rng) -> dict:
    cp1_cp5 = int(rng.integers(1, 8))  # [1, 7]
    cp2_cp6 = int(rng.integers(1, 8))
    cp3 = int(rng.integers(1, 8))
    cp4 = int(rng.integers(1, 8))
    return {
        "CP1": cp1_cp5, "CP5": cp1_cp5,
        "CP2": cp2_cp6, "CP6": cp2_cp6,
        "CP3": cp3,
        "CP4": cp4,
    }


def _neutral_income(strength: int) -> int:
    return max(1, (int(strength) + 1) // 2)


def get_team(player_id: str) -> str:
    return "TEAM_A" if player_id in TEAM_A_PLAYERS else "TEAM_B"


def get_enemy_team(team: str) -> str:
    return "TEAM_B" if team == "TEAM_A" else "TEAM_A"


def alive_players(state: GameState, team: str) -> list:
    return [p for p in state.players.values() if p.team == team and p.alive]
