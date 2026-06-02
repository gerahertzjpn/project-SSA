from enum import Enum, auto


class NodeType(Enum):
    CORE = auto()
    CONTROL_POINT = auto()
    BARON = auto()


class Team(Enum):
    TEAM_A = "TEAM_A"
    TEAM_B = "TEAM_B"
    NEUTRAL = "NEUTRAL"


class UnitType(Enum):
    VANGUARD = "VANGUARD"
    RANGER = "RANGER"
    ARCHER = "RANGER"  # legacy alias; display/docs use Ranger/R
    SKIRMISHER = "SKIRMISHER"
    LORD = "LORD"


class ActionType(Enum):
    HOLD = "HOLD"
    PRODUCE = "PRODUCE"
    MOVE = "MOVE"


TICK_LIMIT = 300
MAX_COMBAT_TICKS = 50

RESOURCE_PER_TICK = 2.0

UNIT_COST = {
    UnitType.VANGUARD: 3,
    UnitType.ARCHER: 4,
    UnitType.SKIRMISHER: 3,
}

CORE_HP = 50
CORE_ATK = 2
DEATH_TIME_START_TICK = 270
DEATH_TIME_CORE_DAMAGE_PER_TICK = (CORE_HP - 1) / (TICK_LIMIT - DEATH_TIME_START_TICK)

TRAVEL_TIME = 3  # world ticks per edge; enables mid-edge collision checks
EDGE_COLLISION_THRESHOLD = 0.05

BARON_HP = 120
BARON_ATK = 16
BARON_RESPAWN_TICKS = 50

PLAYER_IDS = ["A1", "A2", "A3", "B1", "B2", "B3"]
TEAM_A_PLAYERS = ["A1", "A2", "A3"]
TEAM_B_PLAYERS = ["B1", "B2", "B3"]

NODE_POSITIONS = {
    "A1": (70, 90),
    "A2": (70, 210),
    "A3": (70, 330),
    "CP1": (210, 150),
    "CP2": (210, 270),
    "CP3": (340, 80),
    "CP4": (340, 340),
    "CP5": (470, 150),
    "CP6": (470, 270),
    "Baron": (340, 210),
    "B1": (610, 90),
    "B2": (610, 210),
    "B3": (610, 330),
}
