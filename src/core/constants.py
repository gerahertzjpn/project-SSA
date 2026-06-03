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

CORE_HP = 80
CORE_ATK = 6
DEATH_TIME_START_TICK = 270
DEATH_TIME_DAMAGE_TICKS = TICK_LIMIT - DEATH_TIME_START_TICK
DEATH_TIME_CORE_DAMAGE_TOTAL = max(0, CORE_HP - 1)


def death_time_core_damage_for_tick(tick: int) -> int:
    if tick < DEATH_TIME_START_TICK or tick >= TICK_LIMIT:
        return 0
    elapsed = int(tick) - DEATH_TIME_START_TICK + 1
    previous_elapsed = elapsed - 1
    current_total = DEATH_TIME_CORE_DAMAGE_TOTAL * elapsed // DEATH_TIME_DAMAGE_TICKS
    previous_total = DEATH_TIME_CORE_DAMAGE_TOTAL * previous_elapsed // DEATH_TIME_DAMAGE_TICKS
    return max(0, current_total - previous_total)

TRAVEL_TIME = 3  # world ticks per edge; enables mid-edge collision checks
EDGE_COLLISION_THRESHOLD = 0.05

BARON_HP = 120
BARON_ATK = 16
BARON_ATTACK_RATE = 3
BARON_RESPAWN_TICKS = 50
# Combat maps are a 1D [0..20] line, so map radius is 10.
LORD_ATTACK_RANGE = 11.0
CORE_ATTACK_RANGE = 12.0

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
