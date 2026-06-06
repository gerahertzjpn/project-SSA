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
    RAIDER = "RAIDER"
    SPECIALIST_1 = "RAIDER"  # legacy alias for Tech 1 human unit
    BMP = "BMP"
    SPECIALIST_2 = "BMP"  # legacy alias for Tech 2 human unit
    SWARMLING = "SWARMLING"
    SPITTER = "SPITTER"
    BRUTE = "BRUTE"
    RAVAGER = "RAVAGER"
    BEHEMOTH = "BEHEMOTH"
    WALKER = "WALKER"
    LASER = "LASER"
    DRONE = "DRONE"
    GUARDIAN_MECH = "GUARDIAN_MECH"
    SIEGE_PLATFORM = "SIEGE_PLATFORM"
    LORD = "LORD"

    @classmethod
    def _missing_(cls, value):
        legacy = {
            "SPECIALIST_1": cls.RAIDER,
            "SPECIALIST_2": cls.BMP,
            "ARCHER": cls.RANGER,
            "GUARDIAN": cls.GUARDIAN_MECH,
            "SIEGE": cls.SIEGE_PLATFORM,
        }
        return legacy.get(str(value).upper())


class BuildingType(Enum):
    TURRET = "TURRET"
    BOOSTER = "BOOSTER"
    POWER = "POWER"
    LIFE = "LIFE"
    ECONOMY_COLLECTOR = "ECONOMY_COLLECTOR"
    TECH_1 = "TECH_1"
    TECH_2 = "TECH_2"


class ActionType(Enum):
    HOLD = "HOLD"
    PRODUCE = "PRODUCE"
    MOVE = "MOVE"
    BUILD = "BUILD"
    MULTI = "MULTI"


SPECIAL_DEST_ALLY_LORD = "ALLY_LORD"
SPECIAL_DEST_ENEMY_LORD = "ENEMY_LORD"
SPECIAL_LORD_DESTINATIONS = (SPECIAL_DEST_ALLY_LORD, SPECIAL_DEST_ENEMY_LORD)

TICK_LIMIT = 300
MAX_COMBAT_TICKS = 50

RESOURCE_PER_TICK = 2.0

UNIT_COST = {
    UnitType.VANGUARD: 3,
    UnitType.RANGER: 5,
    UnitType.SKIRMISHER: 2,
    UnitType.RAIDER: 8,
    UnitType.BMP: 20,
    UnitType.SWARMLING: 1,
    UnitType.SPITTER: 3,
    UnitType.BRUTE: 4,
    UnitType.RAVAGER: 8,
    UnitType.BEHEMOTH: 20,
    UnitType.WALKER: 5,
    UnitType.LASER: 7,
    UnitType.DRONE: 3,
    UnitType.GUARDIAN_MECH: 10,
    UnitType.SIEGE_PLATFORM: 24,
}

UNIT_TECH_REQUIREMENTS = {
    UnitType.RAIDER: BuildingType.TECH_1,
    UnitType.BMP: BuildingType.TECH_2,
    UnitType.RAVAGER: BuildingType.TECH_1,
    UnitType.BEHEMOTH: BuildingType.TECH_2,
    UnitType.GUARDIAN_MECH: BuildingType.TECH_1,
    UnitType.SIEGE_PLATFORM: BuildingType.TECH_2,
}

UNIT_TECH_REQUIREMENT_CHAINS = {
    UnitType.RAIDER: (BuildingType.TECH_1,),
    UnitType.BMP: (BuildingType.TECH_1, BuildingType.TECH_2),
    UnitType.RAVAGER: (BuildingType.TECH_1,),
    UnitType.BEHEMOTH: (BuildingType.TECH_1, BuildingType.TECH_2),
    UnitType.GUARDIAN_MECH: (BuildingType.TECH_1,),
    UnitType.SIEGE_PLATFORM: (BuildingType.TECH_1, BuildingType.TECH_2),
}


def unit_tech_requirements(unit_type: UnitType) -> tuple[BuildingType, ...]:
    return UNIT_TECH_REQUIREMENT_CHAINS.get(unit_type, ())

FACTION_TYPES = ("humans", "infected", "robots")
UNIT_FACTION = {
    UnitType.VANGUARD: "humans",
    UnitType.RANGER: "humans",
    UnitType.SKIRMISHER: "humans",
    UnitType.RAIDER: "humans",
    UnitType.BMP: "humans",
    UnitType.SWARMLING: "infected",
    UnitType.SPITTER: "infected",
    UnitType.BRUTE: "infected",
    UnitType.RAVAGER: "infected",
    UnitType.BEHEMOTH: "infected",
    UnitType.WALKER: "robots",
    UnitType.LASER: "robots",
    UnitType.DRONE: "robots",
    UnitType.GUARDIAN_MECH: "robots",
    UnitType.SIEGE_PLATFORM: "robots",
}
UNIT_ROSTER_BY_FACTION = {
    faction: tuple(unit for unit, owner_faction in UNIT_FACTION.items() if owner_faction == faction)
    for faction in FACTION_TYPES
}
BASIC_UNITS_BY_FACTION = {
    "humans": (UnitType.VANGUARD, UnitType.RANGER, UnitType.SKIRMISHER),
    "infected": (UnitType.SWARMLING, UnitType.SPITTER, UnitType.BRUTE),
    "robots": (UnitType.WALKER, UnitType.LASER, UnitType.DRONE),
}
TECH1_UNITS_BY_FACTION = {
    "humans": (UnitType.RAIDER,),
    "infected": (UnitType.RAVAGER,),
    "robots": (UnitType.GUARDIAN_MECH,),
}
TECH2_UNITS_BY_FACTION = {
    "humans": (UnitType.BMP,),
    "infected": (UnitType.BEHEMOTH,),
    "robots": (UnitType.SIEGE_PLATFORM,),
}
FACTION_UNIQUE_BUILDINGS = {
    "humans": BuildingType.BOOSTER,
    "robots": BuildingType.POWER,
    "infected": BuildingType.LIFE,
}

CONTROL_POINT_BUILDINGS = (BuildingType.TURRET,)
CORE_BUILDINGS = (
    BuildingType.BOOSTER,
    BuildingType.POWER,
    BuildingType.LIFE,
    BuildingType.ECONOMY_COLLECTOR,
    BuildingType.TECH_1,
    BuildingType.TECH_2,
)
MAX_CONTROL_POINT_BUILDINGS = 1
MAX_CORE_BUILDINGS = 2

BUILDING_COST = {
    BuildingType.TURRET: 24,
    BuildingType.BOOSTER: 18,
    BuildingType.POWER: 22,
    BuildingType.LIFE: 22,
    BuildingType.ECONOMY_COLLECTOR: 30,
    BuildingType.TECH_1: 28,
    BuildingType.TECH_2: 40,
}

BUILDING_BUILD_TICKS = {
    BuildingType.TURRET: 14,
    BuildingType.BOOSTER: 10,
    BuildingType.POWER: 12,
    BuildingType.LIFE: 12,
    BuildingType.ECONOMY_COLLECTOR: 20,
    BuildingType.TECH_1: 16,
    BuildingType.TECH_2: 22,
}

BUILDING_STATS = {
    BuildingType.TURRET: {
        "hp": 125,
        "atk": 5,
        "range": 4.0,
        "attack_rate": 3,
        "income": 0,
    },
    BuildingType.BOOSTER: {
        "hp": 28,
        "atk": 0,
        "range": 0.0,
        "attack_rate": 1,
        "income": 0,
        "faction": "humans",
    },
    BuildingType.POWER: {
        "hp": 34,
        "atk": 0,
        "range": 0.0,
        "attack_rate": 1,
        "income": 0,
        "faction": "robots",
    },
    BuildingType.LIFE: {
        "hp": 34,
        "atk": 0,
        "range": 0.0,
        "attack_rate": 1,
        "income": 0,
        "faction": "infected",
    },
    BuildingType.ECONOMY_COLLECTOR: {
        "hp": 30,
        "atk": 0,
        "range": 0.0,
        "attack_rate": 1,
        "income": 2,
    },
    BuildingType.TECH_1: {
        "hp": 36,
        "atk": 0,
        "range": 0.0,
        "attack_rate": 1,
        "income": 0,
    },
    BuildingType.TECH_2: {
        "hp": 42,
        "atk": 0,
        "range": 0.0,
        "attack_rate": 1,
        "income": 0,
    },
}

BOOSTER_SPEED_BONUS = 1.0
BOOSTER_SPEED_TICKS = 3
POWER_HP_BONUS = 2
LIFE_SPAWN_INTERVAL_TICKS = 6
LIFE_SPAWN_UNIT = UnitType.SWARMLING
BUILDING_REGEN_DELAY_TICKS = 8
BUILDING_ZERO_HP_REGEN_DELAY_MULTIPLIER = 2
BUILDING_REGEN_HP_PER_TICK = 5

CORE_HP = 80
CORE_ATK = 7
CORE_ATTACK_RATE = 2
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

BARON_HP = 300
BARON_ATK = 13
BARON_ATTACK_RATE = 3
BARON_SECONDARY_ATK = 3
BARON_SECONDARY_ATTACK_RATE = 5
BARON_RESPAWN_TICKS = 50
ALLIED_LORD_PUSH_DELAY_TICKS = 2
# Combat maps are a 1D [0..20] line, so map radius is 10.
LORD_ATTACK_RANGE = 4.6
CORE_ATTACK_RANGE = 5.0
NEUTRAL_GUARD_HP = 4
NEUTRAL_GUARD_ATK = 3

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
