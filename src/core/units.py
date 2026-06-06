from dataclasses import dataclass
from .constants import (
    BARON_ATTACK_RATE,
    BARON_ATK,
    BARON_HP,
    BARON_SECONDARY_ATTACK_RATE,
    BARON_SECONDARY_ATK,
    LORD_ATTACK_RANGE,
    NEUTRAL_GUARD_ATK,
    NEUTRAL_GUARD_HP,
    UnitType,
)

# R-P-S: VANGUARD > SKIRMISHER > RANGER > VANGUARD
# Reflected through unit stats: rangers shred vanguards at range before they close,
# skirmishers close quickly and kill rangers, vanguards absorb skirmisher hits.
UNIT_STATS: dict[UnitType, dict] = {
    UnitType.VANGUARD: {
        "hp": 14,
        "atk": 3,
        "range": 1.0,
        "speed": 1.5,  # positions per combat tick
        "slot": "front",
        "base_pos": 3.2,  # starting position offset from own side
    },
    UnitType.RANGER: {
        "hp": 4,
        "atk": 10,
        "range": 3.5,
        "speed": 1.5,
        "slot": "back",
        "base_pos": 0.8,
    },
    UnitType.SKIRMISHER: {
        "hp": 3,
        "atk": 4,
        "range": 1.0,
        "speed": 3.0,
        "slot": "mid",
        "base_pos": 1.9,
    },
    UnitType.RAIDER: {
        "hp": 56,
        "atk": 11,
        "attack_rate": 2,
        "range": 2.8,
        "speed": 3.2,
        "slot": "mid",
        "base_pos": 2.2,
    },
    UnitType.BMP: {
        "hp": 230,
        "atk": 11,
        "attack_rate": 5,
        "range": 4.0,
        "speed": 1.0,
        "slot": "back",
        "base_pos": 1.0,
    },
    UnitType.SWARMLING: {
        "hp": 2,
        "atk": 2,
        "range": 0.8,
        "speed": 3.6,
        "slot": "front",
        "base_pos": 2.4,
    },
    UnitType.SPITTER: {
        "hp": 3,
        "atk": 5,
        "range": 2.5,
        "speed": 2.2,
        "slot": "back",
        "base_pos": 1.1,
    },
    UnitType.BRUTE: {
        "hp": 24,
        "atk": 4,
        "range": 1.0,
        "speed": 1.1,
        "slot": "front",
        "base_pos": 3.2,
    },
    UnitType.RAVAGER: {
        "hp": 72,
        "atk": 11,
        "attack_rate": 2,
        "range": 3.5,
        "speed": 3.4,
        "slot": "mid",
        "base_pos": 2.3,
    },
    UnitType.BEHEMOTH: {
        "hp": 360,
        "atk": 12,
        "attack_rate": 6,
        "range": 3.0,
        "speed": 0.8,
        "slot": "front",
        "base_pos": 3.0,
    },
    UnitType.WALKER: {
        "hp": 34,
        "atk": 5,
        "range": 1.2,
        "speed": 1.0,
        "slot": "front",
        "base_pos": 3.0,
    },
    UnitType.LASER: {
        "hp": 9,
        "atk": 16,
        "range": 4.0,
        "speed": 1.0,
        "slot": "back",
        "base_pos": 0.9,
    },
    UnitType.DRONE: {
        "hp": 6,
        "atk": 5,
        "range": 1.1,
        "speed": 3.2,
        "slot": "mid",
        "base_pos": 2.0,
    },
    UnitType.GUARDIAN_MECH: {
        "hp": 104,
        "atk": 10,
        "attack_rate": 2,
        "range": 3.6,
        "speed": 1.3,
        "slot": "mid",
        "base_pos": 2.2,
    },
    UnitType.SIEGE_PLATFORM: {
        "hp": 260,
        "atk": 12,
        "attack_rate": 6,
        "range": 4.8,
        "speed": 0.6,
        "slot": "back",
        "base_pos": 0.8,
    },
    UnitType.LORD: {
        "hp": BARON_HP,
        "atk": BARON_ATK,
        "attack_rate": BARON_ATTACK_RATE,
        "secondary_atk": BARON_SECONDARY_ATK,
        "secondary_attack_rate": BARON_SECONDARY_ATTACK_RATE,
        "range": LORD_ATTACK_RANGE,
        "speed": 1.0,
        "slot": "front",
        "base_pos": 3.4,
    },
}

# Neutral unit stats (like vanguard but with configurable HP/ATK)
NEUTRAL_UNIT_BASE = {
    "hp": NEUTRAL_GUARD_HP,
    "atk": NEUTRAL_GUARD_ATK,
    "range": 3.5,
    "speed": 0.0,    # neutrals don't advance
    "slot": "front",
    "base_pos": 1,
}


@dataclass
class UnitInstance:
    uid: int
    unit_type: str       # UnitType.value or 'NEUTRAL' or 'CORE'
    team: str            # player_id or 'NEUTRAL'
    hp: float
    max_hp: float
    atk: float
    range: float
    speed: float
    pos: float           # 1D position (attackers start low, defenders start high)
    attack_rate: int = 1
    secondary_atk: float = 0.0
    secondary_attack_rate: int = 0
    move_direction: float = 0.0
    render_offset: float = 0.0
    x: float = 0.0
    y: float = 0.0
    forward_x: float = 1.0
    forward_y: float = 0.0
    visual_id: str = ""

    def is_alive(self) -> bool:
        return self.hp > 0
