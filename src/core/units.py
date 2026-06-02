from dataclasses import dataclass
from .constants import BARON_ATTACK_RATE, BARON_ATK, BARON_HP, UnitType

# R-P-S: VANGUARD > SKIRMISHER > RANGER > VANGUARD
# Reflected through unit stats: rangers shred vanguards at range before they close,
# skirmishers close quickly and kill rangers, vanguards absorb skirmisher hits.
UNIT_STATS: dict[UnitType, dict] = {
    UnitType.VANGUARD: {
        "hp": 10,
        "atk": 2,
        "range": 1.0,
        "speed": 1.0,  # positions per combat tick
        "slot": "front",
        "base_pos": 2,  # starting position offset from own side
    },
    UnitType.RANGER: {
        "hp": 3,
        "atk": 5,
        "range": 3.0,
        "speed": 0.75,
        "slot": "back",
        "base_pos": 10,
    },
    UnitType.SKIRMISHER: {
        "hp": 5,
        "atk": 3,
        "range": 1.5,
        "speed": 1.5,
        "slot": "mid",
        "base_pos": 5,
    },
    UnitType.LORD: {
        "hp": BARON_HP,
        "atk": BARON_ATK,
        "attack_rate": BARON_ATTACK_RATE,
        "range": 2.0,
        "speed": 1.0,
        "slot": "front",
        "base_pos": 1,
    },
}

# Neutral unit stats (like vanguard but with configurable HP/ATK)
NEUTRAL_UNIT_BASE = {
    "hp": 1,         # multiplied by neutral_strength
    "atk": 1,        # total neutral atk is set dynamically from guard strength
    "range": 3.0,
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

    def is_alive(self) -> bool:
        return self.hp > 0
