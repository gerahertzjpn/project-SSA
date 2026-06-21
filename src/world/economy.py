"""Economy — prize money (teams only) + slow-moving player salaries.

Money is awarded to TEAMS only, never players. Players earn a **salary** each
season (paid to everyone on the roster, even non-playing members). Salaries move
slowly: one bad season barely dents a star; three bad seasons cut deep.

All tunables live in ``EconomyConfig`` so the balance can be tuned in one place.
See docs/cloudcode/WORLD_MODE_SPEC.md §5 for the rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ATTR_KEYS = ("attack", "defense", "economy", "tempo", "consistency", "potential")


def _default_upper_prizes() -> list[int]:
    return [1000, 800, 650, 520, 420, 340, 280, 230, 190, 160]


def _default_lower_prizes() -> list[int]:
    return [450, 380, 320, 270, 230, 195, 165, 140, 120, 100, 85, 70]


@dataclass
class EconomyConfig:
    # Placement prizes by finishing position (index 0 = 1st).
    upper_prizes: list[int] = field(default_factory=_default_upper_prizes)
    lower_prizes: list[int] = field(default_factory=_default_lower_prizes)
    # Playoff bonus — only the GRAND FINAL winner gets it.
    upper_playoff_bonus: int = 800
    lower_playoff_bonus: int = 500
    # Starting treasury seed at the first regular season (Season 0 gives none).
    upper_seed_treasury: int = 300
    lower_seed_treasury: int = 200
    # Salary model.
    salary_base: float = 50.0       # salary of a 1000-rating, average-attr player
    salary_alpha: float = 0.25      # EWMA smoothing (slow-moving salaries)
    salary_min: float = 8.0
    rating_pivot: float = 1000.0
    rating_scale: float = 500.0     # +rating_scale rating ~ +1.0 to the value multiplier


CONFIG = EconomyConfig()


# --------------------------------------------------------------------------- #
# Prizes
# --------------------------------------------------------------------------- #
def placement_prize(league: str, place: int, config: EconomyConfig = CONFIG) -> int:
    """Prize for finishing ``place`` (1-based) in a league."""
    table = config.upper_prizes if league == "upper" else config.lower_prizes
    if place < 1 or place > len(table):
        return 0
    return table[place - 1]


def playoff_bonus(league: str, config: EconomyConfig = CONFIG) -> int:
    return config.upper_playoff_bonus if league == "upper" else config.lower_playoff_bonus


# --------------------------------------------------------------------------- #
# Salary
# --------------------------------------------------------------------------- #
def player_value(player: dict, config: EconomyConfig = CONFIG) -> float:
    """Normalized 'worth' multiplier (~1.0 at a 1000-rating average player).

    Dominated by SoloQ rating, with attribute 'coolness' and a small recent-
    performance term (win-rate over the player's official games this/last season).
    """
    rating = float(player.get("soloq_rating", config.rating_pivot))
    rating_mult = 1.0 + (rating - config.rating_pivot) / config.rating_scale

    attrs = player.get("attributes", {})
    if attrs:
        avg_attr = sum(attrs.get(k, 50) for k in ATTR_KEYS) / len(ATTR_KEYS)
    else:
        avg_attr = 50.0
    attr_mult = 0.6 + (avg_attr / 50.0) * 0.4  # 0.6..1.4 around an average card

    perf = player.get("recent_winrate")
    perf_mult = 1.0 if perf is None else (0.85 + 0.30 * float(perf))  # 0.85..1.15

    return max(0.1, rating_mult * attr_mult * perf_mult)


def salary_target(player: dict, config: EconomyConfig = CONFIG) -> float:
    """The salary this player *should* earn now (pre-smoothing)."""
    return max(config.salary_min, config.salary_base * player_value(player, config))


def smoothed_salary(current: float, target: float, config: EconomyConfig = CONFIG) -> float:
    """EWMA step toward the target — the 'slow but sure' salary movement."""
    if current <= 0:
        return round(target, 2)  # first contract jumps straight to target
    return round(current + config.salary_alpha * (target - current), 2)


def update_player_salary(player: dict, config: EconomyConfig = CONFIG) -> float:
    """Advance a player's salary one season toward its target; records history."""
    target = salary_target(player, config)
    new_salary = smoothed_salary(float(player.get("salary", 0.0)), target, config)
    player["salary"] = new_salary
    player.setdefault("salary_history", []).append(new_salary)
    return new_salary


# --------------------------------------------------------------------------- #
# Apply to a world (one regular-season settlement)
# --------------------------------------------------------------------------- #
def award_prizes(world: dict, placements: list[dict], config: EconomyConfig = CONFIG) -> None:
    """Credit placement + grand-final prizes. ``placements`` items:
    ``{team_id, league, place, grand_final_winner: bool}``."""
    for row in placements:
        team = world["teams"][row["team_id"]]
        prize = placement_prize(row["league"], row["place"], config)
        if row.get("grand_final_winner"):
            prize += playoff_bonus(row["league"], config)
        team["treasury"] += prize


def pay_salaries(world: dict, config: EconomyConfig = CONFIG) -> dict:
    """Advance + deduct every rostered player's salary from their team. Rookies
    are paid only after their first played season. Returns per-team totals."""
    totals: dict[str, float] = {}
    for team in world["teams"].values():
        spent = 0.0
        for pid in team["roster"]:
            player = world["players"][pid]
            if player.get("status") == "rookie" and player.get("seasons_played", 0) < 1:
                continue  # rookie: salary starts after the first played season
            spent += update_player_salary(player, config)
        team["treasury"] = round(team["treasury"] - spent, 2)
        totals[team["team_id"]] = round(spent, 2)
    return totals
