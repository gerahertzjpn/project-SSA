"""Team Elo — chess-style power ranking from official matches only.

Only league + playoff series move a team's Elo. Unlike player SoloQ, the
modifier here is the **series score margin**: a clean sweep moves more Elo than
a deciding-game series ("чем больше/меньше карт отдали — тем больше/меньше Elo").
"""
from __future__ import annotations

from src.regionals.elo import expected_score

K_FACTOR = 16.0   # owner: one series should move Power gently (was 32 — single games jumped too hard)
START_ELO = 1000.0


def win_threshold(best_of: int) -> int:
    """Maps needed to win a Bo-N series (Bo3 -> 2, Bo5 -> 3)."""
    return best_of // 2 + 1


def margin_factor(maps_a: int, maps_b: int, best_of: int) -> float:
    """Score-margin weight in (0,1]: 1.0 for a sweep, smaller for a close series."""
    thr = win_threshold(best_of)
    return abs(maps_a - maps_b) / thr


def series_elo_delta(
    elo_a: float,
    elo_b: float,
    maps_a: int,
    maps_b: int,
    best_of: int,
    k: float = K_FACTOR,
) -> float:
    """Elo change for team A from one official series (zero-sum: B gets -delta)."""
    if maps_a == maps_b:
        raise ValueError("official series cannot be drawn")
    actual_a = 1.0 if maps_a > maps_b else 0.0
    expected_a = expected_score(elo_a, elo_b)
    return k * margin_factor(maps_a, maps_b, best_of) * (actual_a - expected_a)


def apply_series_result(
    world: dict,
    team_a_id: str,
    team_b_id: str,
    maps_a: int,
    maps_b: int,
    best_of: int,
) -> float:
    """Update both teams' Elo from a series result. Returns A's delta."""
    teams = world["teams"]
    a, b = teams[team_a_id], teams[team_b_id]
    delta = series_elo_delta(a["elo"], b["elo"], maps_a, maps_b, best_of)
    a["elo"] = round(a["elo"] + delta, 2)
    b["elo"] = round(b["elo"] - delta, 2)
    a["elo_history"].append(a["elo"])
    b["elo_history"].append(b["elo"])
    return delta
