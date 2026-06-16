"""UI label mapping. Data fields keep their names; the UI shows friendlier labels.

The team rating (chess-style Elo from official matches) is displayed as **Power**
in the UI per the owner's preference. The underlying field stays ``team['elo']``
so the engine, tests and persistence are unchanged.
"""
from __future__ import annotations

TEAM_RATING_LABEL = "Power"   # team['elo'] is shown as 'Power' in the UI


def team_power(team: dict) -> float:
    """The team's Power rating (its Elo) — for UI display."""
    return team.get("elo", 1000.0)
