"""League layer: templates, scheduling, simulation, standings and persistence.

The league sits on top of the existing 3v3 match simulator. A league groups
teams of players (each player = model + faction), generates a randomized
round-robin schedule of Best-of-N encounters, simulates every game up front,
and exposes standings/history while hiding unrevealed results from the UI.
"""

from .templates import LEAGUE_TEMPLATES, get_template, list_templates
from .league import (
    LeagueError,
    create_league,
    load_league,
    list_leagues,
    advance_day,
    league_public_view,
    encounter_detail,
)

__all__ = [
    "LEAGUE_TEMPLATES",
    "get_template",
    "list_templates",
    "LeagueError",
    "create_league",
    "load_league",
    "list_leagues",
    "advance_day",
    "league_public_view",
    "encounter_detail",
]
