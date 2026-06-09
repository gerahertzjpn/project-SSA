"""League templates.

A template describes the *rules shape* of a league: how many teams, how many
players per team, the schedule format, and how an encounter is decided. New
templates (different sizes, double round-robin, playoffs, ...) can be added here
without touching the scheduling/simulation code, which reads everything from the
template dict.
"""

from __future__ import annotations

from copy import deepcopy


# Each template is a plain dict so it serializes straight to JSON for the API
# and is stored verbatim inside every league file (rules are frozen at
# creation time, independent of later code changes).
LEAGUE_TEMPLATES: dict[str, dict] = {
    "classic_6x3": {
        "id": "classic_6x3",
        "name": "Classic 6x3",
        "description": (
            "Six teams of three players. Single round-robin, every team plays "
            "every other once. Encounters are Best-of-3, one encounter per day."
        ),
        "team_count": 6,
        "players_per_team": 3,
        "schedule": "round_robin",          # all-vs-all, single circle
        "encounters_per_day": 1,            # one match per day
        "best_of": 3,                       # Bo3: first to 2 game wins
        "points_win": 3,
        "points_draw": 1,
        "points_loss": 0,
    },
}

DEFAULT_TEMPLATE_ID = "classic_6x3"


def list_templates() -> list[dict]:
    """Return template summaries for the UI (deep-copied, safe to mutate)."""
    return [deepcopy(template) for template in LEAGUE_TEMPLATES.values()]


def get_template(template_id: str) -> dict:
    template = LEAGUE_TEMPLATES.get(str(template_id))
    if template is None:
        choices = ", ".join(sorted(LEAGUE_TEMPLATES))
        raise KeyError(f"Unknown league template {template_id!r}; expected one of: {choices}")
    return deepcopy(template)
