"""Regionals templates ("versions" of the mode).

A regionals template = a league stage shape + a playoff stage shape. New
versions (different team counts, playoff brackets, series lengths, ...) are
added here and become selectable in the UI; the engine reads everything from
the template dict, so no scheduling/simulation code changes are needed.
"""

from __future__ import annotations

from copy import deepcopy


REGIONALS_TEMPLATES: dict[str, dict] = {
    "regionals_v1": {
        "id": "regionals_v1",
        "name": "Regionals v1 — Classic 6 + Playoff",
        "description": (
            "Six franchise teams draft 3 players each from the 18-player pool "
            "(LLM managers, 3 snake-free random-order rounds). Single "
            "round-robin Bo3 league, then a step-ladder playoff: 2nd vs 3rd, "
            "the winner faces 1st in the grand final. Playoff series are Bo5."
        ),
        "team_count": 6,
        "players_per_team": 3,
        "draft_rounds": 3,
        # league stage
        "league": {
            "schedule": "round_robin",
            "encounters_per_day": 1,
            "best_of": 3,
            "points_win": 3,
            "points_draw": 1,
            "points_loss": 0,
        },
        # playoff stage
        "playoff": {
            "format": "stepladder_top3",   # 2nd vs 3rd -> winner vs 1st
            "best_of": 5,
        },
        # prize money (credits) by final placement, index 0 = champion
        "prizes": [500, 300, 200, 120, 80, 50],
        "champion_bonus": 250,
    },
}

DEFAULT_TEMPLATE_ID = "regionals_v1"


def list_templates() -> list[dict]:
    """Template summaries for the UI (deep-copied, safe to mutate)."""
    return [deepcopy(template) for template in REGIONALS_TEMPLATES.values()]


def get_template(template_id: str) -> dict:
    template = REGIONALS_TEMPLATES.get(str(template_id))
    if template is None:
        choices = ", ".join(sorted(REGIONALS_TEMPLATES))
        raise KeyError(f"Unknown regionals template {template_id!r}; expected one of: {choices}")
    return deepcopy(template)
