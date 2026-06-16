"""Manager game plan — the franchise personas' second job.

The six franchise managers (src/regionals/franchises.py) already draft players.
Here they get their per-match job: set the team's *game plan* using the order
vocabulary the nets were trained on (see docs/cloudcode/LLM_MANAGER_ORDERS_SPEC.md):
a team ``tempo`` + ``win_path`` and a ``role`` for each of the three positions.

The plan is converted to the env/obs order format and handed to the policy nets,
which follow it as far as their trained "obedience" allows — while still playing
to win. Without an API key the manager falls back to a deterministic, in-character
plan derived from its persona, so the system always produces a usable plan.
"""
from __future__ import annotations

from typing import Optional

from src.training.spaces import (
    ORDER_ROLE,
    ORDER_ROLE_DEFAULT,
    ORDER_TEMPO,
    ORDER_TEMPO_DEFAULT,
    ORDER_WIN_PATH,
    ORDER_WIN_PATH_DEFAULT,
)

# Persona-flavoured defaults — used as the no-LLM fallback AND as a strong prior
# the LLM can override. Each franchise plays to its drafting philosophy.
# (tempo, win_path, [role per position])
PERSONA_DEFAULT_PLANS: dict[str, tuple[str, str, list[str]]] = {
    "iron-falcons": ("aggressive", "core_rush", ["carry_push", "carry_push", "roam_objective"]),
    "quantum-owls": ("balanced", "economy", ["tech_eco", "carry_push", "anchor_defend"]),
    "crimson-pact": ("all_in", "baron_lord", ["carry_push", "roam_objective", "carry_push"]),
    "northwind": ("turtle", "economy", ["anchor_defend", "tech_eco", "support_flex"]),
    "solar-drift": ("balanced", "outlast", ["support_flex", "anchor_defend", "carry_push"]),
    "black-lotus": ("aggressive", "economy", ["carry_push", "tech_eco", "roam_objective"]),
}
_GENERIC_DEFAULT = (ORDER_TEMPO_DEFAULT, ORDER_WIN_PATH_DEFAULT,
                    [ORDER_ROLE_DEFAULT, ORDER_ROLE_DEFAULT, ORDER_ROLE_DEFAULT])

MANAGER_SYSTEM_RULES = (
    "\n\nТы ставишь план на ОДИН матч для своей тройки. Главное правило: команда "
    "ВСЕГДА играет на победу — план это КАК играть, а не «проиграй ради стиля». "
    "Выбери в характере своей философии:\n"
    f"- tempo (темп/риск всей команды): {', '.join(ORDER_TEMPO)};\n"
    f"- win_path (путь к победе): {', '.join(ORDER_WIN_PATH)};\n"
    f"- role для каждого игрока: {', '.join(ORDER_ROLE)}.\n"
    "Отвечай только JSON по схеме, rationale — 1 короткая фраза."
)


def persona_default_plan(franchise_id: str, player_ids: list[str]) -> dict:
    tempo, win_path, roles = PERSONA_DEFAULT_PLANS.get(franchise_id, _GENERIC_DEFAULT)
    return {
        "tempo": tempo,
        "win_path": win_path,
        "roles": {pid: roles[i] if i < len(roles) else ORDER_ROLE_DEFAULT
                  for i, pid in enumerate(player_ids)},
        "rationale": "default-by-persona",
    }


def _valid(plan: dict, player_ids: list[str]) -> bool:
    try:
        return (
            plan.get("tempo") in ORDER_TEMPO
            and plan.get("win_path") in ORDER_WIN_PATH
            and all((plan.get("roles") or {}).get(pid) in ORDER_ROLE for pid in player_ids)
        )
    except (TypeError, AttributeError):
        return False


def match_plan(
    franchise: dict,
    situation: str,
    player_ids: list[str],
    llm=None,
) -> dict:
    """Produce a game plan for ``player_ids`` from this franchise's manager.

    ``franchise`` is a persona dict (id + persona text). ``situation`` is a
    compact, caller-rendered summary (own roster, opponent, series score, recent
    results). Falls back to the persona default on any LLM failure or absence.
    """
    fid = franchise.get("franchise_id", "")
    fallback = persona_default_plan(fid, player_ids)
    if llm is None:
        return fallback
    try:
        plan = llm.plan(
            system=franchise.get("persona", "") + MANAGER_SYSTEM_RULES,
            prompt=situation,
            player_ids=player_ids,
            tempo=list(ORDER_TEMPO),
            win_path=list(ORDER_WIN_PATH),
            role=list(ORDER_ROLE),
        )
    except Exception:
        return fallback
    return plan if _valid(plan, player_ids) else fallback


def plan_to_env_order(plan: dict, player_ids: list[str]) -> dict:
    """Convert a manager plan to the obs/env order format consumed by the nets:
    ``{player_id: {"tempo", "win_path", "role"}}``."""
    roles = plan.get("roles", {}) or {}
    tempo = plan.get("tempo", ORDER_TEMPO_DEFAULT)
    win_path = plan.get("win_path", ORDER_WIN_PATH_DEFAULT)
    return {
        pid: {"tempo": tempo, "win_path": win_path, "role": roles.get(pid, ORDER_ROLE_DEFAULT)}
        for pid in player_ids
    }
