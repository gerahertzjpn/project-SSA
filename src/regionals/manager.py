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

MANAGER_SYSTEM_RULES = (
    "\n\nТы ставишь план на ОДИН матч для своей тройки. Главное правило: команда "
    "ВСЕГДА играет на победу — план это КАК играть, а не «проиграй ради стиля». "
    "Выбери в характере своей философии:\n"
    f"- tempo (темп/риск всей команды): {', '.join(ORDER_TEMPO)};\n"
    f"- win_path (путь к победе): {', '.join(ORDER_WIN_PATH)};\n"
    f"- role для каждого игрока: {', '.join(ORDER_ROLE)}.\n"
    "Отвечай только JSON по схеме, rationale — 1 короткая фраза."
)


def _franchise_by_id() -> dict:
    from src.regionals.franchises import FRANCHISES  # local import avoids cycles
    return {f["franchise_id"]: f for f in FRANCHISES}


def persona_default_plan(franchise, player_ids: list[str]) -> dict:
    """In-character default plan from the franchise's own ``default_plan``.

    ``franchise`` may be the franchise dict or its id. Scales to any pool size —
    each franchise carries its plan (no hardcoded per-franchise table).
    """
    fr = franchise if isinstance(franchise, dict) else _franchise_by_id().get(franchise, {})
    dp = fr.get("default_plan") or {}
    tempo = dp.get("tempo", ORDER_TEMPO_DEFAULT)
    win_path = dp.get("win_path", ORDER_WIN_PATH_DEFAULT)
    roles = dp.get("roles") or []
    return {
        "tempo": tempo,
        "win_path": win_path,
        "roles": {pid: (roles[i] if i < len(roles) else ORDER_ROLE_DEFAULT)
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
    fallback = persona_default_plan(franchise, player_ids)
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


def team_match_order(
    franchise: dict,
    situation: str,
    player_ids: list[str],
    llm=None,
) -> dict:
    """One manager's full per-match order for its team, in env/obs format:
    ``{player_id: {tempo, win_path, role}}``. Convenience wrapper around
    match_plan + plan_to_env_order."""
    plan = match_plan(franchise, situation, player_ids, llm=llm)
    return plan_to_env_order(plan, player_ids)


class ManagerMemory:
    """Lets a manager adapt across a series/season (user choice: learns from results).

    Tracks, per franchise, its recent plan→outcome so the next plan prompt can show
    what it tried and whether it worked. Kept compact for token cost.
    """

    def __init__(self, keep: int = 4):
        self.keep = int(keep)
        self._log: dict[str, list[dict]] = {}

    def record(self, franchise_id: str, opponent: str, plan: dict, result: str) -> None:
        entries = self._log.setdefault(franchise_id, [])
        entries.append({
            "opponent": opponent,
            "tempo": plan.get("tempo"),
            "win_path": plan.get("win_path"),
            "result": result,  # "W" / "L" / "D"
        })
        del entries[: -self.keep]

    def recent_block(self, franchise_id: str) -> str:
        entries = self._log.get(franchise_id, [])
        if not entries:
            return "Прошлых планов нет — это первый матч."
        parts = [f"vs {e['opponent']} ({e['tempo']}/{e['win_path']})→{e['result']}" for e in entries]
        return "Твои последние планы: " + "; ".join(parts) + "."


def build_situation(
    opponent_name: str,
    own_roster: str,
    opponent_roster: str,
    series_score: str,
    recent_results: str,
) -> str:
    """Assemble the compact situation prompt a manager sees before a match.

    Callers (the regionals match loop) render the roster/score strings from real
    card stats; this keeps manager.py decoupled and cheap on tokens.
    """
    return (
        f"Матч против {opponent_name}. Счёт серии: {series_score}.\n"
        f"Твоя тройка: {own_roster}\n"
        f"Состав соперника: {opponent_roster}\n"
        f"{recent_results}\n"
        "Поставь план на этот матч в своём стиле, но цель — победа."
    )
