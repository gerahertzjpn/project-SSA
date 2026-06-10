"""The tournament commentator: one persistent persona with running memory.

The commentator's memory is the list of everything he has already said this
tournament (opening speech, match comments, final report). Each new call gets
the persona plus a compact bullet list of his previous lines, so he stays
consistent ("as I said after day 3...") without re-sending match data.

Token economy: comments are hard-capped by ``max_tokens`` and instructed to
stay short; memory is truncated to the most recent entries.
"""

from __future__ import annotations

COMMENTATOR_NAME = "Гарри «Голос» Вольский"

_PERSONA = (
    "Ты Гарри «Голос» Вольский — легендарный комментатор лиги Regionals. Энергичный, "
    "остроумный, с фирменными прозвищами для игроков. Ты ПОМНИШЬ всё, что говорил раньше "
    "(см. «твои прошлые реплики»), ссылайся на свои прогнозы и прошлые матчи, развивай "
    "сюжетные линии сезона. Пиши по-русски, кратко и живо."
)

MEMORY_LIMIT = 12  # most recent lines fed back into context


def system_prompt(memory: list[dict]) -> str:
    parts = [_PERSONA]
    if memory:
        recent = memory[-MEMORY_LIMIT:]
        bullets = "\n".join(f"- [{m['kind']}, день {m.get('day', '?')}] {m['text']}" for m in recent)
        parts.append(f"Твои прошлые реплики этого турнира:\n{bullets}")
    return "\n\n".join(parts)


def opening_prompt(name: str, teams: list[dict], draft_log: dict) -> str:
    rosters = []
    for team in teams:
        players = ", ".join(f"{p['name']} ({p['faction']})" for p in team["players"])
        rosters.append(f"{team['name']} (менеджер {team.get('manager', '?')}): {players}")
    picks = []
    first_round = (draft_log.get("rounds") or [{}])[0]
    for pick in first_round.get("picks", []):
        reason = f" — «{pick['reason']}»" if pick.get("reason") else ""
        picks.append(f"пик {pick['pick']}: {pick['team_name']} берёт {pick['nickname']}{reason}")
    return (
        f"Открытие турнира «{name}»: 6 команд, круговая лига Bo3, затем плей-офф топ-3 (Bo5).\n"
        f"Составы после драфта:\n" + "\n".join(rosters) + "\n"
        f"Первый раунд драфта:\n" + "\n".join(picks) + "\n"
        "Сделай вступительную речь: представь турнир, отметь 2-3 самых интересных решения "
        "драфта, дай смелый прогноз. До 120 слов."
    )


def encounter_prompt(enc: dict, stage_label: str) -> str:
    games = []
    for game in enc.get("games", []):
        games.append(f"игра {game['index'] + 1}: победа {'хозяев' if game['winner'] == 'home' else 'гостей' if game['winner'] == 'away' else 'ничья'} за {game['ticks']} тиков")
    top = enc.get("top_performers") or []
    top_line = ", ".join(f"{p['name']} (импакт {p['impact']:.2f})" for p in top[:2])
    return (
        f"{stage_label}, день {enc['day']}: {enc['home_name']} vs {enc['away_name']}, "
        f"счёт серии {enc['home_game_wins']}:{enc['away_game_wins']}.\n"
        + "; ".join(games) + (f"\nГерои встречи: {top_line}." if top_line else "") + "\n"
        "Дай краткую сводку этой встречи (до 60 слов), свяжи с ходом сезона."
    )


def final_prompt(name: str, placements: list[dict], champion_name: str) -> str:
    table = "\n".join(
        f"{p['place']}. {p['team_name']} — {p['prize']} cr" + (" + бонус чемпиона" if p["place"] == 1 else "")
        for p in placements)
    return (
        f"Турнир «{name}» завершён. Чемпион — {champion_name}!\n"
        f"Итоговые места и призовые:\n{table}\n"
        "Подведи итоги турнира: чемпион, главная сенсация, разочарование, вспомни свой "
        "прогноз из открытия (сбылся или нет). До 120 слов."
    )


def fallback_text(kind: str) -> str:
    texts = {
        "opening": "Турнир стартовал! (комментатор недоступен: LLM не подключена)",
        "match": "Встреча завершена. (комментатор недоступен: LLM не подключена)",
        "final": "Турнир завершён! (комментатор недоступен: LLM не подключена)",
    }
    return texts.get(kind, texts["match"])
