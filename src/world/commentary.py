"""World commentator — Гарри «Голос» Вольский calls the season (reuses the regionals persona).

Template lines ALWAYS work (deterministic, no cost); when an LLM is supplied the line is rewritten
in persona (richer, still grounded in the same facts). Lines are stored on ``world['commentary']``
so the UI can replay the broadcast: season open, season close (champions / promotion / relegation),
off-season, and — optionally — a match.
"""
from __future__ import annotations

from typing import Optional

from src.world import events


def _persona() -> str:
    from src.regionals.commentary import system_prompt  # the Вольский persona + recent-lines memory
    return system_prompt([])


def _enhance(llm, base: str, brief: str) -> str:
    """Rewrite the factual ``brief`` in persona; fall back to the template ``base`` on any error."""
    if llm is None:
        return base
    try:
        return llm.say(_persona(), f"{brief}\nНапиши 2–3 ярких предложения от лица комментатора "
                       f"(по-русски), не выдумывая фактов.", max_tokens=240).strip() or base
    except Exception:
        return base


def _team(world: dict, tid) -> str:
    return world["teams"][tid]["name"] if tid in world.get("teams", {}) else (tid or "—")


def add_line(world: dict, kind: str, text: str, *, feed: Optional[list] = None) -> str:
    world.setdefault("commentary", []).append(
        {"season": world.get("season", 0), "kind": kind, "text": text})
    if len(world["commentary"]) > 120:
        world["commentary"] = world["commentary"][-120:]
    if feed is not None:
        events.emit(feed, "comment", text)
    return text


def season_open(world: dict, *, llm=None, feed: Optional[list] = None) -> str:
    from src.world import views
    s = world.get("season", 0)
    top = sorted(world["teams"].values(), key=lambda t: -views.team_power(world, t))[:3]
    names = [t["name"] for t in top]
    base = f"Сезон {s} начинается! По силе впереди: {', '.join(names)}. Поехали!"
    return add_line(world, "season_open",
                    _enhance(llm, base, f"Открытие сезона {s}. Фавориты по силе клуба: {names}."),
                    feed=feed)


def season_close(world: dict, summary: dict, *, llm=None, feed: Optional[list] = None) -> str:
    s = summary.get("season")
    uc = _team(world, summary.get("upper_champion"))
    lc = _team(world, summary.get("lower_champion"))
    promo = [_team(world, t) for t in summary.get("promoted", [])]
    releg = [_team(world, t) for t in summary.get("relegated", [])]
    base = (f"Сезон {s} завершён! Чемпион верхней лиги — {uc}. Из нижней поднимается {lc}. "
            f"Повышены: {', '.join(promo) or '—'}; вылетают: {', '.join(releg) or '—'}.")
    return add_line(world, "season_close",
                    _enhance(llm, base, f"Итоги сезона {s}: чемпион {uc}, из низа поднимается {lc}, "
                             f"повышены {promo}, вылетают {releg}."), feed=feed)


_MGR_TEMPLATES = [
    "Сезон {s} — наш. Состав готов, верю в {p}. {rival} сильны, но мы дадим бой.",
    "Делаем ставку на {trait}. {p} в отличной форме, остальные подтянутся.",
    "Не лучшее межсезонье по деньгам, но цель прежняя — {goal}. Будем грызть.",
    "Скаутили {rival} — у них дыры. Подготовим план, {p} их разберёт.",
    "Молодёжь голодная, ветераны злые. {goal} — иначе зачем всё это.",
    "Рынок был дорогим. Сохранили костяк, {p} — наш лидер. Идём за результатом.",
    "Уважаю {rival}, но боюсь только себя. Сезон {s} покажет, кто чего стоит.",
    "Перестроились под {trait}. Риск есть, но без риска нет титулов.",
]
_TRAITS = {"aggression": "агрессию", "frugality": "дисциплину и экономику",
           "youth_focus": "молодёжь", "patience": "терпеливый контроль", "loyalty": "костяк"}


def manager_comments(world: dict, *, llm=None, feed: Optional[list] = None) -> list:
    """One short comment from each club's manager this off-season (plans / hopes / opinions).
    Template-driven (varied by the manager's character) + optionally rewritten by the LLM. Stored
    on world['manager_comments'] for the Менеджеры tab."""
    from src.world import manager as mgr
    season = world.get("season", 0)
    out = []
    for t in world["teams"].values():
        if len(t.get("roster", [])) < 3:
            continue
        prof = mgr.natural_profile(t["franchise_id"])
        league = t.get("league")
        rivals = [x for x in world["teams"].values()
                  if x.get("league") == league and x["team_id"] != t["team_id"]]
        rival = (sorted(rivals, key=lambda x: -x.get("elo", 0))[0]["name"] if rivals else "соперники")
        star = next((world["players"][pid]["nickname"] for pid in t.get("roster", [])
                     if pid in world["players"]), "наш лидер")
        trait = _TRAITS[max(_TRAITS, key=lambda k: getattr(prof, k, 0.0))]
        goal = "титул" if league == "upper" else "повышение в верхнюю лигу"
        idx = (sum(ord(c) for c in t["team_id"]) + season) % len(_MGR_TEMPLATES)
        text = _MGR_TEMPLATES[idx].format(s=season, p=star, rival=rival, trait=trait, goal=goal)
        if llm is not None:
            try:
                text = _enhance(llm, text, f"Менеджер клуба {t['name']} перед сезоном {season}. "
                                f"Звезда: {star}. Главный соперник: {rival}. Уклон: {trait}. Цель: {goal}.")
            except Exception as exc:
                events.fallback(feed, f"comment:{t['team_id']}", str(exc))
        out.append({"season": season, "team_id": t["team_id"], "team": t["name"],
                    "manager": t.get("manager") or "Менеджер", "text": text})
    world["manager_comments"] = (world.get("manager_comments", []) + out)[-66:]
    return out


def offseason_note(world: dict, report: dict, *, llm=None, feed: Optional[list] = None) -> str:
    st = report.get("stages", {})
    auc = ((st.get("training", {}) or {}).get("auction") or {}).get("awarded", 0)
    sign = (st.get("transfers", {}) or {}).get("signings", 0)
    rook = (st.get("rookies") or {}).get("n", 0)
    base = (f"Межсезонье отгремело: на аукционе разыграно {auc} тренировок, {sign} трансферов, "
            f"создано {rook} новичков. Клубы готовятся к новому сезону.")
    return add_line(world, "offseason", _enhance(llm, base, base), feed=feed)
