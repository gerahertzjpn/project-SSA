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


def offseason_note(world: dict, report: dict, *, llm=None, feed: Optional[list] = None) -> str:
    st = report.get("stages", {})
    auc = ((st.get("training", {}) or {}).get("auction") or {}).get("awarded", 0)
    sign = (st.get("transfers", {}) or {}).get("signings", 0)
    rook = (st.get("rookies") or {}).get("n", 0)
    base = (f"Межсезонье отгремело: на аукционе разыграно {auc} тренировок, {sign} трансферов, "
            f"создано {rook} новичков. Клубы готовятся к новому сезону.")
    return add_line(world, "offseason", _enhance(llm, base, base), feed=feed)
