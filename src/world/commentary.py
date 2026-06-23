"""World commentator — Гарри «Голос» Вольский calls the season.

Template lines ALWAYS work (deterministic, no cost, varied by a stable hash so seasons don't repeat
verbatim); when an LLM is supplied the line is rewritten in persona (richer, still grounded in the
same facts). Lines are stored on ``world['commentary']`` so the UI can replay the broadcast. NEW: a
``match_commentary`` generator the Worlds tab calls ON DEMAND for any specific game (template +
optional LLM, ephemeral — not stored).

LOCALIZATION (RU · EN · JA): every public generator takes ``lang`` (default "ru" → byte-identical to
before). Templates are language-keyed; when an LLM is supplied it is instructed to write IN ``lang``
(the facts it's given may stay Russian — the model translates). The viewer passes the player's chosen
language via the ``X-World-Lang`` request header, threaded through the server jobs to here.
"""
from __future__ import annotations

from typing import Optional

from src.world import events

LANGS = ("ru", "en", "ja")


def _norm(lang: Optional[str]) -> str:
    return lang if lang in LANGS else "ru"


# A vivid, self-contained persona. Гарри «Голос» Вольский: old-school, theatrical, loves a good
# upset, name-drops Power and stakes, never invents facts. One persona per output language.
PERSONA = {
    "ru": (
        "Ты — Гарри «Голос» Вольский, ветеран-комментатор 3v3-автобатлера. Говоришь по-русски, ярко и "
        "театрально, с азартом и сухим юмором. Любишь сенсации и камбэки, уважаешь холодный расчёт. "
        "Оперируешь фактами, которые тебе дают (сила-Power клубов, счёт по картам, ставки матча, "
        "повышения/вылеты) — НИКОГДА не выдумываешь имён, цифр или событий. Коротко, образно, по делу."
    ),
    "en": (
        "You are Harry “The Voice” Volsky, a veteran 3v3-autobattler commentator. You speak "
        "English, vividly and theatrically, with passion and dry humor. You love upsets and comebacks, "
        "respect cold calculation. You work only from the facts you are given (club Power, map scores, "
        "match stakes, promotions/relegations) — NEVER invent names, numbers or events. Short, "
        "vivid, to the point."
    ),
    "ja": (
        "あなたはハリー・「ザ・ヴォイス」・ヴォルスキー、3v3オートバトラーのベテラン実況者。日本語で、"
        "華やかかつ劇的に、情熱と辛口のユーモアを交えて語る。番狂わせとカムバックを愛し、冷静な計算を"
        "尊重する。与えられた事実（クラブのPower、マップスコア、試合の重要度、昇格／降格）だけを扱い、"
        "名前・数字・出来事を決して捏造しない。短く、鮮やかに、要点を突く。"
    ),
}

# "write N sentences ... in <lang>" instruction + the default N phrase, per language.
_SAY_INSTR = {
    "ru": "Напиши {n} от лица комментатора — живо, по-русски, строго по фактам выше, ничего не выдумывая.",
    "en": "Write {n} as the commentator — lively, in English, strictly from the facts above, inventing nothing.",
    "ja": "実況者として{n}を書け — 生き生きと、日本語で、上記の事実のみに基づき、何も捏造しないこと。",
}
_SAY_N = {"ru": "2–3 ярких предложения", "en": "2-3 vivid sentences", "ja": "2〜3の生き生きとした文"}
_SAY_N_MGR = {"ru": "1–2 предложения от первого лица", "en": "1-2 first-person sentences", "ja": "一人称で1〜2文"}
# season open/close are the big set-piece moments — let Harry build a STORY from the rich facts.
_SAY_N_BIG = {"ru": "3–4 живых предложения: завяжи сюжет (кто фаворит, кто бросает вызов, что на кону), "
                    "обыграй конкретные клубы и цифры из фактов, без воды и штампов",
              "en": "3-4 lively sentences: build a story (who's favored, who's challenging, what's at stake), "
                    "play off the specific clubs and numbers in the facts, no filler or clichés",
              "ja": "3〜4の生き生きとした文：物語を作れ（本命、挑戦者、何が懸かるか）、事実の具体的な"
                    "クラブ名と数字を生かし、無駄や決まり文句を避けよ"}

_STAGE = {
    "ru": {"league": "регулярки", "ub": "верхней сетки", "lb": "нижней сетки", "lb_r1": "нижней сетки",
           "lb_r2": "нижней сетки", "lb_final": "финала нижней сетки", "grand_final": "ГРАНД-ФИНАЛА"},
    "en": {"league": "the regular season", "ub": "the upper bracket", "lb": "the lower bracket",
           "lb_r1": "the lower bracket", "lb_r2": "the lower bracket", "lb_final": "the lower-bracket final",
           "grand_final": "the GRAND FINAL"},
    "ja": {"league": "レギュラーシーズン", "ub": "アッパーブラケット", "lb": "ロワーブラケット",
           "lb_r1": "ロワーブラケット", "lb_r2": "ロワーブラケット", "lb_final": "ロワーブラケット決勝",
           "grand_final": "グランドファイナル"},
}
_STAGE_DEFAULT = {"ru": "матча", "en": "the match", "ja": "試合"}


def _persona(lang: str = "ru") -> str:
    return PERSONA[_norm(lang)]


def _llm_say(llm, base: str, brief: str, *, lang: str = "ru",
             n: Optional[str] = None, max_tokens: int = 240) -> str:
    """Rewrite the factual ``brief`` in persona IN ``lang``; fall back to the template ``base`` on error."""
    if llm is None:
        return base
    lang = _norm(lang)
    try:
        instr = _SAY_INSTR[lang].format(n=n or _SAY_N[lang])
        out = llm.say(_persona(lang), f"{brief}\n{instr}", max_tokens=max_tokens)
        return (out or "").strip() or base
    except Exception:
        return base


# keep the old name working for any external caller
def _enhance(llm, base: str, brief: str) -> str:
    return _llm_say(llm, base, brief)


def _team(world: dict, tid) -> str:
    return world["teams"][tid]["name"] if tid in world.get("teams", {}) else (tid or "—")


def _power(world: dict, tid) -> int:
    from src.world import views
    t = world.get("teams", {}).get(tid)
    return int(views.team_power(world, t)) if t else 1000


def _hash(*parts) -> int:
    return sum(ord(c) for c in "|".join(str(p) for p in parts))


def add_line(world: dict, kind: str, text: str, *, feed: Optional[list] = None) -> str:
    world.setdefault("commentary", []).append(
        {"season": world.get("season", 0), "kind": kind, "text": text})
    if len(world["commentary"]) > 120:
        world["commentary"] = world["commentary"][-120:]
    if feed is not None:
        events.emit(feed, "comment", text)
    return text


# --------------------------------------------------------------------------- #
# Season open / close — varied templates (pick by a stable hash so it doesn't repeat verbatim)
# --------------------------------------------------------------------------- #
_OPEN_TEMPLATES = {
    "ru": [
        "Свет гаснет, арена ревёт — сезон {s} начинается! По силе впереди {a} ({pa}), но {b} и {c} дышат в спину. Поехали!",
        "Новый сезон {s}, новые надежды. Фаворит по Power — {a} ({pa}). {b} и {c} готовы спорить. Погнали!",
        "Сезон {s}! На бумаге сильнее всех {a} ({pa}), но бумага не выигрывает серий. {b}, {c} — ваш выход.",
        "Звенит гонг сезона {s}. {a} ({pa}) — на острие, {b} и {c} точат ножи. Будет жарко!",
    ],
    "en": [
        "The lights go down, the arena roars — season {s} begins! On Power {a} ({pa}) lead, but {b} and {c} are right behind. Here we go!",
        "New season {s}, new hopes. Power favorite — {a} ({pa}). {b} and {c} are ready to argue. Let's roll!",
        "Season {s}! On paper {a} ({pa}) are the strongest, but paper doesn't win series. {b}, {c} — you're up.",
        "The season-{s} gong sounds. {a} ({pa}) on the edge, {b} and {c} sharpening their knives. It'll be hot!",
    ],
    "ja": [
        "照明が落ち、アリーナが轟く — シーズン{s}開幕！Powerでは{a}（{pa}）が先行、だが{b}と{c}が背後に迫る。さあ行こう！",
        "新シーズン{s}、新たな希望。Power本命は{a}（{pa}）。{b}と{c}も黙っていない。始めよう！",
        "シーズン{s}！紙の上では{a}（{pa}）が最強、だが紙はシリーズを勝たない。{b}、{c} — 出番だ。",
        "シーズン{s}の銅鑼が鳴る。{a}（{pa}）が先頭、{b}と{c}が牙を研ぐ。熱くなるぞ！",
    ],
}
_CLOSE_TEMPLATES = {
    "ru": [
        "Занавес сезона {s}! Корону верхней лиги забирает {uc}. Из низов прорывается {lc}. Повышены: {promo}; вылетают: {releg}. Что за сезон!",
        "Сезон {s} в учебники! Чемпион верха — {uc}, герой низа — {lc}. ↑ {promo}, ↓ {releg}. До новых битв!",
        "Финальный свисток сезона {s}. {uc} — король верхней лиги, {lc} поднимается из нижней. Повышены {promo}, вылетают {releg}.",
    ],
    "en": [
        "Curtain on season {s}! The upper-league crown goes to {uc}. Breaking out from below — {lc}. Promoted: {promo}; relegated: {releg}. What a season!",
        "Season {s} for the books! Upper champion — {uc}, hero of the lower — {lc}. ↑ {promo}, ↓ {releg}. Until the next battles!",
        "Final whistle of season {s}. {uc} — king of the upper league, {lc} rises from the lower. Promoted {promo}, relegated {releg}.",
    ],
    "ja": [
        "シーズン{s}の幕が下りる！上位リーグの王冠は{uc}へ。下位から駆け上がるのは{lc}。昇格：{promo}、降格：{releg}。なんてシーズンだ！",
        "シーズン{s}は歴史に刻まれる！上位の王者は{uc}、下位の英雄は{lc}。↑ {promo}、↓ {releg}。次の戦いまで！",
        "シーズン{s}の最終ホイッスル。{uc}は上位リーグの王、{lc}が下位から昇る。昇格{promo}、降格{releg}。",
    ],
}


def _open_brief(world: dict) -> str:
    """Rich, grounded facts for the season-open call (Russian facts; the LLM writes in the UI lang).
    Pulls in last season's champion / promotions / relegations + star players so Harry has a STORY."""
    from src.world import views
    s = world.get("season", 0)
    top = sorted(world["teams"].values(), key=lambda t: -views.team_power(world, t))[:3]
    favs = ", ".join(f"{t['name']} (Power {int(views.team_power(world, t))})" for t in top)
    parts = [f"Открытие сезона {s}. Фавориты по силе: {favs}."]
    hist = world.get("history") or []
    if hist:
        last = hist[-1]
        champ = _team(world, last.get("upper_champion"))
        if champ and champ != "—":
            parts.append(f"Действующий чемпион — {champ}, выходит защищать корону.")
        promo = [_team(world, t) for t in (last.get("promoted") or [])]
        if promo:
            parts.append(f"Дебютанты верхней лиги (только что поднялись): {', '.join(promo)}.")
        releg = [_team(world, t) for t in (last.get("relegated") or [])]
        if releg:
            parts.append(f"В прошлом сезоне вылетели вниз: {', '.join(releg)} — жаждут реванша.")
    else:
        parts.append("Это самый первый сезон мира — у всех чистый лист.")
    ranked = views.ranked_players(world)[:3]
    if ranked:
        parts.append("Главные звёзды по рейтингу SoloQ: "
                     + ", ".join(f"{r['nickname']} ({r['rating']})" for r in ranked) + ".")
    return " ".join(parts)


def season_open(world: dict, *, llm=None, feed: Optional[list] = None, lang: str = "ru") -> str:
    from src.world import views
    lang = _norm(lang)
    s = world.get("season", 0)
    top = sorted(world["teams"].values(), key=lambda t: -views.team_power(world, t))[:3]
    n = [t["name"] for t in top] + ["—", "—", "—"]
    p = [int(views.team_power(world, t)) for t in top] + [0, 0, 0]
    tpl = _OPEN_TEMPLATES[lang][_hash("open", s) % len(_OPEN_TEMPLATES[lang])]
    base = tpl.format(s=s, a=n[0], b=n[1], c=n[2], pa=p[0])
    text = _llm_say(llm, base, _open_brief(world), lang=lang, n=_SAY_N_BIG[lang], max_tokens=420)
    return add_line(world, "season_open", text, feed=feed)


def _close_brief(world: dict, summary: dict) -> str:
    """Rich facts for the season-close call: champions, promo/releg, whether the title was an upset,
    and the season's standout SoloQ riser — so the recap reads like a real wrap-up, not a list."""
    from src.world import views
    s = summary.get("season")
    uc_id = summary.get("upper_champion")
    uc, lc = _team(world, uc_id), _team(world, summary.get("lower_champion"))
    promo = ", ".join(_team(world, t) for t in summary.get("promoted", [])) or "никто"
    releg = ", ".join(_team(world, t) for t in summary.get("relegated", [])) or "никто"
    parts = [f"Итоги сезона {s}. Чемпион верхней лиги — {uc}. Победитель нижней (идёт наверх) — {lc}.",
             f"Повышены: {promo}. Вылетают: {releg}."]
    # was the champion the pre-eminent club, or a surprise? (rank by current Power)
    if uc_id:
        order = sorted(world["teams"].values(), key=lambda t: -views.team_power(world, t))
        rank = next((i + 1 for i, t in enumerate(order) if t["team_id"] == uc_id), None)
        if rank and rank >= 4:
            parts.append(f"{uc} взяли титул будучи лишь #{rank} по силе — настоящая сенсация.")
        elif rank == 1:
            parts.append(f"{uc} были сильнейшими по Power и подтвердили статус.")
    ranked = views.ranked_players(world)
    if ranked:
        risers = [r for r in ranked if (r.get("delta") or 0) > 0]
        risers.sort(key=lambda r: -(r.get("delta") or 0))
        if risers:
            parts.append(f"Прорыв сезона — {risers[0]['nickname']} (+{risers[0]['delta']} рейтинга).")
    return " ".join(parts)


def season_close(world: dict, summary: dict, *, llm=None, feed: Optional[list] = None, lang: str = "ru") -> str:
    lang = _norm(lang)
    s = summary.get("season")
    uc = _team(world, summary.get("upper_champion"))
    lc = _team(world, summary.get("lower_champion"))
    promo = ", ".join(_team(world, t) for t in summary.get("promoted", [])) or "—"
    releg = ", ".join(_team(world, t) for t in summary.get("relegated", [])) or "—"
    tpl = _CLOSE_TEMPLATES[lang][_hash("close", s) % len(_CLOSE_TEMPLATES[lang])]
    base = tpl.format(s=s, uc=uc, lc=lc, promo=promo, releg=releg)
    text = _llm_say(llm, base, _close_brief(world, summary), lang=lang, n=_SAY_N_BIG[lang], max_tokens=420)
    return add_line(world, "season_close", text, feed=feed)


# --------------------------------------------------------------------------- #
# Match commentary — ON DEMAND for a specific game (Worlds tab) — ephemeral
# --------------------------------------------------------------------------- #
_MARGIN = {
    "ru": {"sweep": "всухую", "tb": "на тай-брейке", "conf": "уверенно"},
    "en": {"sweep": "in a sweep", "tb": "on a tiebreaker", "conf": "convincingly"},
    "ja": {"sweep": "ストレートで", "tb": "タイブレークで", "conf": "確実に"},
}
_PLAYED_BASE = {
    "ru": "{hn} ({hp}) — {an} ({ap}). В рамках {stage} {win} забирают серию {hi}:{lo} {margin}.{flavor}",
    "en": "{hn} ({hp}) — {an} ({ap}). In {stage}, {win} take the series {hi}:{lo} {margin}.{flavor}",
    "ja": "{hn}（{hp}）— {an}（{ap}）。{stage}で{win}がシリーズを{hi}:{lo}で奪取、{margin}。{flavor}",
}
_FLAVOR_UPSET = {
    "ru": " Сенсация {stage}! {win} ({wp}) свергли фаворита {los} ({lp}).",
    "en": " Upset in {stage}! {win} ({wp}) toppled favorite {los} ({lp}).",
    "ja": " {stage}で番狂わせ！{win}（{wp}）が本命{los}（{lp}）を倒した。",
}
_FLAVOR_OK = {
    "ru": " По силе так и должно было быть — {win} были крепче.",
    "en": " By Power it was due — {win} were the stronger side.",
    "ja": " Power通りの結果 — {win}の方が強かった。",
}
_ODDS = {
    "ru": {"even": "игра равная", "fav": "фаворит — {fav}", "clear": "{fav} — явный фаворит"},
    "en": {"even": "an even game", "fav": "the favorite is {fav}", "clear": "{fav} are clear favorites"},
    "ja": {"even": "互角", "fav": "本命は{fav}", "clear": "{fav}が明確な本命"},
}
_UPCOMING_BASE = {
    "ru": "На очереди {hn} ({hp}) против {an} ({ap}). По силе {odds}, но в этой игре всё решат нервы и порядок. {dog} мечтают о сенсации.",
    "en": "Up next: {hn} ({hp}) vs {an} ({ap}). On Power {odds}, but this game comes down to nerves and order. {dog} dream of an upset.",
    "ja": "次は{hn}（{hp}）対{an}（{ap}）。Powerでは{odds}だが、この試合は気持ちと統率で決まる。{dog}は番狂わせを夢見る。",
}


def _find_match(world: dict, home_id: str, away_id: str, season: Optional[int]) -> Optional[dict]:
    """Most recent recorded series between the two clubs — from the canonical calendar, OR (if not
    there yet) the IN-PROGRESS live season. The live season's games sit in ``world['live_replay']``
    and are only written to the calendar at finalize, so without this a just-watched live match is
    found nowhere → commented as 'upcoming'. We only consider REVEALED live games (no spoilers)."""
    found = None
    for e in world.get("calendar", []):
        if season is not None and e.get("season") != season:
            continue
        if {e.get("home"), e.get("away")} == {home_id, away_id}:
            found = e
    if found is not None:
        return found
    lr = world.get("live_replay") or {}
    if lr and (season is None or season == lr.get("season")):
        revealed = set(lr.get("revealed_gids", []))
        for d in lr.get("days", []):
            for s in d.get("series", []):
                if s.get("gid") in revealed and {s.get("home"), s.get("away")} == {home_id, away_id}:
                    found = {"home": s["home"], "away": s["away"], "maps_home": s.get("maps_home"),
                             "maps_away": s.get("maps_away"), "winner": s.get("winner"),
                             "stage": s.get("stage"), "played": True, "season": lr.get("season")}
    return found


def match_commentary(world: dict, home_id: str, away_id: str,
                     season: Optional[int] = None, *, llm=None, lang: str = "ru") -> dict:
    """Rich commentary for ONE game (played or upcoming), generated on demand. Template always; LLM
    enriches if supplied. Ephemeral — NOT stored on world['commentary']."""
    lang = _norm(lang)
    hn, an = _team(world, home_id), _team(world, away_id)
    hp, ap = _power(world, home_id), _power(world, away_id)
    ev = _find_match(world, home_id, away_id, season)
    played = bool(ev and ev.get("played")
                  and ev.get("maps_home") is not None and ev.get("maps_away") is not None)
    if played:
        # orient the score to home/away as recorded
        mh, ma = ev.get("maps_home", 0), ev.get("maps_away", 0)
        if ev.get("home") != home_id:                        # the stored event is flipped
            mh, ma = ma, mh
        win_id, los_id = (home_id, away_id) if mh > ma else (away_id, home_id)
        win, los = _team(world, win_id), _team(world, los_id)
        wp, lp = (hp, ap) if win_id == home_id else (ap, hp)
        stage = _STAGE[lang].get(ev.get("stage"), _STAGE_DEFAULT[lang])
        mk = "sweep" if min(mh, ma) == 0 else "tb" if abs(mh - ma) == 1 else "conf"
        margin = _MARGIN[lang][mk]
        upset = wp < lp - 30
        flavor = (_FLAVOR_UPSET[lang].format(stage=stage, win=win, wp=wp, los=los, lp=lp) if upset
                  else _FLAVOR_OK[lang].format(win=win))
        base = _PLAYED_BASE[lang].format(hn=hn, hp=hp, an=an, ap=ap, stage=stage, win=win,
                                         hi=max(mh, ma), lo=min(mh, ma), margin=margin, flavor=flavor)
        brief = (f"Матч {_STAGE['ru'].get(ev.get('stage'), 'матча')}: {hn} (Power {hp}) против {an} "
                 f"(Power {ap}). Победили {win} со счётом {max(mh, ma)}:{min(mh, ma)}. "
                 f"{'Это апсет — победитель был слабее по силе.' if upset else 'Фаворит подтвердил силу.'}")
    else:
        fav, dog = (hn, an) if hp >= ap else (an, hn)
        gap = abs(hp - ap)
        if gap < 25:
            odds = _ODDS[lang]["even"]
        elif gap < 80:
            odds = _ODDS[lang]["fav"].format(fav=fav)
        else:
            odds = _ODDS[lang]["clear"].format(fav=fav)
        base = _UPCOMING_BASE[lang].format(hn=hn, hp=hp, an=an, ap=ap, odds=odds, dog=dog)
        odds_ru = ("игра равная" if gap < 25 else f"фаворит — {fav}" if gap < 80 else f"{fav} — явный фаворит")
        brief = f"Предстоящий матч {hn} (Power {hp}) — {an} (Power {ap}). По Power {odds_ru}."
    text = _llm_say(llm, base, brief, lang=lang)
    return {"home": hn, "away": an, "home_power": hp, "away_power": ap,
            "played": played, "text": text,
            "score": (f"{max(ev.get('maps_home', 0), ev.get('maps_away', 0))}:"
                      f"{min(ev.get('maps_home', 0), ev.get('maps_away', 0))}") if played else None}


# --------------------------------------------------------------------------- #
# Manager comments + off-season note
# --------------------------------------------------------------------------- #
_MGR_TEMPLATES = {
    "ru": [
        "Сезон {s} — наш. Состав готов, верю в {p}. {rival} сильны, но мы дадим бой.",
        "Делаем ставку на {trait}. {p} в отличной форме, остальные подтянутся.",
        "Не лучшее межсезонье по деньгам, но цель прежняя — {goal}. Будем грызть.",
        "Скаутили {rival} — у них дыры. Подготовим план, {p} их разберёт.",
        "Молодёжь голодная, ветераны злые. {goal} — иначе зачем всё это.",
        "Рынок был дорогим. Сохранили костяк, {p} — наш лидер. Идём за результатом.",
        "Уважаю {rival}, но боюсь только себя. Сезон {s} покажет, кто чего стоит.",
        "Перестроились под {trait}. Риск есть, но без риска нет титулов.",
        "{p} — сердце команды. Дадим ему свободу, а {rival} пусть догоняют.",
        "Хватит вторых мест. {goal} или ничего — так и передайте {rival}.",
        "Тихо готовимся. Меньше слов — больше карт. {p} покажет на арене.",
        "В межсезонье поработали над дисциплиной. {trait} — вот наш козырь в сезоне {s}.",
    ],
    "en": [
        "Season {s} is ours. The roster's ready, I believe in {p}. {rival} are strong, but we'll fight.",
        "We're betting on {trait}. {p} is in great form, the rest will catch up.",
        "Not the best off-season money-wise, but the goal stands — {goal}. We'll claw for it.",
        "We scouted {rival} — they have holes. We'll prep a plan, {p} will take them apart.",
        "The youngsters are hungry, the veterans are angry. {goal} — or why are we even here.",
        "The market was pricey. We kept the core, {p} is our leader. We're going for the result.",
        "I respect {rival}, but I only fear myself. Season {s} will show who's worth what.",
        "We rebuilt around {trait}. There's risk, but no titles without risk.",
        "{p} is the heart of the team. We'll give him freedom, and {rival} can chase.",
        "Enough second places. {goal} or nothing — tell {rival} that.",
        "We prepare quietly. Fewer words — more maps. {p} will show it on the arena.",
        "In the off-season we worked on discipline. {trait} — that's our trump card in season {s}.",
    ],
    "ja": [
        "シーズン{s}は我々のものだ。ロスターは整った、{p}を信じる。{rival}は強いが、我々も戦う。",
        "我々は{trait}に賭ける。{p}は絶好調、他も追いつく。",
        "資金面では最高のオフシーズンではないが、目標は不変 — {goal}。食らいついていく。",
        "{rival}を偵察した — 穴がある。プランを練り、{p}が切り崩す。",
        "若手はハングリー、ベテランは闘志満々。{goal} — でなければ何のためだ。",
        "市場は高騰した。中核を維持し、{p}が我々のリーダー。結果を取りに行く。",
        "{rival}に敬意を払うが、恐れるのは自分だけ。シーズン{s}が真価を示す。",
        "{trait}を軸に再構築した。リスクはあるが、リスクなくしてタイトルなし。",
        "{p}はチームの心臓。彼に自由を与え、{rival}には追わせる。",
        "二位はもうたくさんだ。{goal}か無か — {rival}にそう伝えろ。",
        "静かに準備する。言葉より多くのマップを。{p}がアリーナで示す。",
        "オフシーズンは規律に取り組んだ。{trait} — それがシーズン{s}の切り札だ。",
    ],
}
_TRAITS = {
    "ru": {"aggression": "агрессию", "frugality": "дисциплину и экономику", "youth_focus": "молодёжь",
           "patience": "терпеливый контроль", "loyalty": "костяк"},
    "en": {"aggression": "aggression", "frugality": "discipline and economy", "youth_focus": "youth",
           "patience": "patient control", "loyalty": "the core"},
    "ja": {"aggression": "アグレッション", "frugality": "規律と経済", "youth_focus": "若手",
           "patience": "忍耐の制圧", "loyalty": "中核"},
}
_GOAL = {
    "ru": {"upper": "титул", "lower": "повышение в верхнюю лигу"},
    "en": {"upper": "the title", "lower": "promotion to the upper league"},
    "ja": {"upper": "タイトル", "lower": "上位リーグへの昇格"},
}
# Manager comments must sound like the MANAGER speaking about their OWN club — first person, in
# character — NOT the commentator narrating them. So they get their own persona + instruction instead
# of reusing the Вольский commentator voice (which was turning every comment into third-person narration).
_MGR_PERSONA = {
    "ru": ("Ты — {mgr}, менеджер киберспортивного клуба {team} в 3v3-автобатлере. Говоришь по-русски "
           "ОТ ПЕРВОГО ЛИЦА (я / мы / наш клуб) про СВОЮ команду: состав, новичков, план на сезон, цели "
           "и соперников. Характер — {trait}. Только факты, что тебе дали; ничего не выдумываешь. "
           "НИКОГДА не говоришь о себе или клубе в третьем лице и не комментируешь со стороны."),
    "en": ("You are {mgr}, manager of the esports club {team} in a 3v3 autobattler. You speak English "
           "in the FIRST PERSON (I / we / our club) about YOUR team: roster, newcomers, the season plan, "
           "goals and rivals. Your character — {trait}. Only the facts you are given; invent nothing. "
           "NEVER speak about yourself or the club in the third person, and never narrate from outside."),
    "ja": ("あなたは3v3オートバトラーのeスポーツクラブ{team}のマネージャー{mgr}。日本語で一人称（私／我々／"
           "我がクラブ）で自チームについて語る：ロスター、新加入、シーズンの計画、目標、ライバル。性格は{trait}。"
           "与えられた事実のみ、捏造しない。自分やクラブを三人称で語らず、外から実況しない。"),
}
_MGR_SAY_INSTR = {
    "ru": "Скажи {n} от первого лица как менеджер {team} — про свою команду, планы и цели, строго по фактам выше.",
    "en": "Say {n} in the first person as {team}'s manager — about your team, plans and goals, strictly from the facts above.",
    "ja": "{team}のマネージャーとして一人称で{n}述べよ — 自チーム・計画・目標について、上記の事実のみに基づいて。",
}
_FALLBACK_RIVAL = {"ru": "соперники", "en": "rivals", "ja": "ライバル"}
_FALLBACK_STAR = {"ru": "наш лидер", "en": "our leader", "ja": "我々のリーダー"}
_FALLBACK_MGR = {"ru": "Менеджер", "en": "Manager", "ja": "マネージャー"}


def manager_comments(world: dict, *, llm=None, feed: Optional[list] = None, lang: str = "ru") -> list:
    """One short comment from each club's manager this off-season (plans / hopes / opinions).
    Template-driven (varied by the manager's character + a stable hash) + optionally LLM-rewritten.
    Stored on world['manager_comments'] for the Менеджеры tab."""
    from src.world import manager as mgr
    lang = _norm(lang)
    season = world.get("season", 0)
    out = []
    for t in world["teams"].values():
        if len(t.get("roster", [])) < 3:
            continue
        prof = mgr.natural_profile(t["franchise_id"])
        league = t.get("league")
        rivals = [x for x in world["teams"].values()
                  if x.get("league") == league and x["team_id"] != t["team_id"]]
        rival = (sorted(rivals, key=lambda x: -x.get("elo", 0))[0]["name"]
                 if rivals else _FALLBACK_RIVAL[lang])
        star = next((world["players"][pid]["nickname"] for pid in t.get("roster", [])
                     if pid in world["players"]), _FALLBACK_STAR[lang])
        trait_key = max(_TRAITS["ru"], key=lambda k: getattr(prof, k, 0.0))
        trait = _TRAITS[lang][trait_key]
        goal = _GOAL[lang]["upper" if league == "upper" else "lower"]
        idx = _hash("mgr", t["team_id"], season) % len(_MGR_TEMPLATES[lang])
        text = _MGR_TEMPLATES[lang][idx].format(s=season, p=star, rival=rival, trait=trait, goal=goal)
        if llm is not None:
            try:
                # facts the manager speaks FROM (his own club), then a first-person persona + instruction
                brief = (f"Сезон {season}. Звезда твоего клуба: {star}. Главный соперник: {rival}. "
                         f"Твой управленческий уклон: {_TRAITS['ru'][trait_key]}. "
                         f"Цель сезона: {_GOAL['ru']['upper' if league == 'upper' else 'lower']}.")
                persona = _MGR_PERSONA[lang].format(mgr=t.get("manager") or _FALLBACK_MGR[lang],
                                                    team=t["name"], trait=trait)
                instr = _MGR_SAY_INSTR[lang].format(n=_SAY_N_MGR[lang], team=t["name"])
                out = llm.say(persona, f"{brief}\n{instr}", max_tokens=240)
                text = (out or "").strip() or text
            except Exception as exc:
                events.fallback(feed, f"comment:{t['team_id']}", str(exc))
        out.append({"season": season, "team_id": t["team_id"], "team": t["name"],
                    "manager": t.get("manager") or _FALLBACK_MGR[lang], "text": text})
    world["manager_comments"] = (world.get("manager_comments", []) + out)[-66:]
    return out


_OFFSEASON_BASE = {
    "ru": ("Межсезонье отгремело: на аукционе разыграно {auc} тренировок, {sign} подписаний, "
           "{buys} сделок между клубами, создано {rook} новичков. Клубы перетряхнули составы — "
           "и точат зубы на новый сезон."),
    "en": ("The off-season is done: {auc} trainings auctioned, {sign} signings, {buys} club-to-club "
           "deals, {rook} rookies created. Clubs reshuffled their rosters — and are sharpening their "
           "teeth for the new season."),
    "ja": ("オフシーズンが終わった：オークションでトレーニング{auc}件、契約{sign}件、クラブ間取引{buys}件、"
           "ルーキー{rook}人を創出。各クラブはロスターを刷新 — 新シーズンへ牙を研ぐ。"),
}


def offseason_note(world: dict, report: dict, *, llm=None, feed: Optional[list] = None, lang: str = "ru") -> str:
    lang = _norm(lang)
    st = report.get("stages", {})
    auc = ((st.get("training", {}) or {}).get("auction") or {}).get("awarded", 0)
    sign = (st.get("transfers", {}) or {}).get("signings", 0)
    rook = (st.get("rookies") or {}).get("n", 0)
    tx = (world.get("last_offseason") or {}).get("transactions") or {}
    buys = (tx.get("counts") or {}).get("buys", 0)
    base = _OFFSEASON_BASE[lang].format(auc=auc, sign=sign, buys=buys, rook=rook)
    brief = _OFFSEASON_BASE["ru"].format(auc=auc, sign=sign, buys=buys, rook=rook)
    return add_line(world, "offseason", _llm_say(llm, base, brief, lang=lang), feed=feed)
