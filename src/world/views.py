"""Read-only VIEW builders for the World UI — pure functions over the world dict.

Everything the tabbed simulator UI needs, derived from already-stored state:
  - player_view / ranked_players — every player ranked, with rich detail + per-season
    rating delta (up/down) and the full rating history for a sparkline;
  - league_tables — standings split by league AND group (upper / lower-A / lower-Z),
    each with explicit places (fixes the "groups mixed" UI);
  - team_view — a per-team page (record, Elo curve, trophies, roster);
  - schedule_view — the season's matches as a day-by-day fixture list + the playoff
    bracket grouped by stage.

These are deterministic and torch-free, so the server can return them directly and
tests can exercise them without HTTP.
"""
from __future__ import annotations

from typing import Optional

from src.world.world import get_player

# Playoff stages in bracket order (for rendering a bracket left-to-right). "lb" is the upper
# league's single loser-bracket round (lower league uses lb_r1/lb_r2 instead).
PLAYOFF_STAGE_ORDER = ("lb_r1", "lb_r2", "ub", "lb", "lb_final", "grand_final")


_MAX_POSITION_BONUS = 80.0


def _position_bonus(world: dict, team_id: str) -> float:
    """A bonus from where the club FINISHED last season (1st = full, last = 0). Rewards league
    position on top of Elo, normalized by league size so a champion gets the same regardless of
    league size."""
    hist = world.get("history") or []
    if not hist:
        return 0.0
    last = hist[-1]
    for key in ("upper_ranking", "lower_ranking"):
        rk = last.get(key) or []
        if team_id in rk and len(rk) > 1:
            place = rk.index(team_id)
            return (1.0 - place / (len(rk) - 1)) * _MAX_POSITION_BONUS
    return 0.0


def team_power(world: dict, team: dict) -> int:
    """Team Power = the club's RESULTS, never its players' ratings (a club fields many players and
    swaps them mid-series — strength belongs to the CLUB). Two results-based parts:
      - Elo: opponent-weighted match results — beating a STRONGER club gives more (Elo's
        expected-score), so the quality of wins already counts;
      - a league-FINISH bonus from last season's standing (1st > last).
    Elo is global across leagues (one continuous per-club number, seeded at bootstrap from who
    earned promotion, carried through promotion/relegation) so upper clubs generally outrank lower
    ones — not absolutely; a strong lower club can overtake by results.

    During a day-by-day reveal canonical Elo is held at pre-season (no spoilers), so Power tracks
    the PROVISIONAL Elo up to the revealed day instead — so the column moves as the season unfolds."""
    from src.world import live_season
    rp = live_season.revealed_power(world)
    if rp is not None:
        return rp.get(team["team_id"], round(float(team.get("elo", 1000.0))))
    return round(float(team.get("elo", 1000.0)) + _position_bonus(world, team["team_id"]))
STAGE_LABEL = {
    "ub": "Верхняя сетка", "lb": "Нижняя сетка", "lb_r1": "Нижняя сетка — 1",
    "lb_r2": "Нижняя сетка — 2", "lb_final": "Финал нижней сетки",
    "grand_final": "Гранд-финал", "league": "Регулярка",
}


# --------------------------------------------------------------------------- #
# Players
# --------------------------------------------------------------------------- #
def _season_delta(history: list) -> float:
    """Last rating change (newest season vs the one before). soloq_history gets one
    entry appended per off-season SoloQ run."""
    if not history or len(history) < 2:
        return 0.0
    return round(float(history[-1]) - float(history[-2]), 1)


def _team_label(world: dict, p: dict):
    if p.get("team_id") and p["team_id"] in world["teams"]:
        t = world["teams"][p["team_id"]]
        return t["name"], t["abbrev"]
    if p.get("role") == "streamer":
        return "Стример", "STR"
    return "Свободный агент", "FA"


def player_view(world: dict, pid: str) -> dict:
    """One player's full card for the ratings list + per-player detail."""
    p = get_player(world, pid)
    hist = [round(float(r)) for r in (p.get("soloq_history") or [])]
    team_name, team_abbrev = _team_label(world, p)
    pos = p.get("position_stats") or {}
    pos_view = {k: {"games": v.get("games", 0), "wins": v.get("wins", 0),
                    "winrate": round(v["wins"] / v["games"], 3) if v.get("games") else None}
                for k, v in pos.items()}

    def _nick(x):
        return world["players"][x]["nickname"] if x in world["players"] else x
    soloq_games = [{"season": m.get("season"), "round": m.get("round"), "won": m.get("won"),
                    "delta": m.get("delta"),
                    "teammates": [_nick(x) for x in (m.get("teammates") or [])],
                    "opponents": [_nick(x) for x in (m.get("opponents") or [])],
                    "teammate_ids": list(m.get("teammates") or []),
                    "opponent_ids": list(m.get("opponents") or [])}
                   for m in reversed((p.get("soloq_matches") or [])[-20:])]
    return {
        "player_id": pid,
        "nickname": p.get("nickname", pid),
        "faction": p.get("faction"),
        "archetype": p.get("archetype"),
        "role": p.get("role"),
        "status": p.get("status"),
        "rating": round(float(p.get("soloq_rating", 0))),
        "delta": _season_delta(p.get("soloq_history") or []),
        "rating_history": hist,
        "attributes": {k: int(v) for k, v in (p.get("attributes") or {}).items()},
        "earnings": int(p.get("earnings", 0)),
        "salary": round(float(p.get("salary", 0))),
        "net_id": p.get("net_id"),
        "team_id": p.get("team_id"),
        "team": team_name,
        "team_abbrev": team_abbrev,
        "games": p.get("games", 0),
        "official_games": p.get("official_games", 0),
        "seasons_played": p.get("seasons_played", 0),
        "generation": (p.get("lineage") or {}).get("generation", 0),
        "parent_id": (p.get("lineage") or {}).get("parent_id"),
        "position_stats": pos_view,
        "soloq_games": soloq_games,
        "trophies": p.get("trophies", []),
    }


def player_page(world: dict, pid: str) -> dict:
    """Everything for a player's personal page: the full card + his club history (which club each
    season + result), salary history, lineage, trophies and recent SoloQ games."""
    base = player_view(world, pid)
    p = get_player(world, pid)

    def _team(tid):
        return world["teams"][tid]["name"] if tid in world.get("teams", {}) else tid
    base["team_history"] = [{"season": h.get("season"), "team_id": h.get("team_id"),
                             "team": _team(h.get("team_id")), "role": h.get("role"),
                             "result": h.get("result")}
                            for h in (p.get("history") or [])]
    base["salary"] = round(float(p.get("salary", 0)))
    base["salary_history"] = [round(float(s)) for s in (p.get("salary_history") or [])]
    lin = p.get("lineage") or {}
    parent = lin.get("parent_id")
    base["lineage"] = {"origin": lin.get("origin"), "generation": lin.get("generation", 0),
                       "parent_id": parent,
                       "parent": (world["players"][parent]["nickname"]
                                  if parent in world.get("players", {}) else parent)}
    base["soloq_games"] = (base.get("soloq_games") or [])     # already last-20 from player_view
    base["seasons_played"] = p.get("seasons_played", 0)
    base["official_games"] = p.get("official_games", 0)
    base["soloq_history"] = base.get("rating_history") or []   # SoloQ rating progression (per season)
    base["soloq_total"] = len(p.get("soloq_matches") or [])
    return base


def soloq_match(world: dict, player_id: str, season: int, rnd: int) -> dict:
    """Reconstruct a 3v3 SoloQ game from the round: both ad-hoc teams of 3, each player's rating
    delta in that game, and which side won. Players + their clubs are linkable from the detail."""
    players = world["players"]
    if player_id not in players:
        raise KeyError(f"player {player_id!r} not in world")

    def entry(pid):
        for m in (players.get(pid, {}).get("soloq_matches") or []):
            if m.get("season") == season and m.get("round") == rnd:
                return m
        return None

    pm = entry(player_id)
    if pm is None:
        raise KeyError(f"no SoloQ game for {player_id} at season {season} round {rnd}")
    side_a = [player_id] + list(pm.get("teammates") or [])
    side_b = list(pm.get("opponents") or [])
    # ratings + deltas AS OF the match (all 6 share a 100-step band), snapshotted on the clicked
    # player's entry — so every player resolves even though their stored round ids differ.
    at_time = pm.get("ratings") or {}
    at_deltas = pm.get("deltas") or {}
    # the matchmaking band (floor of the TRUE rating) is the same for all 6 — use it everywhere so a
    # rating that rounded up across a 100-boundary never makes the detail look "cross-band".
    band = pm.get("band")
    if band is None:
        band = (int(at_time.get(player_id, 1000)) // 100) * 100

    mstats = pm.get("stats") or {}                # per-player combat stats (net matches only), all 6

    def pinfo(pid):
        p = players.get(pid, {})
        e = entry(pid)
        tname, tabbr = _team_label(world, p)
        rating = at_time.get(pid, round(float(p.get("soloq_rating", 0))))
        delta = at_deltas.get(pid, e.get("delta") if e else None)
        st = mstats.get(pid) or {}
        return {"player_id": pid, "nickname": p.get("nickname", pid), "faction": p.get("faction"),
                "archetype": p.get("archetype"), "role": p.get("role"),
                "rating": rating, "band": band, "delta": delta,
                "kills": st.get("kills"), "assists": st.get("assists"), "core_losses": st.get("core_losses"),
                "team_id": p.get("team_id"), "team": tname, "team_abbrev": tabbr}

    return {"season": season, "round": rnd, "won_a": bool(pm.get("won")), "band": band,
            "team_a": [pinfo(x) for x in side_a], "team_b": [pinfo(x) for x in side_b]}


def ranked_players(world: dict, *, role: Optional[str] = None,
                   faction: Optional[str] = None, include_reserves: bool = False) -> list[dict]:
    """Every ENTERED player ranked by SoloQ rating (1 = best), with rank + per-season delta.
    Reserves are held out of the game (never played → flat rating 1000), so they are EXCLUDED
    from the ladder by default — they only pollute the bottom with non-players. Set
    ``include_reserves=True`` for a dedicated reserve view. Optional role/faction filters."""
    reserves = set(world.get("reserves", []))
    views = []
    for pid, p in world["players"].items():
        if not include_reserves and (pid in reserves or p.get("status") == "reserve"):
            continue
        if role is not None and p.get("role") != role:
            continue
        if faction is not None and p.get("faction") != faction:
            continue
        views.append(player_view(world, pid))
    views.sort(key=lambda v: -v["rating"])
    for i, v in enumerate(views, 1):
        v["rank"] = i
    return views


# --------------------------------------------------------------------------- #
# Leagues — split by league AND group, with places
# --------------------------------------------------------------------------- #
def _last_ranking(world: dict, league: str) -> Optional[list]:
    hist = world.get("history") or []
    if not hist:
        return None
    return hist[-1].get("upper_ranking" if league == "upper" else "lower_ranking")


def _standings_key(league: str, group: Optional[str]) -> str:
    return "upper" if league == "upper" else f"lower_{group}"


def _last_standings(world: dict, league: str, group: Optional[str]) -> dict:
    """The most recent season's group standings (points/record/place) keyed by
    team_id, or empty before any season has been played."""
    hist = world.get("history") or []
    if not hist:
        return {}
    return (hist[-1].get("standings") or {}).get(_standings_key(league, group), {})


def _team_row(world: dict, t: dict, place: int, stand: Optional[dict] = None) -> dict:
    s = stand or {}
    return {
        "team_id": t["team_id"], "name": t["name"], "abbrev": t["abbrev"],
        "league": t.get("league"), "group": t.get("group"),
        "power": team_power(world, t), "elo": round(float(t.get("elo", 0))),
        "titles": t.get("titles", 0),
        "treasury": round(float(t.get("treasury", 0))), "place": place,
        # points + map delta are visible; cores (the hidden tiebreaker) never appear
        "points": s.get("points", 0),
        "series_wins": s.get("series_wins", 0), "series_losses": s.get("series_losses", 0),
        "maps_won": s.get("maps_won", 0), "maps_lost": s.get("maps_lost", 0),
        "map_diff": s.get("map_diff", 0),
        "roster": [world["players"][pid]["nickname"]
                   for pid in t.get("roster", []) if pid in world["players"]],
    }


def _table(world: dict, league: str, group: Optional[str]) -> list[dict]:
    from src.world import live_season
    rev = live_season.revealed(world)
    if rev is not None:
        # A day-by-day season is being revealed -> show the standings AS OF the revealed day (at
        # day 0 everyone is on 0 points; each revealed day fills them in). Group membership is who
        # is playing this season's group.
        out = []
        for i, r in enumerate(rev["standings"][_standings_key(league, group)]):
            if r["team_id"] not in world["teams"]:
                continue
            row = _team_row(world, world["teams"][r["team_id"]], i + 1, r)
            row["league"], row["group"] = league, group
            out.append(row)
        return out
    standings = _last_standings(world, league, group)
    if standings:
        # Promotion/relegation is applied as soon as a season ends, so CURRENT league
        # membership already differs from who actually played. Drive the table off the
        # recorded participants (in their final places) so points/positions are coherent.
        ordered = sorted(standings.items(), key=lambda kv: kv[1].get("place", 99))
        rows = []
        for i, (tid, s) in enumerate(ordered):
            if tid not in world["teams"]:
                continue
            row = _team_row(world, world["teams"][tid], i + 1, s)
            row["league"], row["group"] = league, group   # the campaign's slot, not the post-promo one
            rows.append(row)
        return rows
    # No season played yet -> current membership ordered by global power.
    teams = [t for t in world["teams"].values()
             if t.get("league") == league and (group is None or t.get("group") == group)]
    teams.sort(key=lambda t: -team_power(world, t))
    return [_team_row(world, t, i + 1) for i, t in enumerate(teams)]


def league_tables(world: dict) -> dict:
    """Standings the Leagues tab switches between: upper (10), lower group A (6),
    lower group Z (6). Each row carries its place + power + roster nicknames."""
    return {
        "upper": _table(world, "upper", None),
        "lower_A": _table(world, "lower", "A"),
        "lower_Z": _table(world, "lower", "Z"),
    }


# --------------------------------------------------------------------------- #
# Team page
# --------------------------------------------------------------------------- #
def team_record(world: dict, team_id: str, *, season: Optional[int] = None) -> dict:
    """Series W/L (and map W/L) for a team from the played calendar."""
    sw = sl = mw = ml = 0
    for e in world.get("calendar", []):
        if not e.get("played"):
            continue
        if season is not None and e.get("season") != season:
            continue
        if team_id not in (e.get("home"), e.get("away")):
            continue
        is_home = e["home"] == team_id
        my_maps = e["maps_home"] if is_home else e["maps_away"]
        opp_maps = e["maps_away"] if is_home else e["maps_home"]
        mw += int(my_maps or 0)
        ml += int(opp_maps or 0)
        if e.get("winner") == team_id:
            sw += 1
        else:
            sl += 1
    return {"series_wins": sw, "series_losses": sl, "maps_won": mw, "maps_lost": ml}


def _abbr(world: dict, tid: str) -> str:
    return world["teams"][tid]["abbrev"] if tid in world.get("teams", {}) else tid


def _match_summary(world: dict, team_id: str, s: dict, played: bool, season_num: int) -> dict:
    is_home = s.get("home") == team_id
    opp = s.get("away") if is_home else s.get("home")
    mf = (s.get("maps_home") if is_home else s.get("maps_away")) if played else None
    ma = (s.get("maps_away") if is_home else s.get("maps_home")) if played else None
    # map-by-map winners from THIS team's view (W/L per map), when the engine recorded a sequence
    seq = s.get("sequence") if played else None
    maps_seq = [("W" if w == team_id else "L") for w in seq] if seq else None
    return {"season": season_num, "day": s.get("day"), "stage": s.get("stage"),
            "stage_label": STAGE_LABEL.get(s.get("stage"), s.get("stage")),
            "opponent": opp, "opponent_abbrev": _abbr(world, opp),
            "opponent_name": (world["teams"][opp]["name"] if opp in world.get("teams", {}) else opp),
            "home": is_home, "best_of": s.get("best_of"),
            "played": played, "maps_for": mf, "maps_against": ma, "maps_seq": maps_seq,
            "won": (s.get("winner") == team_id) if played else None}


def team_matches(world: dict, team_id: str) -> list[dict]:
    """The team's matches THIS season: during a day-by-day reveal from the live store (results only
    for revealed days, upcoming fixtures shown without a result); otherwise the last completed
    season from the calendar. Sorted by day."""
    from src.world import live_season
    rev = live_season.revealed(world)
    out = []
    if rev is not None:
        _revset = set(rev["revealed_gids"])
        for s in list(rev["all_regular"]) + list(rev["playoff_series"]):
            if team_id in (s.get("home"), s.get("away")):
                out.append(_match_summary(world, team_id, s, s["gid"] in _revset, rev["season"]))
    else:
        hist = world.get("history") or []
        season_num = hist[-1]["season"] if hist else world.get("season", 0)
        season_evs = [e for e in world.get("calendar", []) if e.get("season") == season_num]
        day_of = _day_assigner(season_evs)                       # 2 regular series/day, playoffs 1/day
        for e in season_evs:
            if team_id in (e.get("home"), e.get("away")):
                m = _match_summary(world, team_id, e, e.get("played", False), season_num)
                m["day"] = day_of(e)
                out.append(m)
    out.sort(key=lambda m: m.get("day") or 0)
    return out


def _team_transfers(world: dict, team_id: str) -> dict:
    """In/out moves involving the team from the most recent off-season's transactions."""
    tx = ((world.get("last_offseason") or {}).get("transactions") or {})
    incoming = [{"player": s["player"], "kind": s.get("kind"), "bonus": s.get("bonus"), "dir": "signed"}
                for s in tx.get("signings", []) if s.get("team_id") == team_id]
    incoming += [{"player": b["player"], "from": b.get("from_team"), "price": b.get("price"), "dir": "bought"}
                 for b in tx.get("buys", []) if b.get("to_team_id") == team_id]
    outgoing = [{"player": b["player"], "to": b.get("to_team"), "price": b.get("price"), "dir": "sold"}
                for b in tx.get("buys", []) if b.get("from_team_id") == team_id]
    outgoing += [{"player": r["player"], "dir": "released"}
                 for r in tx.get("releases", []) if r.get("team_id") == team_id]
    return {"in": incoming, "out": outgoing}


def _team_season_history(world: dict, team_id: str) -> list[dict]:
    """Per-season finish: which league + final place + champion/promotion flags."""
    out = []
    for h in world.get("history", []):
        up, lo = h.get("upper_ranking") or [], h.get("lower_ranking") or []
        if team_id in up:
            place, league = up.index(team_id) + 1, "upper"
        elif team_id in lo:
            place, league = lo.index(team_id) + 1, "lower"
        else:
            continue
        out.append({"season": h["season"], "league": league, "place": place,
                    "champion": team_id in (h.get("upper_champion"), h.get("lower_champion")),
                    "promoted": team_id in (h.get("promoted") or []),
                    "relegated": team_id in (h.get("relegated") or [])})
    return out


def team_power_ranking(world: dict) -> list[dict]:
    """GLOBAL power ranking of ALL clubs (not split by league) — for the Teams tab sort."""
    rows = [{"team_id": t["team_id"], "name": t["name"], "abbrev": t["abbrev"],
             "league": t.get("league"), "group": t.get("group"),
             "power": team_power(world, t), "titles": t.get("titles", 0),
             "treasury": round(float(t.get("treasury", 0))),
             "roster": [world["players"][pid]["nickname"]
                        for pid in t.get("roster", []) if pid in world["players"]]}
            for t in world["teams"].values()]
    rows.sort(key=lambda r: (-r["power"], r["name"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def _power_rank_of(world: dict, team_id: str) -> int:
    for r in team_power_ranking(world):
        if r["team_id"] == team_id:
            return r["rank"]
    return 0


def _live_record(world: dict, team_id: str) -> dict:
    """Series/map W-L from team_matches (so it tracks the revealed day during a live season)."""
    sw = sl = mw = ml = 0
    for m in team_matches(world, team_id):
        if not m["played"]:
            continue
        mw += int(m["maps_for"] or 0)
        ml += int(m["maps_against"] or 0)
        sw += 1 if m["won"] else 0
        sl += 0 if m["won"] else 1
    return {"series_wins": sw, "series_losses": sl, "maps_won": mw, "maps_lost": ml}


_STRAT_RU = {"standard": "Стандарт", "core_rush": "Раш ядра", "all_in": "Олл-ин",
             "turtle": "Черепаха", "macro": "Макро-эконом", "baron": "Контроль Лорда"}
_FAC_RU = {"humans": "Humans", "infected": "Infected", "robots": "Robots"}


def _readable_when(when: dict) -> str:
    """Translate a playbook condition dict into a human IF-clause (Russian)."""
    parts = []
    if "opp_faction_majority" in when:
        parts.append(f"соперник в основном {_FAC_RU.get(when['opp_faction_majority'], when['opp_faction_majority'])}")
    if when.get("stage") == "playoff":
        parts.append("в плей-офф")
    if "opp_last_strategy" in when:
        parts.append(f"соперник прошлый раз играл «{_STRAT_RU.get(when['opp_last_strategy'], when['opp_last_strategy'])}»")
    if "series_lost" in when:
        parts.append(f"уступаем {when['series_lost']}+ карт в серии")
    if "series_won" in when:
        parts.append(f"ведём {when['series_won']}+ карт в серии")
    if "playoff_losing_streak" in when:
        parts.append(f"серия из {when['playoff_losing_streak']}+ поражений")
    if when.get("series_ahead"):
        parts.append("ведём в серии")
    if when.get("series_behind"):
        parts.append("уступаем в серии")
    if "vs_team" in when:
        parts.append("против конкретного клуба")
    return ", ".join(parts) or "—"


def team_playbook(world: dict, team_id: str) -> dict:
    """The team's tournament PLAYBOOK in readable form: a base strategy + IF-THEN overrides (checked
    in order, first match wins). The manager authored this before the season."""
    try:
        from src.world import playbook as _pb
        plan = _pb.get_plan(world, team_id)
    except Exception:
        plan = (world["teams"].get(team_id, {}) or {}).get("tournament_plan") or {}
    base = plan.get("base") or {}
    overrides = [{"when": _readable_when(ov.get("when") or {}),
                  "strategy": _STRAT_RU.get(ov.get("strategy"), ov.get("strategy")),
                  "strategy_id": ov.get("strategy"), "note": ov.get("note")}
                 for ov in (plan.get("overrides") or [])]
    return {"team_id": team_id, "name": world["teams"].get(team_id, {}).get("name", team_id),
            "base_strategy": _STRAT_RU.get(base.get("strategy"), base.get("strategy") or "—"),
            "base_strategy_id": base.get("strategy"), "overrides": overrides}


def team_view(world: dict, team_id: str) -> dict:
    """Full per-team page (liquipedia-style): identity + global power rank, power curve, trophies,
    roster, THIS season's record (live-aware), played + upcoming matches, transfers, season history."""
    from src.world import live_season
    t = world["teams"][team_id]
    matches = team_matches(world, team_id)
    # Power curve: past seasons' canonical Elo; during a day-by-day reveal extend it with the
    # PROVISIONAL Power so the curve tracks the revealed games (never the unrevealed final).
    elo_hist = [round(float(e)) for e in t.get("elo_history", [])]
    rp = live_season.revealed_power(world)
    if rp is not None:
        elo_hist = elo_hist + [rp.get(team_id, elo_hist[-1] if elo_hist else round(float(t.get("elo", 1000.0))))]
    return {
        "team_id": team_id, "name": t["name"], "abbrev": t["abbrev"],
        "manager": t.get("manager"), "league": t.get("league"), "group": t.get("group"),
        "power": team_power(world, t), "elo": round(float(t.get("elo", 0))),
        "power_rank": _power_rank_of(world, team_id), "n_teams": len(world["teams"]),
        "titles": t.get("titles", 0),
        "treasury": round(float(t.get("treasury", 0))),
        "elo_history": elo_hist,
        "trophies": t.get("trophies", []),
        "certificates": len(t.get("training_certificates", [])),
        "roster": [player_view(world, pid) for pid in t.get("roster", [])
                   if pid in world["players"]],
        "record": _live_record(world, team_id),
        "matches_played": [m for m in matches if m["played"]],
        "matches_upcoming": [m for m in matches if not m["played"]],
        "transfers": _team_transfers(world, team_id),
        "season_history": _team_season_history(world, team_id),
        "playbook": team_playbook(world, team_id),
    }


# --------------------------------------------------------------------------- #
# Schedule + playoff bracket
# --------------------------------------------------------------------------- #
def _day_assigner(evs: list[dict]):
    """Map each season event to its schedule DAY. Trust the stored day when the engine tracked it
    (a live reveal preserves the real grouping: 2 regular series/day, playoffs 1/day); otherwise
    assign 2 regular series per day, then playoffs 1/day — so the schedule ALWAYS reads '2 per day'
    like the LIVE tab, never one match per day."""
    regular = [e for e in evs if e.get("stage") == "league"]
    playoff = [e for e in evs if e.get("stage") != "league"]
    if regular and all(e.get("day") for e in evs):
        return lambda e: e.get("day") or 0
    day = {}
    for i, e in enumerate(regular):
        day[id(e)] = i // 2 + 1
    base = (len(regular) + 1) // 2
    for j, e in enumerate(playoff):
        day[id(e)] = base + j + 1
    return lambda e: day.get(id(e)) or (e.get("day") or 0)


def _match_row(world: dict, e: dict, day: int) -> dict:
    def nm(tid):
        return world["teams"][tid]["abbrev"] if tid in world["teams"] else tid

    def full(tid):
        return world["teams"][tid]["name"] if tid in world["teams"] else tid

    # live series carry group='lower_A'/'lower_Z'; the finished-season calendar event doesn't store it,
    # so fall back to the home club's group ('A'/'Z') — leagueBadge tolerates both forms (endsWith 'Z').
    group = e.get("group") or world["teams"].get(e.get("home"), {}).get("group")
    return {
        "day": day, "league": e.get("league"), "group": group, "stage": e.get("stage"),
        "stage_label": STAGE_LABEL.get(e.get("stage"), e.get("stage")),
        "home": e.get("home"), "away": e.get("away"),
        "home_abbrev": nm(e.get("home")), "away_abbrev": nm(e.get("away")),
        "home_name": full(e.get("home")), "away_name": full(e.get("away")),
        "best_of": e.get("best_of"), "played": e.get("played", False),
        "maps_home": e.get("maps_home"), "maps_away": e.get("maps_away"),
        "winner": e.get("winner"),
        "home_strategy": e.get("home_strategy"), "away_strategy": e.get("away_strategy"),
    }


def draft_view(world: dict) -> dict:
    """The latest recorded draft board (bootstrap or rookie) — pick order with team + player."""
    drafts = world.get("drafts") or []
    if not drafts:
        return {"season": None, "kind": None, "picks": [], "n": 0}
    d = drafts[-1]
    picks = []
    for p in d["picks"]:
        t = world["teams"].get(p["team_id"])
        pl = world["players"].get(p["player_id"])
        picks.append({
            "round": p.get("round"), "overall": p.get("overall"), "team_id": p["team_id"],
            "team": t["abbrev"] if t else p["team_id"], "team_name": t["name"] if t else p["team_id"],
            "player": pl["nickname"] if pl else p["player_id"],
            "faction": pl.get("faction") if pl else None,
            "archetype": pl.get("archetype") if pl else None,
            "rating": round(float(pl.get("soloq_rating", 0))) if pl else None,
        })
    return {"season": d["season"], "kind": d["kind"], "picks": picks, "n": len(picks)}


def _live_bracket(world: dict, rev: dict) -> dict:
    """The playoff bracket in schedule shape — shown FROM DAY 0 with TBD slots and filled
    progressively: round-1 SEEDS appear only once the regular season is fully revealed (never a
    spoiler), later matchups + results appear as their feeder games are revealed. Columns are by
    bracket DEPTH (round), so the graph reads left-to-right as a proper tree."""
    from src.world import live_season
    revset = set(rev["revealed_gids"])
    done_by_lg = rev.get("regular_done_by_league") or {"upper": rev["regular_done"], "lower": rev["regular_done"]}

    def cols(bracket, regular_done):
        def slot(ref, nodes):
            if isinstance(ref, str):                 # a concrete seed — hide until THIS league's regular is done
                return ref if regular_done else None
            return live_season.resolve_for_reveal(ref, nodes, revset)
        nodes = bracket["nodes"]
        depth: dict = {}

        def dof(key):                                # longest path from a seed = the round number
            if key not in depth:
                depth[key] = 0                       # cycle guard (parity with the JS bgLeague side)
                n = nodes[key]
                fed = [(r.get("w") or r.get("l")) for r in (n["home"], n["away"]) if isinstance(r, dict)]
                depth[key] = 0 if not fed else 1 + max(dof(f) for f in fed)
            return depth[key]
        for key in bracket["order"]:
            dof(key)
        by_depth: dict = {}
        for key in bracket["order"]:
            n = nodes[key]
            home, away = slot(n["home"], nodes), slot(n["away"], nodes)
            shown = n.get("gid") in revset
            res = n.get("result")
            # connector lines show only where the WINNER advances (loser-drops are NOT drawn)
            feeders = [r["w"] for r in (n["home"], n["away"]) if isinstance(r, dict) and "w" in r]
            by_depth.setdefault(depth[key], []).append({
                "stage": n["stage"], "stage_label": STAGE_LABEL.get(n["stage"], n["stage"]),
                "key": key, "from": feeders,            # winner-advance feeders for the connector lines
                "home": home, "away": away,
                "home_abbrev": _abbr(world, home) if home else "TBD",
                "away_abbrev": _abbr(world, away) if away else "TBD",
                "best_of": n.get("best_of"), "played": shown,
                "maps_home": res["maps_home"] if shown else None,
                "maps_away": res["maps_away"] if shown else None,
                "winner": res["winner"] if shown else None})
        out = []
        maxd = max(by_depth) if by_depth else 0
        for d in sorted(by_depth):
            ms = by_depth[d]
            stages = {m["stage"] for m in ms}
            label = (STAGE_LABEL.get(next(iter(stages)), next(iter(stages))) if len(stages) == 1
                     else ("Гранд-финал" if d == maxd else f"Раунд {d + 1}"))
            out.append({"stage": f"d{d}", "label": label, "matches": ms})
        return out

    return {"upper": cols(rev["playoff"]["upper"], done_by_lg["upper"]),
            "lower": cols(rev["playoff"]["lower"], done_by_lg["lower"])}


def _live_schedule(world: dict, rev: dict) -> dict:
    """Schedule during a day-by-day reveal. The full regular fixture list is shown FROM DAY 0
    (who plays whom on which day) — a fixture's result appears only once that game is revealed.
    The playoff bracket fills in as it's decided."""
    revset = set(rev["revealed_gids"])

    def ev(s):
        played = s.get("gid") in revset
        pb = s.get("playbook") or {}
        return {**s, "played": played,
                "maps_home": s.get("maps_home") if played else None,
                "maps_away": s.get("maps_away") if played else None,
                "winner": s.get("winner") if played else None,
                "home_strategy": (pb.get("home") or {}).get("strategy") if played else None,
                "away_strategy": (pb.get("away") or {}).get("strategy") if played else None}
    regular = [_match_row(world, ev(s), s.get("day", 0)) for s in rev["all_regular"]]
    return {"season": rev["season"], "regular": regular, "bracket": _live_bracket(world, rev),
            "n_matches": len(rev["all_regular"]) + len(rev["playoff_series"])}


def history_view(world: dict) -> list:
    """Past seasons, enriched with team NAMES + prize money + top finishers + playoff result — for a
    rich «История» tab. Reads the recorded summaries (finalize_season); team ids resolve to current
    names (clubs persist across promotion/relegation)."""
    def nm(tid):
        t = world["teams"].get(tid)
        return {"team_id": tid, "name": t["name"] if t else tid, "abbrev": t["abbrev"] if t else tid}

    out = []
    for h in world.get("history", []):
        money = h.get("money") or {}
        play = h.get("playoff") or {}
        up_rank = h.get("upper_ranking", [])
        lo_rank = h.get("lower_ranking", [])
        std = h.get("standings") or {}
        breakdown = money.get("prize_breakdown") or {}
        pbt = money.get("prizes_by_team", {})

        def row_for(tid, place):
            b = breakdown.get(tid, {})
            return {**nm(tid), "place": b.get("place") or place, "prize": b.get("prize", 0),
                    "bonus": b.get("bonus", 0), "total": b.get("total", pbt.get(tid, 0))}

        # Prize pool by POOL: upper (one league), lower-A and lower-Z (SEPARATE, equal pools — never
        # mixed into one 1..12 column). Each group is ordered by its own place.
        upper_rows = [row_for(tid, i + 1) for i, tid in enumerate(up_rank)]

        def group_rows(gkey):
            members = std.get(gkey) or {}              # {tid: {place, ...}}
            ordered = sorted(members.items(), key=lambda kv: kv[1].get("place", 99))
            return [row_for(tid, s.get("place")) for tid, s in ordered]

        prize_pool = {"upper": upper_rows, "lower_A": group_rows("lower_A"),
                      "lower_Z": group_rows("lower_Z")}
        lower_rows = [row_for(tid, i + 1) for i, tid in enumerate(lo_rank)]   # combined, for the top-3 list

        up_play, lo_play = play.get("upper", {}), play.get("lower", {})
        out.append({
            "season": h.get("season"),
            # the GRAND-FINAL WINNER of each league (this is the season champion)
            "upper_champion": nm(h.get("upper_champion")),
            "lower_champion": nm(h.get("lower_champion")),
            # BOTH grand-final teams per league (winner first), so the UI can show "winner — beat finalist"
            "upper_finalists": [nm(t) for t in (up_play.get("grand_finalists") or [])],
            "lower_finalists": [nm(t) for t in (lo_play.get("grand_finalists") or [])],
            "promoted": [nm(t) for t in h.get("promoted", [])],     # BOTH lower grand-finalists go up
            "relegated": [nm(t) for t in h.get("relegated", [])],
            "upper_top": [dict(r) for r in upper_rows[:3]],
            "lower_top": [dict(r) for r in lower_rows[:3]],
            "prizes_by_team": pbt,
            "prize_pool": prize_pool,
            "bracket": _season_bracket(world, h.get("season")),   # playoff bracket for the History tab
            "n_matches": len(up_rank) and 85,        # 75 league + 10 playoff series (fixed format)
        })
    return out


def _season_bracket(world: dict, season: int) -> dict:
    """The playoff bracket for a PAST season (from the recorded calendar) in the same shape the
    bracket graph renders — so the History tab can show the sketch with connectors."""
    if season is None:
        return {}
    playoff: dict = {}
    for e in world.get("calendar", []):
        if e.get("season") != season or e.get("stage") == "league":
            continue
        lg = e.get("league")
        playoff.setdefault(lg, {}).setdefault(e.get("stage"), []).append(_match_row(world, e, 0))
    bracket = {}
    for lg, by_stage in playoff.items():
        _assign_bracket_feeders(by_stage)
        bracket[lg] = [{"stage": s, "label": STAGE_LABEL.get(s, s), "matches": by_stage[s]}
                       for s in PLAYOFF_STAGE_ORDER if s in by_stage]
    return bracket


def _assign_bracket_feeders(by_stage: dict) -> None:
    """Tag canonical (calendar-derived) playoff matches with ``key`` + feeder ``from`` keys so the
    bracket graph draws connectors for a FINISHED season too (the live path already carries them).
    Keys mirror live_season's bracket DAGs (upper league: ub/lb/lbf/gf; lower: ub/lb1a/lb1b/lb2/lbf/gf)."""
    def tag(stage, key, frm):
        for m in by_stage.get(stage, []):
            m["key"], m["from"] = key, list(frm)
    # NB: `from` lists only WINNER-advance feeders (loser-drops aren't drawn as lines).
    if "lb_r1" in by_stage:                         # lower league (2 first-round losers matches)
        for i, m in enumerate(by_stage.get("lb_r1", [])):
            m["key"], m["from"] = f"lb1{chr(ord('a') + i)}", []
        r1_keys = [m["key"] for m in by_stage.get("lb_r1", [])]
        tag("ub", "ub", [])
        tag("lb_r2", "lb2", r1_keys)
        tag("lb_final", "lbf", ["lb2"])             # lb2 WINNER advances (ub loser drops here — no line)
        tag("grand_final", "gf", ["ub", "lbf"])
    else:                                           # upper league (single loser-bracket round)
        tag("ub", "ub", [])
        tag("lb", "lb", [])
        tag("lb_final", "lbf", ["lb"])              # lb WINNER advances (ub loser drops here — no line)
        tag("grand_final", "gf", ["ub", "lbf"])


def schedule_view(world: dict, season: Optional[int] = None) -> dict:
    """A season's fixtures as a day-by-day list (regular: 2 series/day, playoffs 1/day — the SAME
    grouping the LIVE tab uses) plus the playoff bracket grouped per league + stage. During a
    day-by-day reveal it shows only what's been revealed."""
    from src.world import live_season
    rev = live_season.revealed(world)
    if rev is not None:
        return _live_schedule(world, rev)
    season = world["season"] if season is None else season
    evs = [e for e in world.get("calendar", []) if e.get("season") == season]
    day_of = _day_assigner(evs)                              # 2 regular series/day, playoffs 1/day
    regular, playoff = [], {}
    for e in evs:
        row = _match_row(world, e, day_of(e))
        if e.get("stage") == "league":
            regular.append(row)
        else:
            lg = e.get("league")
            playoff.setdefault(lg, {}).setdefault(e.get("stage"), []).append(row)
    regular.sort(key=lambda m: m.get("day") or 0)
    # order each league's stages for a left-to-right bracket (+ feeder keys for the connector graph)
    bracket = {}
    for lg, by_stage in playoff.items():
        _assign_bracket_feeders(by_stage)
        bracket[lg] = [{"stage": s, "label": STAGE_LABEL.get(s, s), "matches": by_stage[s]}
                       for s in PLAYOFF_STAGE_ORDER if s in by_stage]
    return {"season": season, "regular": regular, "bracket": bracket,
            "n_matches": len(evs)}


# --------------------------------------------------------------------------- #
# League table tab — H2H matrix (who-beat-whom + score) + current positions
# --------------------------------------------------------------------------- #
def league_matrix(world: dict, season: Optional[int] = None) -> dict:
    """For the 'Таблица' tab: per group, a square head-to-head grid (cell = the
    series score from the row team's view) plus the current standings (place +
    points + map delta). Built from the recorded group-stage calendar; the hidden
    cores tiebreaker is never exposed — only the resulting place is. During a day-by-day reveal the
    grid + positions reflect only the matches revealed so far."""
    from src.world import live_season
    rev = live_season.revealed(world)
    hist = world.get("history") or []
    if season is None:
        season = rev["season"] if rev else (hist[-1]["season"] if hist else world.get("season", 0))

    def _series_for(group_key: str) -> list[dict]:
        if rev is not None:                              # revealed-so-far regular series for the group
            return rev["regular"].get(group_key, [])
        return [e for e in world.get("calendar", [])     # last completed season, from the calendar
                if e.get("season") == season and e.get("stage") == "league" and e.get("played")]

    def grid(league: str, group: Optional[str], label: str) -> dict:
        rows = _table(world, league, group)              # ordered by place, with points
        ids = [r["team_id"] for r in rows]
        idset = set(ids)
        h2h: dict = {}
        for e in _series_for(_standings_key(league, group)):
            h, a = e.get("home"), e.get("away")
            if h not in idset or a not in idset:
                continue
            mh, ma = int(e.get("maps_home") or 0), int(e.get("maps_away") or 0)
            h2h[(h, a)] = {"for": mh, "against": ma, "won": e.get("winner") == h}
            h2h[(a, h)] = {"for": ma, "against": mh, "won": e.get("winner") == a}
        cells = [[None if r == c else h2h.get((r, c)) for c in ids] for r in ids]
        teams = [{"team_id": r["team_id"], "abbrev": r["abbrev"], "name": r["name"],
                  "place": r["place"], "points": r["points"], "map_diff": r["map_diff"],
                  "series_wins": r["series_wins"], "series_losses": r["series_losses"]}
                 for r in rows]
        return {"league": league, "group": group, "label": label,
                "teams": teams, "cells": cells}

    return {"season": season, "groups": [
        grid("upper", None, "Высшая лига"),
        grid("lower", "A", "Низшая · группа A"),
        grid("lower", "Z", "Низшая · группа Z"),
    ]}
