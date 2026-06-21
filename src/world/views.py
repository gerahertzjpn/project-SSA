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
                    "opponents": [_nick(x) for x in (m.get("opponents") or [])]}
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
    return base


def ranked_players(world: dict, *, role: Optional[str] = None,
                   faction: Optional[str] = None) -> list[dict]:
    """Every player ranked by SoloQ rating (1 = best), with rank + per-season delta.
    Optional filters by role ('pro'/'streamer') and faction."""
    views = []
    for pid, p in world["players"].items():
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


def team_view(world: dict, team_id: str) -> dict:
    """Full per-team page: identity, league/group, power curve, trophies, roster, record."""
    t = world["teams"][team_id]
    return {
        "team_id": team_id, "name": t["name"], "abbrev": t["abbrev"],
        "manager": t.get("manager"), "league": t.get("league"), "group": t.get("group"),
        "power": team_power(world, t), "elo": round(float(t.get("elo", 0))),
        "titles": t.get("titles", 0),
        "treasury": round(float(t.get("treasury", 0))),
        "elo_history": [round(float(e)) for e in t.get("elo_history", [])],
        "trophies": t.get("trophies", []),
        "certificates": len(t.get("training_certificates", [])),
        "roster": [player_view(world, pid) for pid in t.get("roster", [])
                   if pid in world["players"]],
        "record": team_record(world, team_id),
    }


# --------------------------------------------------------------------------- #
# Schedule + playoff bracket
# --------------------------------------------------------------------------- #
def _match_row(world: dict, e: dict, day: int) -> dict:
    def nm(tid):
        return world["teams"][tid]["abbrev"] if tid in world["teams"] else tid

    def full(tid):
        return world["teams"][tid]["name"] if tid in world["teams"] else tid

    return {
        "day": day, "league": e.get("league"), "stage": e.get("stage"),
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


def _live_schedule(world: dict, rev: dict) -> dict:
    """Schedule view during a day-by-day reveal — only the matches revealed so far (from the live
    store, since the calendar isn't written until the season is finalised)."""
    def ev(s):
        pb = s.get("playbook") or {}
        return {**s, "played": True, "home_strategy": (pb.get("home") or {}).get("strategy"),
                "away_strategy": (pb.get("away") or {}).get("strategy")}
    flat_reg = rev["regular"]["upper"] + rev["regular"]["lower_A"] + rev["regular"]["lower_Z"]
    regular = [_match_row(world, ev(s), i) for i, s in enumerate(flat_reg, 1)]
    playoff: dict = {}
    for s in rev["playoff_series"]:
        playoff.setdefault(s.get("league"), {}).setdefault(s.get("stage"), []).append(
            _match_row(world, ev(s), 0))
    bracket = {}
    for lg, by_stage in playoff.items():
        bracket[lg] = [{"stage": st, "label": STAGE_LABEL.get(st, st), "matches": by_stage[st]}
                       for st in PLAYOFF_STAGE_ORDER if st in by_stage]
    return {"season": rev["season"], "regular": regular, "bracket": bracket,
            "n_matches": len(flat_reg) + len(rev["playoff_series"])}


def schedule_view(world: dict, season: Optional[int] = None) -> dict:
    """A season's fixtures as a day-by-day list (1 series = 1 day, in calendar order)
    plus the playoff bracket grouped per league + stage for a bracket render. During a day-by-day
    reveal it shows only what's been revealed."""
    from src.world import live_season
    rev = live_season.revealed(world)
    if rev is not None:
        return _live_schedule(world, rev)
    season = world["season"] if season is None else season
    evs = [e for e in world.get("calendar", []) if e.get("season") == season]
    regular, playoff = [], {}
    for i, e in enumerate(evs, 1):
        row = _match_row(world, e, i)
        if e.get("stage") == "league":
            regular.append(row)
        else:
            lg = e.get("league")
            playoff.setdefault(lg, {}).setdefault(e.get("stage"), []).append(row)
    # order each league's stages for a left-to-right bracket
    bracket = {}
    for lg, by_stage in playoff.items():
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
