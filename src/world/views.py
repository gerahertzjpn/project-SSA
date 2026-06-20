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

# Playoff stages in bracket order (for rendering a bracket left-to-right).
PLAYOFF_STAGE_ORDER = ("lb_r1", "lb_r2", "ub", "lb_final", "grand_final")


def team_power(world: dict, team: dict) -> int:
    """Team Power = the team's own RESULTS (Elo), NOTHING ELSE. It must not depend on the players'
    SoloQ ratings: a club fields many players and can swap them mid-series, so strength is a property
    of the CLUB's results, not its current roster's individual ratings. Elo is global across leagues
    because it's one continuous per-club number, seeded at bootstrap from who earned promotion (a
    result) and carried through promotion/relegation — so upper clubs generally rank above lower
    ones without that being absolute."""
    return round(float(team.get("elo", 1000.0)))
STAGE_LABEL = {
    "ub": "Верхняя сетка", "lb_r1": "Нижняя сетка — 1", "lb_r2": "Нижняя сетка — 2",
    "lb_final": "Финал нижней сетки", "grand_final": "Гранд-финал", "league": "Регулярка",
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
        "trophies": p.get("trophies", []),
    }


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


def _team_row(world: dict, t: dict, place: int) -> dict:
    return {
        "team_id": t["team_id"], "name": t["name"], "abbrev": t["abbrev"],
        "league": t.get("league"), "group": t.get("group"),
        "power": team_power(world, t), "elo": round(float(t.get("elo", 0))),
        "titles": t.get("titles", 0),
        "treasury": round(float(t.get("treasury", 0))), "place": place,
        "roster": [world["players"][pid]["nickname"]
                   for pid in t.get("roster", []) if pid in world["players"]],
    }


def _table(world: dict, league: str, group: Optional[str]) -> list[dict]:
    teams = [t for t in world["teams"].values()
             if t.get("league") == league and (group is None or t.get("group") == group)]
    ranking = _last_ranking(world, league)

    def place_key(t):
        if ranking and t["team_id"] in ranking:
            return (0, ranking.index(t["team_id"]))
        return (1, -team_power(world, t))         # no ranking yet -> by global power

    teams.sort(key=place_key)
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


def schedule_view(world: dict, season: Optional[int] = None) -> dict:
    """A season's fixtures as a day-by-day list (1 series = 1 day, in calendar order)
    plus the playoff bracket grouped per league + stage for a bracket render."""
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
