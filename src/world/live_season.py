"""Play a whole season ONCE, then reveal it DAY BY DAY (no spoilers).

The owner's model (chosen explicitly): the season is a SINGLE canonical play-through with the
REAL match engine — exactly the same matches "Сыграть сезон" (`run_regular_season`) would play —
captured into a per-day structure. The UI then *reveals* it one day at a time, and only what's
been revealed exists for the viewer: standings, the playoff bracket and provisional Power fill in
as you step through the days; nothing past the cursor is shown. Watching a match in the broadcast
just replays its already-decided map sequence (no scrub).

So this is NOT a separate/parallel simulation and it does NOT change the match mechanic — it is the
same season, played up front, revealed progressively.

Spoiler-safety: ``play_and_store`` plays every series but does NOT touch canonical state (no Elo,
no calendar, no history, no promotion) — those other tabs stay blank for the season until the
reveal finishes. Provisional per-day Power is computed on a LOCAL Elo copy. When the last day is
revealed, the season is settled through the SAME :func:`season.finalize_season` the batch path uses
(Elo + calendar + economy + promotion + history), so the canonical outcome is identical.
"""
from __future__ import annotations

from typing import Optional

from src.world import brackets, season
from src.world.team_elo import series_elo_delta
from src.world.world import upper_teams

PLAYOFF_BEST_OF = brackets.PLAYOFF_BEST_OF


# --------------------------------------------------------------------------- #
# Round-robin scheduling (circle method) — split a group into balanced rounds
# --------------------------------------------------------------------------- #
def schedule_rounds(team_ids: list[str]) -> list[list[tuple[str, str]]]:
    """Split a single round-robin into rounds where each team plays at most once per round."""
    ids: list[Optional[str]] = list(team_ids)
    if len(ids) % 2:
        ids.append(None)
    n = len(ids)
    rounds = []
    arr = ids[:]
    for _ in range(n - 1):
        pairs = [(arr[i], arr[n - 1 - i]) for i in range(n // 2)
                 if arr[i] is not None and arr[n - 1 - i] is not None]
        rounds.append(pairs)
        arr = [arr[0]] + [arr[-1]] + arr[1:-1]
    return rounds


# --------------------------------------------------------------------------- #
# Playoff brackets as small DAGs (home/away are concrete ids or {"w"|"l": node})
# --------------------------------------------------------------------------- #
def _upper_bracket(seeds_: list[str]) -> dict:
    return {
        "order": ["ub", "lb", "lbf", "gf"],
        "nodes": {
            "ub":  {"stage": "ub", "home": seeds_[0], "away": seeds_[1], "best_of": PLAYOFF_BEST_OF},
            "lb":  {"stage": "lb", "home": seeds_[2], "away": seeds_[3], "best_of": PLAYOFF_BEST_OF},
            "lbf": {"stage": "lb_final", "home": {"w": "lb"}, "away": {"l": "ub"}, "best_of": PLAYOFF_BEST_OF},
            "gf":  {"stage": "grand_final", "home": {"w": "ub"}, "away": {"w": "lbf"}, "best_of": PLAYOFF_BEST_OF},
        },
        "finalists_of": ["ub", "lbf"],
    }


def _lower_bracket(a: list[str], z: list[str]) -> dict:
    return {
        "order": ["ub", "lb1a", "lb1b", "lb2", "lbf", "gf"],
        "nodes": {
            "ub":   {"stage": "ub", "home": a[0], "away": z[0], "best_of": PLAYOFF_BEST_OF},
            "lb1a": {"stage": "lb_r1", "home": a[1], "away": z[2], "best_of": PLAYOFF_BEST_OF},
            "lb1b": {"stage": "lb_r1", "home": z[1], "away": a[2], "best_of": PLAYOFF_BEST_OF},
            "lb2":  {"stage": "lb_r2", "home": {"w": "lb1a"}, "away": {"w": "lb1b"}, "best_of": PLAYOFF_BEST_OF},
            "lbf":  {"stage": "lb_final", "home": {"w": "lb2"}, "away": {"l": "ub"}, "best_of": PLAYOFF_BEST_OF},
            "gf":   {"stage": "grand_final", "home": {"w": "ub"}, "away": {"w": "lbf"}, "best_of": PLAYOFF_BEST_OF},
        },
        "finalists_of": ["ub", "lbf"],
    }


def _resolve(ref, nodes: dict, *, max_day: Optional[int] = None) -> Optional[str]:
    """Concrete team id for a node endpoint. With ``max_day`` set, a source result only counts
    if it was played on day <= max_day (so the reveal never leaks a future match's outcome)."""
    if isinstance(ref, str):
        return ref
    node = nodes.get(ref.get("w") or ref.get("l"))
    res = node and node.get("result")
    if not res:
        return None
    if max_day is not None and node.get("day", 0) > max_day:
        return None
    return res["winner"] if "w" in ref else (res["home"] if res["winner"] == res["away"] else res["away"])


def _ready_nodes(bracket: dict) -> list[str]:
    nodes = bracket["nodes"]
    return [k for k in bracket["order"] if not nodes[k].get("result")
            and _resolve(nodes[k]["home"], nodes) and _resolve(nodes[k]["away"], nodes)]


def _bracket_done(bracket: dict) -> bool:
    return all(bracket["nodes"][k].get("result") for k in bracket["order"])


# --------------------------------------------------------------------------- #
# Play the whole season up front, capture it day-by-day (no canonical mutation)
# --------------------------------------------------------------------------- #
def play_and_store(world: dict, *, play=None, seed: Optional[int] = None) -> dict:
    """Play the ENTIRE season (real engine via ``play``) and store it day-by-day on
    ``world['live_replay']`` for progressive reveal. Canonical state (Elo / calendar / history /
    promotion) is left untouched until the reveal finishes."""
    season_num = world["season"]
    play = play or season.make_play_fn(world, seed=seed)
    up_ids = [t["team_id"] for t in upper_teams(world)]
    a_ids = season._group_members(world, "A")
    z_ids = season._group_members(world, "Z")
    rounds = {"upper": schedule_rounds(up_ids), "lower_A": schedule_rounds(a_ids),
              "lower_Z": schedule_rounds(z_ids)}
    n_regular = max(len(rounds["upper"]), len(rounds["lower_A"]), len(rounds["lower_Z"]))

    results = {"upper": [], "lower_A": [], "lower_Z": []}
    local_elo = {tid: float(t["elo"]) for tid, t in world["teams"].items()}

    def _bump_elo(s: dict) -> None:                 # provisional Power, on a LOCAL copy only
        d = series_elo_delta(local_elo[s["home"]], local_elo[s["away"]],
                             s["maps_home"], s["maps_away"], s.get("best_of", 3))
        local_elo[s["home"]] += d
        local_elo[s["away"]] -= d

    days = []
    power_by_day = [{tid: round(e) for tid, e in local_elo.items()}]   # day 0 = pre-season

    # ---- regular season: one round-robin round per day ------------------ #
    for r in range(n_regular):
        series_today = []
        for gkey in ("upper", "lower_A", "lower_Z"):
            grounds = rounds[gkey]
            if r >= len(grounds):
                continue
            for home, away in grounds[r]:
                s = dict(play(home, away, brackets.LEAGUE_BEST_OF), stage="league",
                         group=gkey, league=_league_of(gkey))
                results[gkey].append(s)
                _bump_elo(s)
                series_today.append(s)
        days.append({"phase": "regular", "series": series_today})
        power_by_day.append({tid: round(e) for tid, e in local_elo.items()})

    # ---- seed playoffs from the (now complete) group standings ---------- #
    up_rows = brackets.standings(up_ids, results["upper"])
    a_rows = brackets.standings(a_ids, results["lower_A"])
    z_rows = brackets.standings(z_ids, results["lower_Z"])
    up_br = _upper_bracket([r["team_id"] for r in up_rows])
    lo_br = _lower_bracket([r["team_id"] for r in a_rows], [r["team_id"] for r in z_rows])

    # ---- playoffs: each day plays every currently-decidable series ------- #
    day_idx = n_regular
    while not (_bracket_done(up_br) and _bracket_done(lo_br)):
        day_idx += 1
        series_today = []
        for league_key, br in (("upper", up_br), ("lower", lo_br)):
            for key in _ready_nodes(br):
                node = br["nodes"][key]
                home = _resolve(node["home"], br["nodes"])
                away = _resolve(node["away"], br["nodes"])
                s = dict(play(home, away, node["best_of"]), stage=node["stage"], league=league_key)
                node["result"] = s
                node["day"] = day_idx
                _bump_elo(s)
                series_today.append(s)
        days.append({"phase": "playoff", "series": series_today})
        power_by_day.append({tid: round(e) for tid, e in local_elo.items()})

    n_days = len(days)
    world["live_replay"] = {
        "season": season_num,
        "cursor": 0,
        "n_days": n_days,
        "n_regular_days": n_regular,
        "groups": {"upper": up_ids, "lower_A": a_ids, "lower_Z": z_ids},
        "days": days,                              # full series incl cores (server-side only)
        "playoff": {"upper": up_br, "lower": lo_br},
        "power_by_day": power_by_day,
        "finalize": {
            "up_results": results["upper"], "a_results": results["lower_A"],
            "z_results": results["lower_Z"],
            "up_play": _playoff_outcome(up_br),
            "lo_play": _playoff_outcome(lo_br, promote=True),
        },
    }
    world["phase"] = "live"
    return world["live_replay"]


def _league_of(group_key: str) -> str:
    return "upper" if group_key == "upper" else "lower"


def _playoff_outcome(bracket: dict, *, promote: bool = False) -> dict:
    nodes = bracket["nodes"]
    finalists = [_resolve({"w": k}, nodes) for k in bracket["finalists_of"]]
    out = {"champion": nodes["gf"]["result"]["winner"], "grand_finalists": finalists,
           "series": [nodes[k]["result"] for k in bracket["order"] if nodes[k].get("result")]}
    if promote:
        out["promoted"] = finalists
    return out


# --------------------------------------------------------------------------- #
# Reveal — spoiler-free snapshot at the current cursor
# --------------------------------------------------------------------------- #
def is_live(world: dict) -> bool:
    ls = world.get("live_replay")
    return bool(ls) and ls.get("cursor", 0) < ls.get("n_days", 0)


def _team(world: dict, tid: Optional[str]) -> dict:
    t = world["teams"].get(tid) if tid else None
    return {"team_id": tid, "abbrev": t["abbrev"] if t else tid, "name": t["name"] if t else tid}


def _fixture(world: dict, s: dict) -> dict:
    """An upcoming match — matchup only, NO result (so it's safe to show before it's revealed)."""
    return {"home": s["home"], "away": s["away"],
            "home_abbrev": _team(world, s["home"])["abbrev"],
            "away_abbrev": _team(world, s["away"])["abbrev"],
            "stage": s.get("stage"), "group": s.get("group"), "league": s.get("league"),
            "best_of": s.get("best_of")}


def _result(world: dict, s: dict, *, seq: bool = False) -> dict:
    """A revealed match — score + winner, cores STRIPPED (hidden tiebreaker); sequence on request."""
    out = _fixture(world, s)
    out.update({"maps_home": s["maps_home"], "maps_away": s["maps_away"], "winner": s["winner"]})
    if seq:
        out["sequence"] = s.get("sequence", [])
    return out


def _standings_so_far(world: dict, ls: dict, cursor: int) -> dict:
    """Group standings from the REVEALED regular-season days only (cores used for the order, never
    exposed). Each row carries the provisional Power at this point in the reveal."""
    revealed = {"upper": [], "lower_A": [], "lower_Z": []}
    for day in ls["days"][:cursor]:
        if day["phase"] != "regular":
            continue
        for s in day["series"]:
            revealed[s["group"]].append(s)
    power = ls["power_by_day"][min(cursor, len(ls["power_by_day"]) - 1)]
    out = {}
    for gkey, ids in ls["groups"].items():
        rows = brackets.standings(ids, revealed[gkey])
        out[gkey] = [{"team_id": r["team_id"], "abbrev": _team(world, r["team_id"])["abbrev"],
                      "name": _team(world, r["team_id"])["name"], "place": r["place"],
                      "points": r["points"], "series_wins": r["series_wins"],
                      "series_losses": r["series_losses"], "map_diff": r["map_diff"],
                      "power": power.get(r["team_id"], round(float(world["teams"][r["team_id"]]["elo"])))}
                     for r in rows]
    return out


def _bracket_so_far(world: dict, bracket: Optional[dict], cursor: int) -> list[dict]:
    if not bracket:
        return []
    nodes = bracket["nodes"]
    out = []
    for key in bracket["order"]:
        node = nodes[key]
        home = _resolve(node["home"], nodes, max_day=cursor)
        away = _resolve(node["away"], nodes, max_day=cursor)
        res = node.get("result")
        shown = bool(res) and node.get("day", 0) <= cursor
        out.append({"key": key, "stage": node["stage"],
                    "home": _team(world, home) if home else None,
                    "away": _team(world, away) if away else None,
                    "played": shown,
                    "maps_home": res["maps_home"] if shown else None,
                    "maps_away": res["maps_away"] if shown else None,
                    "winner": res["winner"] if shown else None})
    return out


def view(world: dict) -> dict:
    """Spoiler-free snapshot at the current cursor: what's been revealed, the next day's fixtures
    (the only matches you may watch), standings + bracket SO FAR, provisional Power."""
    ls = world.get("live_replay")
    if not ls:
        return {"active": False}
    c = ls["cursor"]
    n = ls["n_days"]
    playoff_started = c >= ls["n_regular_days"]
    next_fixtures = [_fixture(world, s) for s in ls["days"][c]["series"]] if c < n else []
    just_played = [_result(world, s) for s in ls["days"][c - 1]["series"]] if c > 0 else []
    return {
        "active": c < n,
        "season": ls["season"],
        "phase": ls["days"][c]["phase"] if c < n else "done",
        "day": min(c + 1, n), "cursor": c, "total_days": n, "n_regular_days": ls["n_regular_days"],
        "next_fixtures": next_fixtures,
        "just_played": just_played,
        "standings": _standings_so_far(world, ls, c),
        "bracket": {
            "upper": _bracket_so_far(world, ls["playoff"]["upper"], c) if playoff_started else [],
            "lower": _bracket_so_far(world, ls["playoff"]["lower"], c) if playoff_started else [],
        },
        "done": _done_summary(world, ls) if c >= n else None,
    }


def _done_summary(world: dict, ls: dict) -> dict:
    fin = ls["finalize"]
    return {"season": ls["season"],
            "upper_champion": _team(world, fin["up_play"]["champion"])["abbrev"],
            "lower_champion": _team(world, fin["lo_play"]["champion"])["abbrev"]}


def reveal_next_day(world: dict) -> dict:
    """Reveal the next day (instant — already played). Revealing the LAST day settles the season
    through the shared finaliser (real Elo + calendar + economy + promotion + history)."""
    ls = world.get("live_replay")
    if not ls:
        raise ValueError("no live season to reveal")
    if ls["cursor"] >= ls["n_days"]:
        return {"already_done": True, "view": view(world)}
    ls["cursor"] += 1
    revealed_day = ls["cursor"]
    finished = ls["cursor"] >= ls["n_days"]
    if finished:
        _finalize(world, ls)
    return {"day": revealed_day, "finished": finished, "view": view(world)}


def _finalize(world: dict, ls: dict) -> dict:
    fin = ls["finalize"]
    up_rows = brackets.standings(ls["groups"]["upper"], fin["up_results"])
    a_rows = brackets.standings(ls["groups"]["lower_A"], fin["a_results"])
    z_rows = brackets.standings(ls["groups"]["lower_Z"], fin["z_results"])
    summary = season.finalize_season(
        world, ls["season"], up_rows, a_rows, z_rows, fin["up_play"], fin["lo_play"],
        fin["up_results"], fin["a_results"], fin["z_results"],
        settle_economy=True, record=True, apply_elo=True)
    world["phase"] = "offseason_due"        # whole season revealed -> off-season is now due
    return summary


def reveal_watch(world: dict, home: str, away: str) -> dict:
    """Replay one of the NEXT day's matches map-by-map (no scrub). The result is already decided;
    this only returns its stored map sequence for the broadcast. Does not advance the day."""
    ls = world.get("live_replay")
    if not ls or ls["cursor"] >= ls["n_days"]:
        raise ValueError("nothing to watch")
    for s in ls["days"][ls["cursor"]]["series"]:
        if s["home"] == home and s["away"] == away:
            return _result(world, s, seq=True)
    raise ValueError("that match is not on the next day to be revealed")
