"""Play a whole season ONCE, then reveal it DAY BY DAY (no spoilers).

The owner's model (chosen explicitly): the season is a SINGLE canonical play-through with the
REAL match engine (the SAME `season.play_one_series` / net engine «Сыграть сезон» uses), captured
into a per-day structure. The UI then *reveals* it one day at a time, and only what's been revealed
exists for the viewer: standings, the playoff bracket and provisional Power fill in as you step
through the days; nothing past the cursor is shown. Watching a match in the broadcast just replays
its already-decided map sequence (no scrub).

So this does NOT change the match mechanic — it is one season, played up front, revealed
progressively, and settled through the same finaliser. (It is NOT a bit-identical re-run of a
specific batch `run_regular_season`: per-series RNG is keyed by call order and the two paths play
the groups in a different order, so a given seed yields a different — but equally valid — canonical
season. The promotion/points/series-count invariants hold either way.)

Spoiler-safety: ``play_and_store`` plays every series but does NOT touch canonical state (no Elo,
no calendar, no history, no promotion) — those other tabs stay blank for the season until the
reveal finishes. Provisional per-day Power is computed on a LOCAL Elo copy. When the last day is
revealed, the season is settled through the SAME :func:`season.finalize_season` the batch path uses
(Elo + calendar + economy + promotion + history), so the canonical outcome is identical.
"""
from __future__ import annotations

import copy
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
    # The up-front play records per-game position_stats (winrate per position). Snapshot them now,
    # restore after playing, and stash the post-season state — so during the reveal the Ratings tab
    # shows ONLY pre-season position data (no leak), and the real stats land at finalize.
    pos_before = {pid: copy.deepcopy(p.get("position_stats"))
                  for pid, p in world["players"].items() if "position_stats" in p}

    def _bump_elo(s: dict) -> None:                 # provisional Power, on a LOCAL copy only
        d = series_elo_delta(local_elo[s["home"]], local_elo[s["away"]],
                             s["maps_home"], s["maps_away"], s.get("best_of", 3))
        local_elo[s["home"]] += d
        local_elo[s["away"]] -= d

    days = []        # ONE series per day (owner: «1 день — 1 игра»)
    power_by_day = [{tid: round(e) for tid, e in local_elo.items()}]   # day 0 = pre-season

    def _add_day(phase: str, s: dict) -> None:
        s["day"] = len(days) + 1
        _bump_elo(s)
        days.append({"phase": phase, "day": s["day"], "series": s})
        power_by_day.append({tid: round(e) for tid, e in local_elo.items()})

    # ---- regular season: ONE game per day (round-major, then group, then pairing) ---- #
    for r in range(n_regular):
        for gkey in ("upper", "lower_A", "lower_Z"):
            grounds = rounds[gkey]
            if r >= len(grounds):
                continue
            for home, away in grounds[r]:
                s = dict(play(home, away, brackets.LEAGUE_BEST_OF), stage="league",
                         group=gkey, league=_league_of(gkey))
                results[gkey].append(s)
                _add_day("regular", s)
    n_regular_days = len(days)

    # ---- seed playoffs from the (now complete) group standings ---------- #
    up_rows = brackets.standings(up_ids, results["upper"])
    a_rows = brackets.standings(a_ids, results["lower_A"])
    z_rows = brackets.standings(z_ids, results["lower_Z"])
    up_br = _upper_bracket([r["team_id"] for r in up_rows])
    lo_br = _lower_bracket([r["team_id"] for r in a_rows], [r["team_id"] for r in z_rows])

    # ---- playoffs: ONE decidable series per day (a wave of ready nodes spreads over days) ---- #
    while not (_bracket_done(up_br) and _bracket_done(lo_br)):
        ready = [(lg, key, br) for lg, br in (("upper", up_br), ("lower", lo_br))
                 for key in _ready_nodes(br)]
        if not ready:
            break
        for lg, key, br in ready:
            node = br["nodes"][key]
            home = _resolve(node["home"], br["nodes"])
            away = _resolve(node["away"], br["nodes"])
            s = dict(play(home, away, node["best_of"]), stage=node["stage"], league=lg)
            node["result"] = s
            _add_day("playoff", s)
            node["day"] = s["day"]

    n_days = len(days)
    # capture the post-season position_stats, then restore the pre-season ones (re-applied at finalize)
    pos_after = {pid: copy.deepcopy(world["players"][pid].get("position_stats")) for pid in pos_before}
    for pid, snap in pos_before.items():
        world["players"][pid]["position_stats"] = snap
    world["live_replay"] = {
        "season": season_num,
        "cursor": 0,
        "n_days": n_days,
        "n_regular_days": n_regular_days,          # number of regular GAMES (1 game = 1 day)
        "groups": {"upper": up_ids, "lower_A": a_ids, "lower_Z": z_ids},
        "days": days,                              # one series per day; full series incl cores (server-side)
        "playoff": {"upper": up_br, "lower": lo_br},
        "power_by_day": power_by_day,
        "pos_after": pos_after,                    # real position_stats, applied when reveal finishes
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


def is_revealing(world: dict) -> bool:
    """A day-by-day season exists (being revealed, or fully revealed but not yet off-season'd).
    While true, the read-models must show the season SO FAR, not last season's finished data."""
    return bool(world.get("live_replay"))


def revealed_power(world: dict) -> Optional[dict]:
    """Provisional per-team Power at the revealed day (cheap — no standings recompute), or None.
    Only while the reveal is in progress — once the last day is revealed the season is finalised, so
    the read-models switch back to canonical Elo/standings (history), not the provisional snapshot."""
    ls = world.get("live_replay")
    if not ls or ls.get("cursor", 0) >= ls.get("n_days", 0):
        return None
    pbd = ls["power_by_day"]
    return pbd[min(ls["cursor"], len(pbd) - 1)]


def revealed(world: dict) -> Optional[dict]:
    """Raw revealed-so-far data the read-models render the CURRENT season from (everything up to the
    reveal cursor — nothing past it): group standings rows, the revealed regular + playoff series,
    the bracket, and provisional per-team Power. None when no day-by-day season is active.

    This is what makes EVERY tab day-accurate during a reveal: at cursor 0 nobody has points; each
    revealed day adds its results. cores_destroyed stays internal (callers must not surface it).
    Returns None once the reveal is complete (the season is finalised — tabs use canonical data)."""
    ls = world.get("live_replay")
    if not ls or ls.get("cursor", 0) >= ls.get("n_days", 0):
        return None
    c = ls["cursor"]
    reg = {"upper": [], "lower_A": [], "lower_Z": []}
    playoff_series = []
    for day in ls["days"][:c]:
        s = day["series"]
        if day["phase"] == "regular":
            reg[s["group"]].append(s)
        else:
            playoff_series.append(s)
    # the FULL regular fixture list is known up front (the schedule shows who-plays-when from day 0)
    all_regular = [d["series"] for d in ls["days"] if d["phase"] == "regular"]
    standings = {g: brackets.standings(ls["groups"][g], reg[g]) for g in reg}
    power = ls["power_by_day"][min(c, len(ls["power_by_day"]) - 1)]
    return {"cursor": c, "season": ls["season"], "groups": ls["groups"],
            "n_regular_days": ls["n_regular_days"], "regular": reg, "all_regular": all_regular,
            "playoff_series": playoff_series, "standings": standings, "power": power,
            "playoff": ls["playoff"]}


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
        if day["phase"] == "regular":
            revealed[day["series"]["group"]].append(day["series"])
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


def _match_card(world: dict, ls: dict, cursor: int) -> dict:
    """Rich card for the NEXT match (the LIVE tab's centerpiece): both teams with their provisional
    Power, current place/points/record, and recent form — all from what's been revealed so far."""
    s = ls["days"][cursor]["series"]
    power = ls["power_by_day"][min(cursor, len(ls["power_by_day"]) - 1)]
    standings = _standings_so_far(world, ls, cursor)
    rows = {r["team_id"]: r for g in standings.values() for r in g}

    def info(tid):
        row = rows.get(tid, {})
        form = []
        for d in ls["days"][:cursor]:
            ser = d["series"]
            if tid in (ser["home"], ser["away"]):
                form.append("W" if ser["winner"] == tid else "L")
        t = world["teams"].get(tid, {})
        return {"team_id": tid, "abbrev": t.get("abbrev", tid), "name": t.get("name", tid),
                "power": power.get(tid, round(float(t.get("elo", 1000)))),
                "place": row.get("place"), "points": row.get("points", 0),
                "series_wins": row.get("series_wins", 0), "series_losses": row.get("series_losses", 0),
                "map_diff": row.get("map_diff", 0), "form": form[-6:]}

    return {"stage": s["stage"], "stage_label": STAGE_LABEL_LIVE.get(s["stage"], s["stage"]),
            "best_of": s.get("best_of"), "league": s.get("league"), "group": s.get("group"),
            "day": cursor + 1, "home": info(s["home"]), "away": info(s["away"])}


STAGE_LABEL_LIVE = {"league": "Регулярный сезон", "ub": "Плей-офф · Верхняя сетка",
                    "lb": "Плей-офф · Нижняя сетка", "lb_r1": "Плей-офф · Нижняя сетка, раунд 1",
                    "lb_r2": "Плей-офф · Нижняя сетка, раунд 2", "lb_final": "Финал нижней сетки",
                    "grand_final": "ГРАНД-ФИНАЛ"}


def view(world: dict) -> dict:
    """Spoiler-free snapshot at the current cursor: what's been revealed, the next day's fixtures
    (the only matches you may watch), standings + bracket SO FAR, provisional Power."""
    ls = world.get("live_replay")
    if not ls:
        return {"active": False}
    c = ls["cursor"]
    n = ls["n_days"]
    playoff_started = c >= ls["n_regular_days"]
    next_fixtures = [_fixture(world, ls["days"][c]["series"])] if c < n else []
    just_played = [_result(world, ls["days"][c - 1]["series"])] if c > 0 else []
    return {
        "active": c < n,
        "season": ls["season"],
        "phase": ls["days"][c]["phase"] if c < n else "done",
        "day": min(c + 1, n), "cursor": c, "total_days": n, "n_regular_days": ls["n_regular_days"],
        "next_fixtures": next_fixtures,
        "just_played": just_played,
        "match": _match_card(world, ls, c) if c < n else None,
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
    for pid, stats in (ls.get("pos_after") or {}).items():   # the season's real position_stats land now
        if pid in world["players"] and stats is not None:
            world["players"][pid]["position_stats"] = stats
    world["phase"] = "offseason_due"        # whole season revealed -> off-season is now due
    return summary


def reveal_watch(world: dict, home: str, away: str) -> dict:
    """Replay one of the NEXT day's matches map-by-map (no scrub). The result is already decided;
    this only returns its stored map sequence for the broadcast. Does not advance the day."""
    ls = world.get("live_replay")
    if not ls or ls["cursor"] >= ls["n_days"]:
        raise ValueError("nothing to watch")
    s = ls["days"][ls["cursor"]]["series"]
    if s["home"] == home and s["away"] == away:
        return _result(world, s, seq=True)
    raise ValueError("that match is not on the next day to be revealed")
