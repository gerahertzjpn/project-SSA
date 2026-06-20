"""Day-by-day season engine — play a season ONE DAY AT A TIME, live.

The all-at-once :func:`season.run_regular_season` plays a whole season in one call. This
engine instead reveals it progressively, the way the owner asked for:

  * the regular season is a sequence of round-robin ROUNDS (one per day); each team plays at
    most once per day, across upper + both lower groups simultaneously;
  * the playoffs are a small bracket DAG per league — each day plays every series whose inputs
    are already known (no peeking at matches that haven't happened);
  * Elo + the calendar update series-by-series, so Power, standings, the H2H matrix and the
    bracket all fill in AS the tournament progresses — there is no future information;
  * a single series can be watched live (map-by-map, no scrub) via :func:`play_series_live`,
    and the result it produces is the OFFICIAL result for that fixture.

State lives on ``world["live_season"]`` (plain JSON — serialisable across requests). When the
last grand final is played the season is finalised through the SAME
:func:`season.finalize_season` the batch path uses, so prizes / promotion / history are identical.
"""
from __future__ import annotations

import random
from typing import Optional

from src.core import rng as seeds
from src.world import brackets, calendar as cal, season, team_elo
from src.world.world import upper_teams

# Playoff brackets as small DAGs: each node's home/away is either a concrete team id (a string)
# or a reference {"w"|"l": node_key} resolved from an earlier node's result.
PLAYOFF_BEST_OF = brackets.PLAYOFF_BEST_OF


# --------------------------------------------------------------------------- #
# Round-robin scheduling (circle method) — split a group into balanced rounds
# --------------------------------------------------------------------------- #
def schedule_rounds(team_ids: list[str]) -> list[list[tuple[str, str]]]:
    """Split a single round-robin into rounds where each team plays at most once per round.
    ``n-1`` rounds for an even count (a bye is added + dropped for odd counts)."""
    ids: list[Optional[str]] = list(team_ids)
    if len(ids) % 2:
        ids.append(None)                     # phantom 'bye' opponent
    n = len(ids)
    rounds = []
    arr = ids[:]
    for _ in range(n - 1):
        pairs = [(arr[i], arr[n - 1 - i]) for i in range(n // 2)
                 if arr[i] is not None and arr[n - 1 - i] is not None]
        rounds.append(pairs)
        arr = [arr[0]] + [arr[-1]] + arr[1:-1]   # rotate, fixing index 0
    return rounds


# --------------------------------------------------------------------------- #
# Bracket templates
# --------------------------------------------------------------------------- #
def _upper_bracket(seeds_: list[str]) -> dict:
    return {
        "order": ["ub", "lb", "lbf", "gf"],
        "nodes": {
            "ub":  {"stage": "ub", "home": seeds_[0], "away": seeds_[1], "best_of": PLAYOFF_BEST_OF},
            "lb":  {"stage": "lb", "home": seeds_[2], "away": seeds_[3], "best_of": PLAYOFF_BEST_OF},
            "lbf": {"stage": "lb_final", "home": {"w": "lb"}, "away": {"l": "ub"},
                    "best_of": PLAYOFF_BEST_OF},
            "gf":  {"stage": "grand_final", "home": {"w": "ub"}, "away": {"w": "lbf"},
                    "best_of": PLAYOFF_BEST_OF},
        },
        "champion_of": "gf", "finalists_of": ["ub", "lbf"],
    }


def _lower_bracket(a: list[str], z: list[str]) -> dict:
    return {
        "order": ["ub", "lb1a", "lb1b", "lb2", "lbf", "gf"],
        "nodes": {
            "ub":   {"stage": "ub", "home": a[0], "away": z[0], "best_of": PLAYOFF_BEST_OF},
            "lb1a": {"stage": "lb_r1", "home": a[1], "away": z[2], "best_of": PLAYOFF_BEST_OF},
            "lb1b": {"stage": "lb_r1", "home": z[1], "away": a[2], "best_of": PLAYOFF_BEST_OF},
            "lb2":  {"stage": "lb_r2", "home": {"w": "lb1a"}, "away": {"w": "lb1b"},
                     "best_of": PLAYOFF_BEST_OF},
            "lbf":  {"stage": "lb_final", "home": {"w": "lb2"}, "away": {"l": "ub"},
                     "best_of": PLAYOFF_BEST_OF},
            "gf":   {"stage": "grand_final", "home": {"w": "ub"}, "away": {"w": "lbf"},
                     "best_of": PLAYOFF_BEST_OF},
        },
        "champion_of": "gf", "finalists_of": ["ub", "lbf"],
    }


def _resolve(ref, nodes: dict) -> Optional[str]:
    """Concrete team id for a node endpoint, or None if its source isn't decided yet."""
    if isinstance(ref, str):
        return ref
    node = nodes.get(ref.get("w") or ref.get("l"))
    res = node and node.get("result")
    if not res:
        return None
    if "w" in ref:
        return res["winner"]
    return res["home"] if res["winner"] == res["away"] else res["away"]   # the loser


def _ready_nodes(bracket: dict) -> list[str]:
    """Node keys whose both endpoints are known and which haven't been played yet."""
    nodes = bracket["nodes"]
    ready = []
    for key in bracket["order"]:
        node = nodes[key]
        if node.get("result"):
            continue
        if _resolve(node["home"], nodes) and _resolve(node["away"], nodes):
            ready.append(key)
    return ready


def _bracket_done(bracket: dict) -> bool:
    return all(bracket["nodes"][k].get("result") for k in bracket["order"])


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
def is_live(world: dict) -> bool:
    ls = world.get("live_season")
    return bool(ls) and ls.get("phase") != "done"


def start(world: dict, *, seed: Optional[int] = None) -> dict:
    """Begin a day-by-day season: precompute the regular-season rounds (no results yet)
    and set the engine to day 1, regular phase. Stored on ``world['live_season']``."""
    up_ids = [t["team_id"] for t in upper_teams(world)]
    a_ids = season._group_members(world, "A")
    z_ids = season._group_members(world, "Z")
    rounds = {
        "upper": [[list(p) for p in rnd] for rnd in schedule_rounds(up_ids)],
        "lower_A": [[list(p) for p in rnd] for rnd in schedule_rounds(a_ids)],
        "lower_Z": [[list(p) for p in rnd] for rnd in schedule_rounds(z_ids)],
    }
    n_rounds = max(len(rounds["upper"]), len(rounds["lower_A"]), len(rounds["lower_Z"]))
    ls = {
        "season": world["season"],
        "seed": seeds.resolve_seed(seed),
        "phase": "regular",
        "day": 1,
        "n_regular_days": n_rounds,
        "rounds": rounds,
        "cursor": 0,                         # next regular round index to play
        "results": {"upper": [], "lower_A": [], "lower_Z": []},
        "playoff": None,
        "groups": {"upper": up_ids, "lower_A": a_ids, "lower_Z": z_ids},
        "done": None,
    }
    world["live_season"] = ls
    world["phase"] = "live"                   # header shows the day-by-day controls
    return ls


def _play_fn(world: dict, seed_salt: int):
    rng = random.Random(seeds.resolve_seed(None) ^ (seed_salt * 2654435761 & 0xFFFFFFFF))

    def play(home: str, away: str, best_of: int) -> dict:
        # Fresh entropy per series (a live match is played once — its result is official).
        return season.play_one_series(world, home, away, best_of, rng)
    return play


def _record(world: dict, season_num: int, league: str, series: dict) -> None:
    pb = series.get("playbook") or {}
    cal.add_match(world, season=season_num, stage=series.get("stage", "league"), league=league,
                  home=series["home"], away=series["away"], best_of=series.get("best_of", 3),
                  maps_home=series["maps_home"], maps_away=series["maps_away"],
                  winner=series["winner"],
                  home_strategy=(pb.get("home") or {}).get("strategy"),
                  away_strategy=(pb.get("away") or {}).get("strategy"))
    team_elo.apply_series_result(world, series["home"], series["away"],
                                 series["maps_home"], series["maps_away"], series.get("best_of", 3))


def _league_of(world: dict, group_key: str) -> str:
    return "upper" if group_key == "upper" else "lower"


def play_day(world: dict, *, play=None) -> dict:
    """Play exactly ONE day and return what happened. Regular days play a round-robin round
    across all groups; playoff days play every currently-decidable series. The final day
    finalises the season. Raises if no live season is in progress."""
    ls = world.get("live_season")
    if not ls or ls.get("phase") == "done":
        raise ValueError("no live season in progress")
    play = play or _play_fn(world, ls["day"])
    if ls["phase"] == "regular":
        return _play_regular_day(world, ls, play)
    return _play_playoff_day(world, ls, play)


def _played_pairs(results: list[dict]) -> set:
    """Unordered {home, away} pairs already recorded — a fixture watched live must not replay."""
    return {frozenset((s["home"], s["away"])) for s in results}


def _play_regular_day(world: dict, ls: dict, play) -> dict:
    rnd = ls["cursor"]
    played = []
    for gkey in ("upper", "lower_A", "lower_Z"):
        rounds = ls["rounds"][gkey]
        if rnd >= len(rounds):
            continue
        done_pairs = _played_pairs(ls["results"][gkey])
        for home, away in rounds[rnd]:
            if frozenset((home, away)) in done_pairs:
                continue                         # already watched live this round — don't replay
            series = play(home, away, brackets.LEAGUE_BEST_OF)
            series = dict(series, stage="league")
            ls["results"][gkey].append(series)
            _record(world, ls["season"], _league_of(world, gkey), series)
            played.append({"group": gkey, **_series_public(world, series)})
    ls["cursor"] += 1
    day = ls["day"]
    ls["day"] += 1
    if ls["cursor"] >= ls["n_regular_days"]:
        _seed_playoffs(world, ls)
    return {"day": day, "phase": "regular", "played": played, "next_phase": ls["phase"]}


def _seed_playoffs(world: dict, ls: dict) -> None:
    up_rows = brackets.standings(ls["groups"]["upper"], ls["results"]["upper"])
    a_rows = brackets.standings(ls["groups"]["lower_A"], ls["results"]["lower_A"])
    z_rows = brackets.standings(ls["groups"]["lower_Z"], ls["results"]["lower_Z"])
    ls["playoff"] = {
        "upper": _upper_bracket([r["team_id"] for r in up_rows]),
        "lower": _lower_bracket([r["team_id"] for r in a_rows], [r["team_id"] for r in z_rows]),
    }
    ls["phase"] = "playoff"


def _play_playoff_day(world: dict, ls: dict, play) -> dict:
    played = []
    for league_key, bracket in (("upper", ls["playoff"]["upper"]), ("lower", ls["playoff"]["lower"])):
        nodes = bracket["nodes"]
        for key in _ready_nodes(bracket):
            node = nodes[key]
            home = _resolve(node["home"], nodes)
            away = _resolve(node["away"], nodes)
            series = play(home, away, node.get("best_of", PLAYOFF_BEST_OF))
            series = dict(series, stage=node["stage"])
            node["result"] = series
            _record(world, ls["season"], league_key, series)
            played.append({"league": league_key, "stage": node["stage"],
                           **_series_public(world, series)})
    day = ls["day"]
    ls["day"] += 1
    if _bracket_done(ls["playoff"]["upper"]) and _bracket_done(ls["playoff"]["lower"]):
        summary = _finalize(world, ls)
        return {"day": day, "phase": "playoff", "played": played,
                "next_phase": "done", "summary": summary}
    return {"day": day, "phase": "playoff", "played": played, "next_phase": ls["phase"]}


def _bracket_series(bracket: dict) -> list[dict]:
    return [bracket["nodes"][k]["result"] for k in bracket["order"] if bracket["nodes"][k].get("result")]


def _finalize(world: dict, ls: dict) -> dict:
    """All grand finals are in — settle the season through the shared finaliser (Elo + calendar
    were already applied per-day, so skip them here)."""
    up = ls["playoff"]["upper"]
    lo = ls["playoff"]["lower"]
    up_nodes, lo_nodes = up["nodes"], lo["nodes"]
    up_rows = brackets.standings(ls["groups"]["upper"], ls["results"]["upper"])
    a_rows = brackets.standings(ls["groups"]["lower_A"], ls["results"]["lower_A"])
    z_rows = brackets.standings(ls["groups"]["lower_Z"], ls["results"]["lower_Z"])
    up_champ = up_nodes["gf"]["result"]["winner"]
    up_finalists = [_resolve({"w": k}, up_nodes) for k in up["finalists_of"]]
    lo_champ = lo_nodes["gf"]["result"]["winner"]
    lo_finalists = [_resolve({"w": k}, lo_nodes) for k in lo["finalists_of"]]
    up_play = {"champion": up_champ, "grand_finalists": up_finalists,
               "series": _bracket_series(up)}
    lo_play = {"champion": lo_champ, "promoted": lo_finalists, "grand_finalists": lo_finalists,
               "series": _bracket_series(lo)}
    summary = season.finalize_season(
        world, ls["season"], up_rows, a_rows, z_rows, up_play, lo_play,
        ls["results"]["upper"], ls["results"]["lower_A"], ls["results"]["lower_Z"],
        settle_economy=True, record=False, apply_elo=False, seed=ls["seed"])
    ls["phase"] = "done"
    ls["done"] = {"season": summary["season"], "upper_champion": up_champ,
                  "lower_champion": lo_champ}
    world["phase"] = "offseason_due"          # season played -> off-season is now due
    return summary


# --------------------------------------------------------------------------- #
# Spoiler-free view for the UI
# --------------------------------------------------------------------------- #
def _team(world: dict, tid: Optional[str]) -> dict:
    t = world["teams"].get(tid) if tid else None
    return {"team_id": tid, "abbrev": t["abbrev"] if t else tid, "name": t["name"] if t else tid}


def _series_public(world: dict, series: dict) -> dict:
    return {"home": series["home"], "away": series["away"],
            "home_abbrev": _team(world, series["home"])["abbrev"],
            "away_abbrev": _team(world, series["away"])["abbrev"],
            "maps_home": series["maps_home"], "maps_away": series["maps_away"],
            "winner": series["winner"], "best_of": series.get("best_of"),
            "sequence": series.get("sequence", [])}


def _next_day_fixtures(world: dict, ls: dict) -> list[dict]:
    """The matches that WILL be played next (concrete teams) — never any further ahead."""
    out = []
    if ls["phase"] == "regular":
        rnd = ls["cursor"]
        for gkey in ("upper", "lower_A", "lower_Z"):
            rounds = ls["rounds"][gkey]
            if rnd < len(rounds):
                for home, away in rounds[rnd]:
                    out.append({"group": gkey, "stage": "league", "best_of": brackets.LEAGUE_BEST_OF,
                                "home": home, "away": away,
                                "home_abbrev": _team(world, home)["abbrev"],
                                "away_abbrev": _team(world, away)["abbrev"]})
    elif ls["phase"] == "playoff":
        for league_key, bracket in (("upper", ls["playoff"]["upper"]),
                                    ("lower", ls["playoff"]["lower"])):
            nodes = bracket["nodes"]
            for key in _ready_nodes(bracket):
                node = nodes[key]
                home = _resolve(node["home"], nodes)
                away = _resolve(node["away"], nodes)
                out.append({"league": league_key, "stage": node["stage"],
                            "best_of": node.get("best_of"), "home": home, "away": away,
                            "home_abbrev": _team(world, home)["abbrev"],
                            "away_abbrev": _team(world, away)["abbrev"]})
    return out


def _bracket_public(world: dict, bracket: Optional[dict]) -> list[dict]:
    """The bracket SO FAR — resolved slots show teams/scores, undecided slots stay blank."""
    if not bracket:
        return []
    nodes = bracket["nodes"]
    out = []
    for key in bracket["order"]:
        node = nodes[key]
        home = _resolve(node["home"], nodes)
        away = _resolve(node["away"], nodes)
        res = node.get("result")
        out.append({"key": key, "stage": node["stage"],
                    "home": _team(world, home) if home else None,
                    "away": _team(world, away) if away else None,
                    "played": bool(res),
                    "maps_home": res["maps_home"] if res else None,
                    "maps_away": res["maps_away"] if res else None,
                    "winner": res["winner"] if res else None})
    return out


def view(world: dict) -> dict:
    """Spoiler-free snapshot for the UI: progress, the NEXT day's fixtures (the only thing you
    may watch live), the bracket resolved so far, and the live group standings."""
    ls = world.get("live_season")
    if not ls:
        return {"active": False}
    total = ls["n_regular_days"] + 4          # +4 playoff days (lower bracket is the longest)
    standings = {key: brackets.standings(ls["groups"][key], ls["results"][key])
                 for key in ("upper", "lower_A", "lower_Z")}
    public_standings = {key: [_standing_row(world, r) for r in rows]
                        for key, rows in standings.items()}
    return {
        "active": ls.get("phase") != "done",
        "season": ls["season"], "phase": ls["phase"],
        "day": ls["day"], "total_days": total, "n_regular_days": ls["n_regular_days"],
        "next_fixtures": _next_day_fixtures(world, ls) if ls.get("phase") != "done" else [],
        "standings": public_standings,
        "bracket": {"upper": _bracket_public(world, (ls.get("playoff") or {}).get("upper")),
                    "lower": _bracket_public(world, (ls.get("playoff") or {}).get("lower"))},
        "done": ls.get("done"),
    }


def _standing_row(world: dict, r: dict) -> dict:
    t = world["teams"].get(r["team_id"], {})
    # cores_destroyed is deliberately omitted — hidden tiebreaker, never shown.
    return {"team_id": r["team_id"], "abbrev": t.get("abbrev", r["team_id"]),
            "name": t.get("name", r["team_id"]), "place": r["place"], "points": r["points"],
            "series_wins": r["series_wins"], "series_losses": r["series_losses"],
            "map_diff": r["map_diff"]}


def play_series_live(world: dict, home: str, away: str) -> dict:
    """Play ONE upcoming fixture as the official result (used by the live-broadcast watch). The
    fixture must be the/an immediate next match; returns the full map sequence for a no-scrub
    reveal. Marks it played in the engine + records Elo/calendar, exactly like ``play_day`` would."""
    ls = world.get("live_season")
    if not ls or ls.get("phase") == "done":
        raise ValueError("no live season in progress")
    nxt = {(f["home"], f["away"]): f for f in _next_day_fixtures(world, ls)}
    fx = nxt.get((home, away))
    if fx is None:
        raise ValueError("that fixture is not the next one to be played")
    play = _play_fn(world, ls["day"] * 1000 + 7)
    series = dict(play(home, away, fx["best_of"]), stage=fx["stage"])
    if ls["phase"] == "regular":
        gkey = fx["group"]
        ls["results"][gkey].append(series)
        _record(world, ls["season"], _league_of(world, gkey), series)
    else:
        bracket = ls["playoff"][fx["league"]]
        nodes = bracket["nodes"]
        for key in _ready_nodes(bracket):
            n = nodes[key]
            if _resolve(n["home"], nodes) == home and _resolve(n["away"], nodes) == away:
                n["result"] = series
                break
        _record(world, ls["season"], fx["league"], series)
    return _series_public(world, series)
