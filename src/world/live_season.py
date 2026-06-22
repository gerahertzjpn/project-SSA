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
import math
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
REGULAR_GAMES_PER_DAY = 2     # owner: 2 regular matches per day; playoffs stay 1 per day


# --------------------------------------------------------------------------- #
# Win probability — MULTI-FACTOR, computed ONCE before a match is played (frozen)
# --------------------------------------------------------------------------- #
def _roster_soloq(world: dict, tid: str) -> float:
    """Avg SoloQ of a club's best-3 (its likely starters) — the on-paper roster strength."""
    t = world["teams"].get(tid, {})
    rs = sorted((world["players"][pid]["soloq_rating"] for pid in t.get("roster", [])
                 if pid in world["players"]), reverse=True)[:3]
    return sum(rs) / len(rs) if rs else 1000.0


def _form_score(played: list, tid: str, n: int = 4) -> float:
    """Recent form of a club in [-0.5, +0.5] from its last ``n`` results so far."""
    res = [1 if s.get("winner") == tid else 0 for s in played if tid in (s.get("home"), s.get("away"))][-n:]
    return (sum(res) / len(res) - 0.5) if res else 0.0


def _series_prob(pmap: float, best_of: int) -> float:
    """P(home wins a Bo-N series) given a per-map win prob (race to ceil(N/2) maps)."""
    need = best_of // 2 + 1
    ph = sum(math.comb(need - 1 + j, j) * pmap ** need * (1 - pmap) ** j for j in range(need))
    pa = sum(math.comb(need - 1 + j, j) * (1 - pmap) ** need * pmap ** j for j in range(need))
    return ph / (ph + pa) if (ph + pa) > 0 else 0.5


def win_probability(world: dict, base_elo: dict, played: list, home: str, away: str, best_of: int):
    """HOME-win % for a series, from the state BEFORE it is played — blends five signals into an
    effective Elo gap, then to a Bo-N series probability:
      • Power (chess-Elo, provisional from base + games played so far)
      • roster SoloQ (best-3 strength on paper)
      • head-to-head (prior meetings this season, capped)
      • recent form (last 4 results)
      • standing momentum (total wins so far)
    Returns ``(pct, factors)``; the pct is stored on the series and never recomputed (frozen)."""
    elo = dict(base_elo)
    for s in played:                                     # provisional Power from games already played
        if s.get("maps_home") == s.get("maps_away"):
            continue
        d = series_elo_delta(elo.get(s["home"], 1000.0), elo.get(s["away"], 1000.0),
                             s["maps_home"], s["maps_away"], s.get("best_of", 3))
        elo[s["home"]] = elo.get(s["home"], 1000.0) + d
        elo[s["away"]] = elo.get(s["away"], 1000.0) - d
    power = elo.get(home, 1000.0) - elo.get(away, 1000.0)
    roster = 0.35 * (_roster_soloq(world, home) - _roster_soloq(world, away))
    h2h_raw = sum(1 if s.get("winner") == home else -1
                  for s in played if {s.get("home"), s.get("away")} == {home, away})
    h2h = 25.0 * max(-3, min(3, h2h_raw))
    form = 80.0 * (_form_score(played, home) - _form_score(played, away))
    wins_h = sum(1 for s in played if s.get("winner") == home)
    wins_a = sum(1 for s in played if s.get("winner") == away)
    standing = 10.0 * (wins_h - wins_a)
    diff = power + roster + h2h + form + standing
    pmap = 1.0 / (1.0 + 10 ** (-diff / 400.0))
    pct = max(1, min(99, round(100 * _series_prob(pmap, best_of))))
    factors = {"power": round(power), "roster": round(roster), "h2h": round(h2h),
               "form": round(form), "standing": round(standing), "total": round(diff)}
    return pct, factors


def play_and_store(world: dict, *, play=None, seed: Optional[int] = None,
                   series_player=None, workers: int = 1) -> dict:
    """Play the ENTIRE season (real engine via ``play``) and store it for progressive reveal on
    ``world['live_replay']``. Each game gets a unique ``gid``; reveal is PER GAME (revealed_gids),
    so a single match can be watched and committed on its own. Regular days hold 2 games, playoff
    days 1 (upper bracket fully, then lower). Canonical state is untouched until the reveal ends.

    Speed (safe): the REGULAR season is a set of INDEPENDENT round-robin series. When a picklable
    ``series_player`` (NetSeriesPlayer) is supplied it is played with PRE-ASSIGNED per-series seeds —
    over a process pool when ``workers > 1`` — then folded back in canonical order (win prob + position
    credit + gid). Because each match re-seeds torch from its own seed, the result is BIT-IDENTICAL to
    sequential regardless of worker count. PLAYOFFS are a dependency DAG and stay strictly sequential.
    Without a ``series_player`` (rating/abstract worlds, tests) everything runs sequentially via
    ``play`` exactly as before."""
    from src.core.rng import Spawner, resolve_seed
    from src.world import net_match, parallel

    season_num = world["season"]
    if series_player is None:
        play = play or season.make_play_fn(world, seed=seed)
    # independent seed streams so regular & playoff series never share entropy
    _master = Spawner(resolve_seed(seed))
    reg_spawner = Spawner(_master.seed())
    po_spawner = Spawner(_master.seed())
    up_ids = [t["team_id"] for t in upper_teams(world)]
    a_ids = season._group_members(world, "A")
    z_ids = season._group_members(world, "Z")
    rounds = {"upper": schedule_rounds(up_ids), "lower_A": schedule_rounds(a_ids),
              "lower_Z": schedule_rounds(z_ids)}
    n_rounds = max(len(rounds["upper"]), len(rounds["lower_A"]), len(rounds["lower_Z"]))

    results = {"upper": [], "lower_A": [], "lower_Z": []}
    pos_before = {pid: copy.deepcopy(p.get("position_stats"))
                  for pid, p in world["players"].items() if "position_stats" in p}
    _gid = [0]
    base_elo_snap = {tid: round(float(t.get("elo", 1000.0))) for tid, t in world["teams"].items()}
    played: list = []           # running, in play order — the pre-match snapshot for the win prob

    def _tag(s):
        s["gid"] = _gid[0]
        _gid[0] += 1
        s["revealed"] = False
        return s

    def _finish(s, home, away, best_of, **extra):
        # FREEZE the multi-factor win prob from PRIOR results (play order). win_probability ignores
        # THIS series' own outcome, so computing it after the play is identical to before. Then credit
        # per-position winrate (deferred by the net player so workers stay world-read-only), tag, append.
        pct, factors = win_probability(world, base_elo_snap, played, home, away, best_of)
        lineups = s.pop("map_lineups", None)
        if lineups:
            net_match.apply_position_credit(world, lineups)
        s.update(extra)
        s["win_prob"], s["win_prob_factors"] = pct, factors
        _tag(s)
        played.append(s)
        return s

    def _play(home, away, best_of, spawner):
        # one series with an explicit pre-assigned seed: the picklable net player (same code the pool
        # runs), else the sequential `play` (which carries its own seeding).
        if series_player is not None:
            return series_player(home, away, best_of, spawner.seed())
        return play(home, away, best_of)

    # ---- regular season: every round-robin pairing, round-major then group ---- #
    reg_specs = []                       # canonical play order: (group_key, home, away)
    for r in range(n_rounds):
        for gkey in ("upper", "lower_A", "lower_Z"):
            grounds = rounds[gkey]
            if r >= len(grounds):
                continue
            for home, away in grounds[r]:
                reg_specs.append((gkey, home, away))
    bo = brackets.LEAGUE_BEST_OF
    regular_games = []
    if series_player is not None:
        # pre-assign every series' seed up front (single-thread, order-independent), play them in
        # parallel, then fold the results back in canonical order so win_prob/standings are identical.
        seeds = [reg_spawner.seed() for _ in reg_specs]
        tasks = [(home, away, bo, sd) for (gkey, home, away), sd in zip(reg_specs, seeds)]
        raw = parallel.run_batch(series_player, tasks, workers)
        for (gkey, home, away), s in zip(reg_specs, raw):
            _finish(s, home, away, bo, stage="league", group=gkey, league=_league_of(gkey))
            results[gkey].append(s)
            regular_games.append(s)
    else:
        for gkey, home, away in reg_specs:
            s = _finish(_play(home, away, bo, reg_spawner), home, away, bo,
                        stage="league", group=gkey, league=_league_of(gkey))
            results[gkey].append(s)
            regular_games.append(s)

    days = []        # each day: {phase, series:[1-2 games]}
    for i in range(0, len(regular_games), REGULAR_GAMES_PER_DAY):
        days.append({"phase": "regular", "series": regular_games[i:i + REGULAR_GAMES_PER_DAY]})
    n_regular_days = len(days)

    # ---- seed playoffs from the (now complete) group standings ---------- #
    up_rows = brackets.standings(up_ids, results["upper"])
    a_rows = brackets.standings(a_ids, results["lower_A"])
    z_rows = brackets.standings(z_ids, results["lower_Z"])
    up_br = _upper_bracket([r["team_id"] for r in up_rows])
    lo_br = _lower_bracket([r["team_id"] for r in a_rows], [r["team_id"] for r in z_rows])

    # ---- playoffs: ONE series per day; UPPER bracket fully, THEN lower (not interleaved). A
    # dependency DAG (each series feeds the next) -> STRICTLY sequential, never parallelised. ---- #
    def _play_bracket(br, lg):
        while not _bracket_done(br):
            ready = _ready_nodes(br)
            if not ready:
                break
            for key in ready:
                node = br["nodes"][key]
                home = _resolve(node["home"], br["nodes"])
                away = _resolve(node["away"], br["nodes"])
                s = _finish(_play(home, away, node["best_of"], po_spawner), home, away,
                            node["best_of"], stage=node["stage"], league=lg)
                node["result"] = s
                node["gid"] = s["gid"]
                days.append({"phase": "playoff", "series": [s]})
    _play_bracket(up_br, "upper")
    _play_bracket(lo_br, "lower")

    for di, d in enumerate(days, 1):                 # stamp each game with its (display) day number
        for s in d["series"]:
            s["day"] = di
    n_days = len(days)

    pos_after = {pid: copy.deepcopy(world["players"][pid].get("position_stats")) for pid in pos_before}
    for pid, snap in pos_before.items():
        world["players"][pid]["position_stats"] = snap
    world["live_replay"] = {
        "season": season_num,
        "n_days": n_days,
        "n_regular_days": n_regular_days,
        "groups": {"upper": up_ids, "lower_A": a_ids, "lower_Z": z_ids},
        "days": days,                              # full series incl cores (server-side only)
        "revealed_gids": [],                       # which GAMES have been revealed (per-game)
        "shown_day": 0,                            # the day the LIVE tab displays — advances ONLY on
                                                   #   «Следующий день» (watching a game never advances)
        "playoff": {"upper": up_br, "lower": lo_br},
        "base_elo": {tid: round(float(t.get("elo", 1000))) for tid, t in world["teams"].items()},
        "pos_after": pos_after,
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
# Reveal — PER GAME (revealed_gids), spoiler-free
# --------------------------------------------------------------------------- #
STAGE_LABEL_LIVE = {"league": "Регулярный сезон", "ub": "Плей-офф · Верхняя сетка",
                    "lb": "Плей-офф · Нижняя сетка", "lb_r1": "Плей-офф · Нижняя сетка, раунд 1",
                    "lb_r2": "Плей-офф · Нижняя сетка, раунд 2", "lb_final": "Финал нижней сетки",
                    "grand_final": "ГРАНД-ФИНАЛ"}


def _revealed_set(ls: dict) -> set:
    return set(ls.get("revealed_gids", []))


def _league_label(s: dict) -> str:
    if s.get("league") == "upper":
        return "Высшая лига"
    g = s.get("group")
    return "Низшая лига" + (" · группа A" if g == "lower_A" else " · группа Z" if g == "lower_Z" else "")


def _provisional_power(world: dict, ls: dict, revset: set) -> dict:
    """Provisional per-team Power = base (pre-season) Elo + the Elo deltas of every REVEALED game so
    far (in play order). Cheap replay over the stored games — no canonical mutation."""
    elo = {tid: float(e) for tid, e in ls["base_elo"].items()}
    for d in ls["days"]:
        for s in d["series"]:
            if s["gid"] in revset:
                dd = series_elo_delta(elo[s["home"]], elo[s["away"]],
                                      s["maps_home"], s["maps_away"], s.get("best_of", 3))
                elo[s["home"]] += dd
                elo[s["away"]] -= dd
    return {tid: round(e) for tid, e in elo.items()}


def is_live(world: dict) -> bool:
    """The reveal is in progress until the owner advances PAST the last day (which finalises). Note:
    watching every game does NOT end it — the season completes only on «Следующий день» off the last
    day. So this is keyed on ``shown_day``, not on all games being revealed."""
    ls = world.get("live_replay")
    return bool(ls) and ls.get("shown_day", 0) < ls["n_days"]


def revealed_power(world: dict) -> Optional[dict]:
    ls = world.get("live_replay")
    if not ls or not is_live(world):
        return None
    return _provisional_power(world, ls, _revealed_set(ls))


def revealed(world: dict) -> Optional[dict]:
    """Raw revealed-so-far data the read-models render the CURRENT season from. None once the
    reveal is complete (season finalised — tabs use canonical data) or if no live season."""
    ls = world.get("live_replay")
    if not ls or not is_live(world):
        return None
    revset = _revealed_set(ls)
    reg = {"upper": [], "lower_A": [], "lower_Z": []}
    playoff_series = []
    for d in ls["days"]:
        for s in d["series"]:
            if s["gid"] not in revset:
                continue
            if d["phase"] == "regular":
                reg[s["group"]].append(s)
            else:
                playoff_series.append(s)
    all_regular = [s for d in ls["days"] if d["phase"] == "regular" for s in d["series"]]
    regular_done = all(s["gid"] in revset for s in all_regular)
    # per-league completion: the LOWER bracket can seed as soon as the LOWER regular is done, even if the
    # UPPER regular is still playing (and vice-versa) — don't make one league wait on the other (#1).
    def _done(lg):
        return all(s["gid"] in revset for s in all_regular if s.get("league") == lg)
    standings = {g: brackets.standings(ls["groups"][g], reg[g]) for g in reg}
    return {"season": ls["season"], "groups": ls["groups"], "n_regular_days": ls["n_regular_days"],
            "regular": reg, "all_regular": all_regular, "playoff_series": playoff_series,
            "standings": standings, "power": _provisional_power(world, ls, revset),
            "playoff": ls["playoff"], "revealed_gids": list(revset), "regular_done": regular_done,
            "regular_done_by_league": {"upper": _done("upper"), "lower": _done("lower")}}


def _team(world: dict, tid: Optional[str]) -> dict:
    t = world["teams"].get(tid) if tid else None
    return {"team_id": tid, "abbrev": t["abbrev"] if t else tid, "name": t["name"] if t else tid}


def _result(world: dict, s: dict, *, seq: bool = False) -> dict:
    """A revealed match — score + winner, cores STRIPPED (hidden tiebreaker); sequence on request.
    Carries ``gid`` so the server can attach this match's real replay files (``map_replays``)."""
    out = {"gid": s["gid"], "home": s["home"], "away": s["away"],
           "home_abbrev": _team(world, s["home"])["abbrev"], "away_abbrev": _team(world, s["away"])["abbrev"],
           "stage": s.get("stage"), "league": s.get("league"), "best_of": s.get("best_of"),
           "maps_home": s["maps_home"], "maps_away": s["maps_away"], "winner": s["winner"]}
    if seq:
        out["sequence"] = s.get("sequence", [])
    return out


def _standings_rows(world: dict, ls: dict, revset: set, power: dict) -> dict:
    reg = {"upper": [], "lower_A": [], "lower_Z": []}
    for d in ls["days"]:
        if d["phase"] != "regular":
            continue
        for s in d["series"]:
            if s["gid"] in revset:
                reg[s["group"]].append(s)
    out = {}
    for gkey, ids in ls["groups"].items():
        rows = brackets.standings(ids, reg[gkey])
        out[gkey] = [{"team_id": r["team_id"], "abbrev": _team(world, r["team_id"])["abbrev"],
                      "name": _team(world, r["team_id"])["name"], "place": r["place"],
                      "points": r["points"], "series_wins": r["series_wins"],
                      "series_losses": r["series_losses"], "map_diff": r["map_diff"],
                      "power": power.get(r["team_id"], round(float(world["teams"][r["team_id"]]["elo"])))}
                     for r in rows]
    return out


def _bracket_so_far(world: dict, bracket: Optional[dict], revset: set) -> list[dict]:
    if not bracket:
        return []
    nodes = bracket["nodes"]
    out = []
    for key in bracket["order"]:
        node = nodes[key]
        home = resolve_for_reveal(node["home"], nodes, revset)
        away = resolve_for_reveal(node["away"], nodes, revset)
        shown = node.get("gid") in revset
        res = node.get("result")
        out.append({"key": key, "stage": node["stage"],
                    "home": _team(world, home) if home else None,
                    "away": _team(world, away) if away else None,
                    "played": shown,
                    "maps_home": res["maps_home"] if shown else None,
                    "maps_away": res["maps_away"] if shown else None,
                    "winner": res["winner"] if shown else None})
    return out


def resolve_for_reveal(ref, nodes: dict, revset: set) -> Optional[str]:
    """Concrete team id for a playoff endpoint, but only if its source game is REVEALED (so the
    bracket never leaks an unrevealed result). Concrete seeds are returned as-is — the caller gates
    those on the regular season being fully revealed."""
    if isinstance(ref, str):
        return ref
    node = nodes.get(ref.get("w") or ref.get("l"))
    res = node and node.get("result")
    if not res or node.get("gid") not in revset:
        return None
    return res["winner"] if "w" in ref else (res["home"] if res["winner"] == res["away"] else res["away"])


def _card(world: dict, ls: dict, s: dict, revset: set, power: dict, rows: dict) -> dict:
    """A single LIVE match card: both teams (Power/place/points/record/form), context, and the
    result + map sequence once this game is revealed."""
    def info(tid):
        row = rows.get(tid, {})
        form = []
        for d in ls["days"]:
            for ss in d["series"]:
                if ss["gid"] in revset and tid in (ss["home"], ss["away"]):
                    form.append("W" if ss["winner"] == tid else "L")
        t = world["teams"].get(tid, {})
        starters = sorted((world["players"][pid] for pid in t.get("roster", []) if pid in world["players"]),
                          key=lambda p: -float(p.get("soloq_rating", 0)))[:3]
        players = [{"nickname": p.get("nickname", p["player_id"]), "faction": p.get("faction"),
                    "rating": round(float(p.get("soloq_rating", 1000))), "archetype": p.get("archetype")}
                   for p in starters]
        return {"team_id": tid, "abbrev": t.get("abbrev", tid), "name": t.get("name", tid),
                "power": power.get(tid, round(float(t.get("elo", 1000)))),
                "place": row.get("place"), "points": row.get("points", 0),
                "series_wins": row.get("series_wins", 0), "series_losses": row.get("series_losses", 0),
                "map_diff": row.get("map_diff", 0), "form": form[-6:], "players": players}

    revealed_ = s["gid"] in revset
    out = {"gid": s["gid"], "stage": s.get("stage"), "stage_label": STAGE_LABEL_LIVE.get(s.get("stage"), s.get("stage")),
           "league": s.get("league"), "league_label": _league_label(s), "group": s.get("group"),
           "best_of": s.get("best_of"), "day": s.get("day"),
           # FROZEN pre-match win probability (computed once before the match; multi-factor)
           "win_prob": s.get("win_prob"), "win_prob_factors": s.get("win_prob_factors"),
           "home": info(s["home"]), "away": info(s["away"]), "revealed": revealed_}
    if revealed_:
        out.update({"maps_home": s["maps_home"], "maps_away": s["maps_away"],
                    "winner": s["winner"], "sequence": s.get("sequence", [])})
    return out


def _revset_upto(ls: dict, day_num: int, full: set) -> set:
    """Revealed gids whose (1-based) display day is <= ``day_num`` — the state as of the END of that
    day, with nothing from later days leaking in (used to render a PAST day spoiler-free)."""
    return {s["gid"] for d in ls["days"] for s in d["series"]
            if s["gid"] in full and s.get("day", 0) <= day_num}


def view(world: dict, day: Optional[int] = None) -> dict:
    """Spoiler-free snapshot of ONE day. ``day`` (0-based) lets the UI rewind to any ALREADY-PLAYED
    day: the live cursor (``shown_day``) is the current broadcast day; earlier days are shown as
    finished RESULTS (no broadcast) as of the end of that day, never leaking later games. With
    ``day=None`` it shows the live cursor (or the legacy "done" screen once the season is settled)."""
    ls = world.get("live_replay")
    if not ls:
        return {"active": False}
    full = _revealed_set(ls)
    n = ls["n_days"]
    cur = ls.get("shown_day", 0)
    live_active = cur < n
    max_view = cur if live_active else n - 1            # the furthest day you may look at
    viewing = max_view if day is None else max(0, min(int(day), max_view))
    is_current_day = live_active and viewing == cur
    showing_done = not live_active and day is None      # legacy "season finished" entry screen

    # the live cursor renders from everything revealed so far; a PAST day is frozen at its own end
    revset = full if (is_current_day or showing_done) else _revset_upto(ls, viewing + 1, full)
    power = _provisional_power(world, ls, revset)
    standings = _standings_rows(world, ls, revset, power)
    rows = {r["team_id"]: r for g in standings.values() for r in g}

    day_matches = [] if showing_done else [_card(world, ls, s, revset, power, rows)
                                           for s in ls["days"][viewing]["series"]]
    playoff_started = (n - 1 if showing_done else viewing) >= ls["n_regular_days"]
    return {
        "active": live_active,
        "season": ls["season"],
        "phase": "done" if showing_done else ls["days"][viewing]["phase"],
        "day": viewing + 1, "cursor": cur, "viewing_day": viewing,
        "total_days": n, "n_regular_days": ls["n_regular_days"],
        "is_current_day": is_current_day,
        "can_prev": viewing > 0,
        "can_next": viewing < max_view,
        "day_matches": day_matches,
        "standings": standings,
        "bracket": {
            "upper": _bracket_so_far(world, ls["playoff"]["upper"], revset) if playoff_started else [],
            "lower": _bracket_so_far(world, ls["playoff"]["lower"], revset) if playoff_started else [],
        },
        "done": _done_summary(world, ls) if showing_done else None,
    }


def _done_summary(world: dict, ls: dict) -> dict:
    fin = ls["finalize"]
    return {"season": ls["season"],
            "upper_champion": _team(world, fin["up_play"]["champion"])["abbrev"],
            "lower_champion": _team(world, fin["lo_play"]["champion"])["abbrev"]}


def reveal_next_day(world: dict) -> dict:
    """«Следующий день»: reveal any of the SHOWN day's still-unwatched games (instant), then advance
    the shown day. Advancing PAST the last day settles the season through the shared finaliser — that
    is the ONLY thing that finalises (watching never does), so the day moves only when the owner says."""
    ls = world.get("live_replay")
    if not ls:
        raise ValueError("no live season to reveal")
    n = ls["n_days"]
    sd = ls.get("shown_day", 0)
    if sd >= n:
        return {"already_done": True, "view": view(world)}
    revset = _revealed_set(ls)
    for s in ls["days"][sd]["series"]:
        if s["gid"] not in revset:
            ls["revealed_gids"].append(s["gid"])
    ls["shown_day"] = sd + 1
    finished = ls["shown_day"] >= n
    if finished:                            # advanced past the last day -> settle (all games revealed)
        _finalize(world, ls)
    return {"day": sd + 1, "finished": finished, "view": view(world)}


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
    """Watch (and COMMIT) ONE match of the SHOWN day — replay its stored map sequence and reveal it so
    every read-model updates immediately. Watching NEVER advances the day nor finalises the season
    (that is «Следующий день»'s job) — even watching the last game just commits it."""
    ls = world.get("live_replay")
    if not ls:
        raise ValueError("nothing to watch")
    sd = ls.get("shown_day", 0)
    if sd >= ls["n_days"]:
        raise ValueError("nothing to watch")
    revset = _revealed_set(ls)
    for s in ls["days"][sd]["series"]:
        if s["gid"] not in revset and s["home"] == home and s["away"] == away:
            ls["revealed_gids"].append(s["gid"])
            return {"result": _result(world, s, seq=True), "finished": False, "view": view(world)}
    raise ValueError("that match is not available to watch in the current day")
