"""Season orchestration: draft, regular season, and the Season-0 bootstrap.

Ties the foundation together. Matches are produced by an injected ``play``
callable; a deterministic default (``make_play_fn``) simulates series from roster
strength so the whole world runs with no trained nets. When the real engine is
wired, pass a ``play`` that runs actual games — nothing else changes.

See docs/cloudcode/WORLD_MODE_SPEC.md §1, §5, §8.
"""
from __future__ import annotations

import random
from typing import Callable, Optional

from src.core import rng as seeds
from src.regionals.elo import expected_score
from src.world import brackets, calendar as cal, economy, playbook, soloq, team_elo
from src.world.world import (
    STARTERS, get_player, lower_teams, roster_players, upper_teams,
)

UPPER_SIZE = 10
LOWER_SIZE = 12
GROUP_SIZE = 6
BOOTSTRAP_EXTRA_RUNS = 3  # lower-league runs after the upper league fills

PlayFn = Callable[[str, str, int], dict]


# --------------------------------------------------------------------------- #
# Deterministic default match engine (no nets needed)
# --------------------------------------------------------------------------- #
def team_strength(world: dict, team_id: str) -> float:
    """Base proxy strength for the deterministic sim: avg SoloQ of the best
    starting three (highest value). Extra roster players sit on the bench. The
    manager's PLAYBOOK adds a matchup modifier on top of this (see playbook.py,
    applied in make_play_fn)."""
    team = world["teams"][team_id]
    players = roster_players(world, team)
    if not players:
        return 1000.0
    starters = sorted(players, key=lambda p: -economy.player_value(p))[:3]
    return sum(p["soloq_rating"] for p in starters) / len(starters)


def make_play_fn(world: dict, seed: Optional[int] = None, *,
                 use_playbook: bool = True, record_positions: bool = True) -> PlayFn:
    """A ``play(home, away, best_of)`` that simulates a Bo-N series from roster
    strength, modulated by each team's ACTIVE playbook (strategy counter + execution
    + position-aware lineup/substitutions). Deterministic given an explicit seed
    (independent child stream per series via Spawner); UNIQUE per run when
    ``seed is None`` (§0-ter). ``use_playbook=False`` for pure roster sims/tests;
    ``record_positions=False`` for Season 0 (which persists only SoloQ)."""
    spawner = seeds.Spawner(seed)

    def play(home: str, away: str, best_of: int) -> dict:
        return play_one_series(world, home, away, best_of, spawner.random(),
                               use_playbook=use_playbook, record_positions=record_positions)

    return play


def play_one_series(world: dict, home: str, away: str, best_of: int, rng: random.Random,
                    *, use_playbook: bool = True, record_positions: bool = True) -> dict:
    """Simulate ONE Bo-N series from roster strength (modulated by the active playbook), playing
    map-by-map. Returns the result dict incl. ``sequence`` (the winner of each map, in order — for
    a no-scrub live broadcast) and ``cores_home``/``cores_away`` (the hidden league tiebreaker).
    Shared by ``make_play_fn`` (batch) and ``live_season`` (day-by-day / live watch)."""
    sh, sa = team_strength(world, home), team_strength(world, away)
    meta = None
    if use_playbook:
        dh, da, meta = playbook.matchup_modifiers(world, home, away, best_of)
        sh += dh
        sa += da
    p_home = expected_score(sh, sa)
    need = best_of // 2 + 1
    mh = ma = 0
    cores_h = cores_a = 0              # hidden granular stat: cores smashed per map won
    sequence: list[str] = []
    while mh < need and ma < need:
        if rng.random() < p_home:
            mh += 1
            cores_h += rng.randint(1, 3)
            sequence.append(home)
        else:
            ma += 1
            cores_a += rng.randint(1, 3)
            sequence.append(away)
    winner = home if mh > ma else away
    result = {"home": home, "away": away, "maps_home": mh, "maps_away": ma,
              "winner": winner, "cores_home": cores_h, "cores_away": cores_a,
              "best_of": best_of, "sequence": sequence}
    if meta:
        result["playbook"] = meta
        if record_positions:   # attribute the series to each starter's position
            playbook.record_position_results(world, meta["home"]["lineup"], winner == home)
            playbook.record_position_results(world, meta["away"]["lineup"], winner == away)
    return result


# --------------------------------------------------------------------------- #
# Draft
# --------------------------------------------------------------------------- #
DraftPickFn = Callable[[dict, list], Optional[str]]  # (team, available_players) -> pid|None


def balanced_draft(world: dict, team_ids: list[str], rng: random.Random,
                   target_size: int = 3, *, pick_fn: Optional[DraftPickFn] = None) -> None:
    """Snake-style draft from the free-agent pool until every team has
    ``target_size`` players. **Season 0 has NO weak teams — all are equal — so the
    pick order among teams still needing a player is RANDOM (fair), NOT
    weakest-first.** (The weakest-first rule begins at Season 1's rookie draft,
    see ``transfers.rookie_draft``.) ``pick_fn(team, available)`` lets the LLM pick
    per slot (L2 draft); falling back to best-by-value when it returns None/invalid.
    Mutates the world."""
    free = world["free_agents"]
    picks: list[dict] = []
    rnd = 0
    while any(len(world["teams"][tid]["roster"]) < target_size for tid in team_ids):
        rnd += 1
        # fewest players first; ties broken RANDOMLY (equal teams in Season 0).
        order = sorted(
            team_ids,
            key=lambda tid: (len(world["teams"][tid]["roster"]), rng.random()),
        )
        progressed = False
        for tid in order:
            team = world["teams"][tid]
            if len(team["roster"]) >= target_size:
                continue
            if not free:
                break  # pool exhausted
            pick = None
            if pick_fn is not None:
                pick = pick_fn(team, [get_player(world, pid) for pid in free])
            if pick is None or pick not in free:   # heuristic fallback: best by value
                free.sort(key=lambda pid: -economy.player_value(get_player(world, pid)))
                pick = free[0]
            free.remove(pick)
            player = get_player(world, pick)
            player["status"] = "rostered"
            player["team_id"] = tid
            team["roster"].append(pick)
            picks.append({"round": rnd, "overall": len(picks) + 1, "team_id": tid, "player_id": pick})
            progressed = True
        if not progressed:
            break
    if picks:                # record the draft board for the UI
        world.setdefault("drafts", []).append(
            {"season": world.get("season", 0), "kind": "bootstrap", "picks": picks})


# --------------------------------------------------------------------------- #
# Group assignment
# --------------------------------------------------------------------------- #
def assign_lower_groups(world: dict, team_ids: list[str],
                        forced: Optional[dict] = None) -> None:
    """Split ``team_ids`` into balanced groups A/Z (snake by Elo). ``forced`` maps
    team_id -> group to honor specific placements (e.g. relegated 9th->A, 10th->Z)."""
    forced = forced or {}
    groups: dict[str, list[str]] = {"A": [], "Z": []}
    remaining = []
    for tid in team_ids:
        g = forced.get(tid)
        if g in ("A", "Z") and len(groups[g]) < GROUP_SIZE:
            groups[g].append(tid)
        else:
            remaining.append(tid)
    remaining.sort(key=lambda tid: -world["teams"][tid]["elo"])
    for tid in remaining:
        target = "A" if len(groups["A"]) <= len(groups["Z"]) and len(groups["A"]) < GROUP_SIZE else "Z"
        groups[target].append(tid)
    for g, members in groups.items():
        for tid in members:
            t = world["teams"][tid]
            t["league"] = "lower"
            t["group"] = g


def _group_members(world: dict, group: str) -> list[str]:
    return [t["team_id"] for t in world["teams"].values()
            if t["league"] == "lower" and t["group"] == group]


# --------------------------------------------------------------------------- #
# Regular season
# --------------------------------------------------------------------------- #
def _lower_final_ranking(champ: str, finalists: list[str],
                         rows_a: list[dict], rows_z: list[dict]) -> list[str]:
    """Approximate 1..12 ranking for prize money: grand finalists first, then the
    rest by group standing (place, map_diff)."""
    other_finalist = [t for t in finalists if t != champ]
    head = [champ] + other_finalist
    rest = [r for r in (rows_a + rows_z) if r["team_id"] not in head]
    rest.sort(key=lambda r: (r["place"], -r["map_diff"], -r["maps_won"]))
    return head + [r["team_id"] for r in rest]


_PUBLIC_STANDING_KEYS = ("series_wins", "series_losses", "maps_won", "maps_lost",
                         "map_diff", "points", "place")


def _public_standings(rows: list[dict]) -> dict:
    """Standings keyed by team_id with only the PUBLIC columns — cores_destroyed
    (the hidden tiebreaker) is intentionally excluded so it never reaches the UI."""
    return {r["team_id"]: {k: r[k] for k in _PUBLIC_STANDING_KEYS} for r in rows}


def run_regular_season(world: dict, play: Optional[PlayFn] = None, *,
                       seed: Optional[int] = None, llm=None,
                       settle_economy: bool = True, record: bool = True) -> dict:
    """Play one regular season: groups -> playoffs -> Elo -> economy ->
    promotion/relegation -> calendar. Returns a season summary. ``seed`` is
    optional: None (production) → fresh entropy so each season plays out uniquely;
    pass an explicit seed (tests) for a reproducible season (§0-ter)."""
    season = world["season"]
    resolved_seed = seeds.resolve_seed(seed)     # recorded in the summary for history
    play = play or make_play_fn(world, seed=resolved_seed)

    up_ids = [t["team_id"] for t in upper_teams(world)]
    a_ids = _group_members(world, "A")
    z_ids = _group_members(world, "Z")

    # ---- group stage ---------------------------------------------------- #
    up_results = brackets.play_group(up_ids, play)
    a_results = brackets.play_group(a_ids, play)
    z_results = brackets.play_group(z_ids, play)

    up_rows = brackets.standings(up_ids, up_results)
    a_rows = brackets.standings(a_ids, a_results)
    z_rows = brackets.standings(z_ids, z_results)

    # post-group reaction (flag post_group_replan, default off): struggling clubs may rewrite their
    # BASE strategy before playoffs (lineup frozen). play reads the plan per match, so it takes effect
    # in the playoffs below. No-op without the flag + an active llm.
    from src.world import llm_manager
    llm_manager.replan_after_groups(world, [up_rows, a_rows, z_rows], llm=llm)

    # ---- playoffs ------------------------------------------------------- #
    up_play = brackets.upper_playoff([r["team_id"] for r in up_rows], play)
    lo_play = brackets.lower_playoff([r["team_id"] for r in a_rows],
                                     [r["team_id"] for r in z_rows], play)

    return finalize_season(world, season, up_rows, a_rows, z_rows, up_play, lo_play,
                           up_results, a_results, z_results,
                           settle_economy=settle_economy, record=record, seed=resolved_seed)


def finalize_season(world: dict, season: int, up_rows: list[dict], a_rows: list[dict],
                    z_rows: list[dict], up_play: dict, lo_play: dict,
                    up_results: list[dict], a_results: list[dict], z_results: list[dict],
                    *, settle_economy: bool = True, record: bool = True,
                    apply_elo: bool = True, seed: Optional[int] = None) -> dict:
    """Close a season from its resolved standings + playoff results: apply Elo, record the
    calendar, settle economy + trophies, run promotion/relegation, append the history summary.

    Shared by the all-at-once ``run_regular_season`` and the day-by-day ``live_season`` engine
    so both settle a season IDENTICALLY (single source of truth for prizes/promo/history).
    ``apply_elo``/``record`` are False for the live engine, which already applied Elo + recorded
    the calendar series-by-series as each day was played."""
    # ---- Elo from all official series ----------------------------------- #
    if apply_elo:
        all_series = (up_results + a_results + z_results
                      + up_play["series"] + lo_play["series"])
        for r in all_series:
            team_elo.apply_series_result(world, r["home"], r["away"],
                                         r["maps_home"], r["maps_away"], r.get("best_of", 3))

    # ---- calendar ------------------------------------------------------- #
    if record:
        cal.record_series(world, season, "upper", up_results + up_play["series"])
        cal.record_series(world, season, "lower", a_results + z_results + lo_play["series"])

    # ---- economy -------------------------------------------------------- #
    upper_ranking = [r["team_id"] for r in up_rows]            # 1..10
    lower_ranking = _lower_final_ranking(lo_play["champion"], lo_play["grand_finalists"],
                                         a_rows, z_rows)        # 1..12
    prizes_by_team: dict = {}
    prize_breakdown: dict = {}
    salaries_by_team: dict = {}
    treasury_before = {tid: round(float(t.get("treasury", 0))) for tid, t in world["teams"].items()}
    if settle_economy:
        placements = []
        for place, tid in enumerate(upper_ranking, start=1):
            placements.append({"team_id": tid, "league": "upper", "place": place,
                               "grand_final_winner": tid == up_play["champion"]})
        # Lower prizes are PER GROUP (A and Z are separate, equal pools) — place = rank within the
        # group, never a mixed 1..12. The grand-final winner still gets the lower champion bonus.
        for grp, rows in (("A", a_rows), ("Z", z_rows)):
            for r in rows:
                placements.append({"team_id": r["team_id"], "league": "lower", "place": r["place"],
                                   "group": grp, "grand_final_winner": r["team_id"] == lo_play["champion"]})
        prize_breakdown = economy.award_prizes(world, placements)   # {tid: {place, league, prize, bonus, total}}
        prizes_by_team = {tid: v["total"] for tid, v in prize_breakdown.items()}
        salaries_by_team = economy.pay_salaries(world)
        _record_trophies(world, season, up_play["champion"], lo_play["champion"])

    # ---- promotion / relegation ---------------------------------------- #
    relegated = upper_ranking[8:10]    # 9th, 10th
    promoted = lo_play["promoted"]
    _apply_promotion_relegation(world, promoted, relegated)

    summary = {
        "season": season,
        "seed": seed,                              # this run's seed (unique unless pinned)
        "upper_ranking": upper_ranking,
        "lower_ranking": lower_ranking,
        "upper_champion": up_play["champion"],
        "lower_champion": lo_play["champion"],
        "promoted": promoted,
        "relegated": relegated,
        # group standings the Leagues/Table tabs read — points + map delta are PUBLIC,
        # cores_destroyed is the hidden tiebreaker and is deliberately dropped here.
        "standings": {
            "upper": _public_standings(up_rows),
            "lower_A": _public_standings(a_rows),
            "lower_Z": _public_standings(z_rows),
        },
        # who won what + the season's money (treasury alone doesn't show it) — for History + Economy
        "playoff": {
            "upper": {"champion": up_play["champion"],
                      "grand_finalists": up_play.get("grand_finalists", [])},
            "lower": {"champion": lo_play["champion"],
                      "grand_finalists": lo_play.get("grand_finalists", []),
                      "promoted": lo_play.get("promoted", [])},
        },
        "money": {
            "prizes_by_team": prizes_by_team,           # {tid: total} (int) — back-compat
            "prize_breakdown": prize_breakdown,         # {tid: {place, league, prize, bonus, total}}
            "salaries_by_team": salaries_by_team,
            "treasury_before": treasury_before,
            "treasury_after": {tid: round(float(t.get("treasury", 0))) for tid, t in world["teams"].items()},
        },
    }
    world.setdefault("history", []).append(summary)
    return summary


def _record_trophies(world: dict, season: int, upper_champ: str, lower_champ: str) -> None:
    for tid, title in ((upper_champ, "upper_champion"), (lower_champ, "lower_promotion_champion")):
        team = world["teams"][tid]
        team["titles"] += 1
        team["trophies"].append({"season": season, "title": title})
        for pid in team["roster"]:
            get_player(world, pid)["trophies"].append({"season": season, "title": title})


def _apply_promotion_relegation(world: dict, promoted: list[str], relegated: list[str]) -> None:
    """Swap promoted (lower->upper) and relegated (upper->lower). Lower groups are
    rebuilt 6/6 with relegated 9th->A, 10th->Z honored."""
    for tid in promoted:
        t = world["teams"][tid]
        t["league"], t["group"] = "upper", None
    forced = {}
    if len(relegated) >= 1:
        forced[relegated[0]] = "A"   # 9th
    if len(relegated) >= 2:
        forced[relegated[1]] = "Z"   # 10th
    # New lower membership = (old lower minus promoted) + relegated.
    new_lower = [t["team_id"] for t in world["teams"].values()
                 if t["league"] == "lower" and t["team_id"] not in promoted]
    new_lower += [tid for tid in relegated if tid not in new_lower]
    # clear groups then reassign
    for tid in new_lower:
        world["teams"][tid]["group"] = None
    assign_lower_groups(world, new_lower, forced=forced)


# --------------------------------------------------------------------------- #
# Season 0 bootstrap
# --------------------------------------------------------------------------- #
def _preseason_digest(world: dict, prev: dict) -> dict:
    """Leader + biggest gainer/loser after a SoloQ pre-season, vs the ``prev`` rating snapshot.
    Reserves (held out, never played) are excluded so the 'leader' is always a real player."""
    active = [(pid, p) for pid, p in world["players"].items() if p.get("status") != "reserve"]
    nick = lambda pid: world["players"][pid].get("nickname", pid)
    rt = lambda p: round(float(p.get("soloq_rating", 1000.0)))
    leader_pid, leader_p = max(active, key=lambda kp: float(kp[1].get("soloq_rating", 1000.0)))
    deltas = [(pid, rt(p) - round(float(prev.get(pid, 1000.0)))) for pid, p in active]
    gain_pid, gain_d = max(deltas, key=lambda x: x[1], default=(leader_pid, 0))
    lose_pid, lose_d = min(deltas, key=lambda x: x[1], default=(leader_pid, 0))
    return {"leader_nick": nick(leader_pid), "leader_rating": rt(leader_p),
            "gainer_nick": nick(gain_pid), "gainer_delta": gain_d,
            "loser_nick": nick(lose_pid), "loser_delta": lose_d}


def bootstrap_world(world: dict, *, seed: Optional[int] = None,
                    play: Optional[PlayFn] = None,
                    draft_pick_fn: Optional[DraftPickFn] = None,
                    soloq_match_fn: Optional[Callable] = None,
                    soloq_match_player: Optional[Callable] = None,
                    workers: int = 1,
                    soloq_rounds: int = soloq.DEFAULT_ROUNDS,
                    soloq_preseasons: int = 8,
                    progress_cb: Optional[Callable[[str, dict], None]] = None) -> dict:
    """Run the Season-0 bootstrap (see spec §8), in real-world order:

    1. DRAFT FIRST — every club picks its 3 starters (LLM via ``draft_pick_fn``, else heuristic),
       BEFORE anyone has grinded ranked (Season 0: equal/random order; no weakest-first yet).
    2. SoloQ pre-seasons — the drafted + free-agent players grind ranked, building rating history.
    3. Lower-league series promote 2 each run until the upper league fills (10), refilling lower
       from a reserve queue; then +3 warm-up runs.
    No money, no salaries, no match history, no Elo persist — ONLY SoloQ.
    ``seed`` optional: None → fresh entropy (unique world); explicit → reproducible.
    ``progress_cb(msg, data)`` (optional) fires at every stage for LIVE progress + a stored chronicle.
    Ends with leagues set and treasuries seeded for Season 1.
    """
    resolved = seeds.resolve_seed(seed)
    sp = seeds.Spawner(resolved)        # independent child streams (no base+offset)
    chronicle: list[dict] = []

    def _log(msg: str, **data) -> None:
        chronicle.append({"msg": msg, **data})
        if progress_cb:
            try:
                progress_cb(msg, data)
            except Exception:           # progress reporting must NEVER break the bootstrap
                pass

    all_team_ids = list(world["teams"].keys())

    # 1. DRAFT FIRST (owner: draft -> then SoloQ + seasons).
    _log("🎬 Драфт стартовых составов…", kind="draft_start")
    balanced_draft(world, all_team_ids, sp.random(), target_size=STARTERS, pick_fn=draft_pick_fn)
    _log(f"✍️ Драфт завершён — {len(all_team_ids)} клубов набрали по {STARTERS} игрока", kind="draft_done")

    # 2. SoloQ pre-seasons (AFTER the draft) — net-backed when soloq_match_fn is supplied (real games),
    #    else the rating-based stochastic model (tests stay net-free). Each pre-season adds one point to
    #    every player's rating history; we log each pre-season's leader + biggest mover.
    prev = {pid: float(p.get("soloq_rating", 1000.0)) for pid, p in world["players"].items()}
    npre = max(1, soloq_preseasons)
    for i in range(1, npre + 1):
        soloq.apply_soloq(world, rounds=soloq_rounds, seed=sp.seed(), match_fn=soloq_match_fn,
                          match_player=soloq_match_player, workers=workers)
        d = _preseason_digest(world, prev)
        _log(f"🎮 Пред-сезон SoloQ {i}/{npre}: 👑 {d['leader_nick']} ({d['leader_rating']}) · "
             f"▲ {d['gainer_nick']} (+{d['gainer_delta']}) · ▼ {d['loser_nick']} ({d['loser_delta']})",
             kind="soloq_preseason", n=i, total=npre, **d)
        prev = {pid: float(p.get("soloq_rating", 1000.0)) for pid, p in world["players"].items()}

    # 3. Bootstrap leagues
    _log("🏟 Квалификация: повышаем клубы в верхнюю лигу…", kind="league_start")
    lower_ids = all_team_ids[:LOWER_SIZE]
    reserve = list(all_team_ids[LOWER_SIZE:])
    assign_lower_groups(world, lower_ids)
    upper_ids: list[str] = []
    # Power starts at 1000 for EVERYONE and is COMPUTED from the bootstrap qualification matches
    # (chess-Elo): each series moves Elo by margin × (result − expected), so clubs that win their way
    # up earn a higher Season-1 Power — no artificial league seed.
    for t in world["teams"].values():
        t["elo"] = team_elo.START_ELO
        t["elo_history"] = [round(team_elo.START_ELO)]
    # per-position stats stay off in Season 0; but Power (Elo) DOES accrue from these games now.
    base_play = play or make_play_fn(world, seed=sp.seed(), record_positions=False)

    def play(home, away, best_of):                       # wrap: apply chess-Elo after each series
        r = base_play(home, away, best_of)
        if r.get("maps_home") != r.get("maps_away"):     # decisive series only (Elo needs a winner)
            team_elo.apply_series_result(world, r["home"], r["away"],
                                         r["maps_home"], r["maps_away"], r.get("best_of", best_of))
        return r

    runs = 0
    guard = 0
    while len(upper_ids) < UPPER_SIZE and guard < 50:
        guard += 1
        a_ids = _group_members(world, "A")
        z_ids = _group_members(world, "Z")
        if len(a_ids) < 3 or len(z_ids) < 3:
            break
        a_rows = brackets.standings(a_ids, brackets.play_group(a_ids, play))
        z_rows = brackets.standings(z_ids, brackets.play_group(z_ids, play))
        result = brackets.lower_playoff([r["team_id"] for r in a_rows],
                                        [r["team_id"] for r in z_rows], play)
        runs += 1
        for tid in result["promoted"]:
            if len(upper_ids) < UPPER_SIZE:
                upper_ids.append(tid)
                world["teams"][tid]["league"] = "upper"
                world["teams"][tid]["group"] = None
                _log(f"⬆️ {world['teams'][tid]['abbrev']} вышел в верхнюю лигу "
                     f"({len(upper_ids)}/{UPPER_SIZE})", kind="promotion", team_id=tid)
        # refill lower from reserve
        current_lower = [t["team_id"] for t in world["teams"].values()
                         if t["league"] == "lower"]
        while len(current_lower) < LOWER_SIZE and reserve:
            current_lower.append(reserve.pop(0))
        for tid in current_lower:
            world["teams"][tid]["group"] = None
        if current_lower:
            assign_lower_groups(world, current_lower)

    # +3 warm-up lower runs (nothing persists; honored for fidelity)
    for _ in range(BOOTSTRAP_EXTRA_RUNS):
        a_ids = _group_members(world, "A")
        z_ids = _group_members(world, "Z")
        if len(a_ids) >= 3 and len(z_ids) >= 3:
            brackets.lower_playoff(
                [r["team_id"] for r in brackets.standings(a_ids, brackets.play_group(a_ids, play))],
                [r["team_id"] for r in brackets.standings(z_ids, brackets.play_group(z_ids, play))],
                play)

    # Power now reflects the bootstrap matches (Elo accrued above) — promoted clubs naturally sit
    # higher, having WON their way up. Reset each club's history to a SINGLE clean point (its earned
    # Power) so the Season-1 sparkline starts there. Power depends ONLY on results, never on ratings.
    for t in world["teams"].values():
        t["elo"] = round(float(t.get("elo", team_elo.START_ELO)), 2)
        t["elo_history"] = [round(t["elo"])]

    # Seed treasuries for Season 1.
    for t in world["teams"].values():
        if t["league"] == "upper":
            t["treasury"] = economy.CONFIG.upper_seed_treasury
        elif t["league"] == "lower":
            t["treasury"] = economy.CONFIG.lower_seed_treasury

    _log(f"🏆 Бутстрап завершён — верхняя лига заполнена "
         f"({len(upper_ids)}/{UPPER_SIZE}), сезон 1 готов", kind="done")
    world["season"] = 1
    world["phase"] = "regular"
    world["bootstrap"] = {"lower_runs": runs, "extra_runs": BOOTSTRAP_EXTRA_RUNS,
                          "upper_filled": len(upper_ids) == UPPER_SIZE,
                          "seed": resolved, "chronicle": chronicle}
    return world
