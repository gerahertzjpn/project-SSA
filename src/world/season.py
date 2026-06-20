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
        rng = spawner.random()
        sh, sa = team_strength(world, home), team_strength(world, away)
        meta = None
        if use_playbook:
            dh, da, meta = playbook.matchup_modifiers(world, home, away, best_of)
            sh += dh
            sa += da
        p_home = expected_score(sh, sa)
        need = best_of // 2 + 1
        mh = ma = 0
        while mh < need and ma < need:
            if rng.random() < p_home:
                mh += 1
            else:
                ma += 1
        winner = home if mh > ma else away
        result = {"home": home, "away": away, "maps_home": mh, "maps_away": ma,
                  "winner": winner}
        if meta:
            result["playbook"] = meta
            if record_positions:   # attribute the series to each starter's position
                playbook.record_position_results(world, meta["home"]["lineup"], winner == home)
                playbook.record_position_results(world, meta["away"]["lineup"], winner == away)
        return result

    return play


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
    while any(len(world["teams"][tid]["roster"]) < target_size for tid in team_ids):
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
                return  # pool exhausted
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
            progressed = True
        if not progressed:
            break


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


def run_regular_season(world: dict, play: Optional[PlayFn] = None, *,
                       seed: Optional[int] = None,
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

    # ---- playoffs ------------------------------------------------------- #
    up_play = brackets.upper_playoff([r["team_id"] for r in up_rows], play)
    lo_play = brackets.lower_playoff([r["team_id"] for r in a_rows],
                                     [r["team_id"] for r in z_rows], play)

    # ---- Elo from all official series ----------------------------------- #
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
    if settle_economy:
        placements = []
        for place, tid in enumerate(upper_ranking, start=1):
            placements.append({"team_id": tid, "league": "upper", "place": place,
                               "grand_final_winner": tid == up_play["champion"]})
        for place, tid in enumerate(lower_ranking, start=1):
            placements.append({"team_id": tid, "league": "lower", "place": place,
                               "grand_final_winner": tid == lo_play["champion"]})
        economy.award_prizes(world, placements)
        economy.pay_salaries(world)
        _record_trophies(world, season, up_play["champion"], lo_play["champion"])

    # ---- promotion / relegation ---------------------------------------- #
    relegated = upper_ranking[8:10]    # 9th, 10th
    promoted = lo_play["promoted"]
    _apply_promotion_relegation(world, promoted, relegated)

    summary = {
        "season": season,
        "seed": resolved_seed,                     # this run's seed (unique unless pinned)
        "upper_ranking": upper_ranking,
        "lower_ranking": lower_ranking,
        "upper_champion": up_play["champion"],
        "lower_champion": lo_play["champion"],
        "promoted": promoted,
        "relegated": relegated,
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
def bootstrap_world(world: dict, *, seed: Optional[int] = None,
                    play: Optional[PlayFn] = None,
                    draft_pick_fn: Optional[DraftPickFn] = None,
                    soloq_match_fn: Optional[Callable] = None,
                    soloq_rounds: int = soloq.DEFAULT_ROUNDS) -> dict:
    """Run the Season-0 bootstrap (see spec §8):

    1. SoloQ seeds player ratings (persists).
    2. Balanced draft to 3 for all teams (Season 0: equal/random order). The
       ``draft_pick_fn`` hook lets the LLM draft per-pick (L2); default = heuristic.
    3. Lower-league series promote 2 each run until the upper league fills (10),
       refilling lower from a reserve queue; then +3 warm-up runs.
    No money, no salaries, no match history, no Elo persist — ONLY SoloQ.
    ``seed`` optional: None → fresh entropy (unique world); explicit → reproducible.
    Ends with leagues set and treasuries seeded for Season 1.
    """
    resolved = seeds.resolve_seed(seed)
    sp = seeds.Spawner(resolved)        # independent child streams (no base+offset)

    # 1. SoloQ (net-backed when soloq_match_fn is supplied — real games seed ratings;
    #    default None -> the rating-based stochastic model, so tests stay net-free)
    soloq.apply_soloq(world, rounds=soloq_rounds, seed=sp.seed(), match_fn=soloq_match_fn)

    # 2. Draft all teams to 3 (equal/random order in Season 0)
    all_team_ids = list(world["teams"].keys())
    if len(all_team_ids) != UPPER_SIZE + LOWER_SIZE:
        # Foundation supports the canonical 22; degrade gracefully otherwise.
        pass
    balanced_draft(world, all_team_ids, sp.random(), target_size=STARTERS, pick_fn=draft_pick_fn)

    # 3. Bootstrap leagues
    lower_ids = all_team_ids[:LOWER_SIZE]
    reserve = list(all_team_ids[LOWER_SIZE:])
    assign_lower_groups(world, lower_ids)
    upper_ids: list[str] = []
    # Season 0 persists ONLY SoloQ -> do not accumulate per-position stats here.
    play = play or make_play_fn(world, seed=sp.seed(), record_positions=False)

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

    # Season 0 doesn't persist per-series Elo, but the league a club EARNED (promotion IS a result)
    # seeds its starting Power, so Elo is GLOBAL from day one — promoted (upper) clubs start above
    # lower ones, then real results refine it. Within a tier it's flat (Season 1+ differentiates).
    # A strong lower club can still overtake a weak upper one by out-performing (>seed gap) — not
    # an absolute rule. Power depends ONLY on results, never on player ratings.
    UPPER_SEED = team_elo.START_ELO + 120.0
    for t in world["teams"].values():
        seed = UPPER_SEED if t["league"] == "upper" else team_elo.START_ELO
        t["elo"] = seed
        t["elo_history"] = [round(seed)]

    # Seed treasuries for Season 1.
    for t in world["teams"].values():
        if t["league"] == "upper":
            t["treasury"] = economy.CONFIG.upper_seed_treasury
        elif t["league"] == "lower":
            t["treasury"] = economy.CONFIG.lower_seed_treasury

    world["season"] = 1
    world["phase"] = "regular"
    world["bootstrap"] = {"lower_runs": runs, "extra_runs": BOOTSTRAP_EXTRA_RUNS,
                          "upper_filled": len(upper_ids) == UPPER_SIZE,
                          "seed": resolved}
    return world
