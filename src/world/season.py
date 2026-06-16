"""Season orchestration: draft, regular season, and the Season-0 bootstrap.

Ties the foundation together. Matches are produced by an injected ``play``
callable; a deterministic default (``make_play_fn``) simulates series from roster
strength so the whole world runs with no trained nets. When the real engine is
wired, pass a ``play`` that runs actual games — nothing else changes.

See docs/cloudcode/WORLD_MODE_SPEC.md §1, §5, §8.
"""
from __future__ import annotations

import random
import zlib
from typing import Callable, Optional

from src.regionals.elo import expected_score
from src.world import brackets, calendar as cal, economy, soloq, team_elo
from src.world.world import (
    get_player, lower_teams, roster_players, upper_teams,
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
    """Avg SoloQ rating of a team's roster — the proxy strength for the
    foundation's deterministic match simulation."""
    team = world["teams"][team_id]
    players = roster_players(world, team)
    if not players:
        return soloq.K_FACTOR * 0 + 1000.0
    return sum(p["soloq_rating"] for p in players) / len(players)


def make_play_fn(world: dict, seed: int = 0) -> PlayFn:
    """A seeded ``play(home, away, best_of)`` that simulates a Bo-N series from
    roster strength. Deterministic given the seed (per-series counter)."""
    state = {"n": 0}
    base = random.Random(seed)

    def play(home: str, away: str, best_of: int) -> dict:
        state["n"] += 1
        pair = zlib.crc32(f"{home}|{away}".encode("utf-8"))  # stable across processes
        rng = random.Random(seed * 1_000_003 + state["n"] * 97 + (pair & 0xFFFF))
        sh, sa = team_strength(world, home), team_strength(world, away)
        p_home = expected_score(sh, sa)
        need = best_of // 2 + 1
        mh = ma = 0
        while mh < need and ma < need:
            if rng.random() < p_home:
                mh += 1
            else:
                ma += 1
        winner = home if mh > ma else away
        return {"home": home, "away": away, "maps_home": mh, "maps_away": ma, "winner": winner}

    # keep base referenced (future: shared noise); avoids lint on unused.
    _ = base
    return play


# --------------------------------------------------------------------------- #
# Draft
# --------------------------------------------------------------------------- #
def _roster_value(world: dict, team: dict) -> float:
    return sum(economy.player_value(get_player(world, pid)) for pid in team["roster"])


def balanced_draft(world: dict, team_ids: list[str], rng: random.Random,
                   target_size: int = 3) -> None:
    """Snake-style balanced draft from the free-agent pool until every team has
    ``target_size`` players. Each round the weakest roster (lowest total value)
    picks first; it takes the best available free agent. Mutates the world."""
    free = world["free_agents"]
    while any(len(world["teams"][tid]["roster"]) < target_size for tid in team_ids):
        # order teams: fewest players first, then weakest roster, then stable.
        order = sorted(
            team_ids,
            key=lambda tid: (len(world["teams"][tid]["roster"]),
                             _roster_value(world, world["teams"][tid]), tid),
        )
        progressed = False
        for tid in order:
            team = world["teams"][tid]
            if len(team["roster"]) >= target_size:
                continue
            if not free:
                return  # pool exhausted
            # best available by value
            free.sort(key=lambda pid: -economy.player_value(get_player(world, pid)))
            pick = free.pop(0)
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
                       settle_economy: bool = True, record: bool = True) -> dict:
    """Play one regular season: groups -> playoffs -> Elo -> economy ->
    promotion/relegation -> calendar. Returns a season summary."""
    season = world["season"]
    play = play or make_play_fn(world, seed=season)

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
def bootstrap_world(world: dict, *, seed: int = 0, play: Optional[PlayFn] = None,
                    soloq_rounds: int = soloq.DEFAULT_ROUNDS) -> dict:
    """Run the Season-0 bootstrap (see spec §8):

    1. SoloQ seeds player ratings (persists).
    2. Balanced draft to 3 for all teams.
    3. Lower-league series promote 2 each run until the upper league fills (10),
       refilling lower from a reserve queue; then +3 warm-up runs.
    No money, no salaries, no match history, no Elo persist — ONLY SoloQ.
    Ends with leagues set and treasuries seeded for Season 1.
    """
    rng = random.Random(seed)

    # 1. SoloQ
    soloq.apply_soloq(world, rounds=soloq_rounds, seed=seed)

    # 2. Draft all teams to 3
    all_team_ids = list(world["teams"].keys())
    if len(all_team_ids) != UPPER_SIZE + LOWER_SIZE:
        # Foundation supports the canonical 22; degrade gracefully otherwise.
        pass
    balanced_draft(world, all_team_ids, rng, target_size=3)

    # 3. Bootstrap leagues
    lower_ids = all_team_ids[:LOWER_SIZE]
    reserve = list(all_team_ids[LOWER_SIZE:])
    assign_lower_groups(world, lower_ids)
    upper_ids: list[str] = []
    play = play or make_play_fn(world, seed=seed + 1)

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

    # Season 0 does not persist Elo — reset to start.
    for t in world["teams"].values():
        t["elo"] = team_elo.START_ELO
        t["elo_history"] = []

    # Seed treasuries for Season 1.
    for t in world["teams"].values():
        if t["league"] == "upper":
            t["treasury"] = economy.CONFIG.upper_seed_treasury
        elif t["league"] == "lower":
            t["treasury"] = economy.CONFIG.lower_seed_treasury

    world["season"] = 1
    world["phase"] = "regular"
    world["bootstrap"] = {"lower_runs": runs, "extra_runs": BOOTSTRAP_EXTRA_RUNS,
                          "upper_filled": len(upper_ids) == UPPER_SIZE}
    return world
