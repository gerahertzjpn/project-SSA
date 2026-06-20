"""League groups + playoff brackets.

Pure structure: given seeded team ids and a ``play(home, away, best_of)`` callable
(which returns a series result), resolve the round-robin standings and the
playoff trees. ``play`` is injected so the same brackets run with a real engine
or a deterministic test stub.

``play(home, away, best_of) -> {"home","away","maps_home","maps_away","winner"}``
"""
from __future__ import annotations

import hashlib
from typing import Callable

PlayFn = Callable[[str, str, int], dict]

LEAGUE_BEST_OF = 3
PLAYOFF_BEST_OF = 5
POINTS_PER_WIN = 3       # league points awarded for a series win (no draws in Bo-N)


def _stable_tiebreak(team_id: str) -> int:
    """A deterministic-but-arbitrary per-team value used as the FINAL standings
    tiebreaker — when points, map delta AND cores destroyed all tie, 'random'
    decides (we make it reproducible from the id so a season replays identically)."""
    return int(hashlib.sha1(team_id.encode("utf-8")).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- #
# Round robin + standings
# --------------------------------------------------------------------------- #
def round_robin_pairings(team_ids: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i, home in enumerate(team_ids):
        for away in team_ids[i + 1:]:
            pairs.append((home, away))
    return pairs


def play_group(team_ids: list[str], play: PlayFn, best_of: int = LEAGUE_BEST_OF) -> list[dict]:
    """Play a full single round-robin; return the list of series results."""
    return [play(h, a, best_of) for h, a in round_robin_pairings(team_ids)]


def standings(team_ids: list[str], results: list[dict]) -> list[dict]:
    """Rank a group. The position is decided, in order:
      1. ``points`` (POINTS_PER_WIN per series win)
      2. map differential (the visible 'дельта' tiebreaker)
      3. ``cores_destroyed`` — a HIDDEN granular stat (cores smashed across won maps)
      4. a stable pseudo-random value (when all three tie, 'random' decides)

    Returns rows ``{team_id, series_wins, series_losses, maps_won, maps_lost,
    map_diff, points, cores_destroyed, place}`` in finishing order. ``cores_destroyed``
    is computed but must NOT be surfaced to the user (it only breaks ties)."""
    rec = {tid: {"team_id": tid, "series_wins": 0, "series_losses": 0,
                 "maps_won": 0, "maps_lost": 0, "cores_destroyed": 0} for tid in team_ids}
    for r in results:
        h, a = r["home"], r["away"]
        if h not in rec or a not in rec:
            continue
        rec[h]["maps_won"] += r["maps_home"]
        rec[h]["maps_lost"] += r["maps_away"]
        rec[a]["maps_won"] += r["maps_away"]
        rec[a]["maps_lost"] += r["maps_home"]
        rec[h]["cores_destroyed"] += int(r.get("cores_home", 0) or 0)
        rec[a]["cores_destroyed"] += int(r.get("cores_away", 0) or 0)
        if r["maps_home"] > r["maps_away"]:
            rec[h]["series_wins"] += 1
            rec[a]["series_losses"] += 1
        else:
            rec[a]["series_wins"] += 1
            rec[h]["series_losses"] += 1
    rows = list(rec.values())
    for row in rows:
        row["map_diff"] = row["maps_won"] - row["maps_lost"]
        row["points"] = row["series_wins"] * POINTS_PER_WIN
    rows.sort(key=lambda x: (-x["points"], -x["map_diff"], -x["cores_destroyed"],
                             _stable_tiebreak(x["team_id"])))
    for i, row in enumerate(rows):
        row["place"] = i + 1
    return rows


# --------------------------------------------------------------------------- #
# Playoff helpers
# --------------------------------------------------------------------------- #
def _series(play: PlayFn, stage: str, home: str, away: str,
            log: list[dict], best_of: int = PLAYOFF_BEST_OF) -> tuple[str, str]:
    """Play one playoff series, tag it with ``stage``, append to ``log``.
    Returns (winner_id, loser_id)."""
    r = play(home, away, best_of)
    r = dict(r)
    r["stage"] = stage
    r["best_of"] = best_of
    log.append(r)
    winner = r["winner"]
    loser = away if winner == home else home
    return winner, loser


# --------------------------------------------------------------------------- #
# Lower league playoff — promotes 2
# --------------------------------------------------------------------------- #
def lower_playoff(group_a: list[str], group_z: list[str], play: PlayFn) -> dict:
    """``group_a``/``group_z`` are standings-ordered team ids (index 0 = 1st).

    Bracket (per spec):
      UB:  A1 vs Z1            -> winner to GRAND FINAL, loser to LB final
      LB:  A2 vs Z3 ; Z2 vs A3 -> winners meet -> winner plays LB final vs UB loser
      LB final winner -> GRAND FINAL
      Grand final: BOTH finalists PROMOTE; GF winner takes the playoff bonus.
    """
    if len(group_a) < 3 or len(group_z) < 3:
        raise ValueError("each group needs at least 3 ranked teams")
    log: list[dict] = []

    ub_winner, ub_loser = _series(play, "ub", group_a[0], group_z[0], log)
    lb1a_w, _ = _series(play, "lb_r1", group_a[1], group_z[2], log)   # A2 vs Z3
    lb1b_w, _ = _series(play, "lb_r1", group_z[1], group_a[2], log)   # Z2 vs A3
    lb2_w, _ = _series(play, "lb_r2", lb1a_w, lb1b_w, log)
    lbf_w, _ = _series(play, "lb_final", lb2_w, ub_loser, log)
    champion, _ = _series(play, "grand_final", ub_winner, lbf_w, log)

    return {
        "league": "lower",
        "promoted": [ub_winner, lbf_w],       # both grand finalists
        "grand_finalists": [ub_winner, lbf_w],
        "champion": champion,
        "series": log,
    }


# --------------------------------------------------------------------------- #
# Upper league playoff — TEMPORARY STUB (no promotion; relegation by standings)
# --------------------------------------------------------------------------- #
def upper_playoff(seeds: list[str], play: PlayFn) -> dict:
    """``seeds`` are standings-ordered (index 0 = 1st).

    Stub bracket (per spec):
      UB:  1 vs 2 -> winner to GRAND FINAL, loser to LB final
      LB:  3 vs 4 -> winner to LB final
      LB final: LB winner vs UB loser -> winner to GRAND FINAL
      Grand final -> champion (playoff bonus).
    """
    if len(seeds) < 4:
        raise ValueError("upper playoff needs at least 4 seeds")
    log: list[dict] = []

    ub_winner, ub_loser = _series(play, "ub", seeds[0], seeds[1], log)
    lb_winner, _ = _series(play, "lb", seeds[2], seeds[3], log)
    lbf_w, _ = _series(play, "lb_final", lb_winner, ub_loser, log)
    champion, _ = _series(play, "grand_final", ub_winner, lbf_w, log)

    return {
        "league": "upper",
        "grand_finalists": [ub_winner, lbf_w],
        "champion": champion,
        "series": log,
    }
