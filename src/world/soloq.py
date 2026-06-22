"""SoloQ — the player rating mechanic.

Players' ratings are no longer driven by their team's official wins/losses;
they come from SoloQ: a simulated ladder run before the season. Rules:

- 10 rounds. Each round players are grouped into rating **bands of step 100**
  (band = floor(rating/100)*100); a 900-1000 player never shares a match with a
  1000-1100 player.
- Within a band, random 3v3 teams are formed, but the two teams' **average
  ratings are balanced** as far as possible (greedy). Matches are **Bo1**.
- A player with no room (band has < 6, or the leftover after grouping) **skips**.
- Rating change is a chess-Elo delta vs the enemy team's average, modulated by
  the player's in-game **impact up to ±60%** ("up to 60% of the win points").

The match outcome is provided by a pluggable ``match_fn`` so SoloQ runs with no
trained nets (default = a rating-based stochastic model). Everything is seeded.
"""
from __future__ import annotations

import random
from typing import Callable, Optional

from src.regionals.elo import expected_score

TEAM_SIZE = 3
BAND_STEP = 100
DEFAULT_ROUNDS = 10
K_FACTOR = 32.0
SOLOQ_MATCH_CAP = 100       # keep the most recent N SoloQ games per player (bootstrap plays ~80)
IMPACT_SPREAD = 0.35  # ±35% impact modulation — the Elo win/loss stays dominant (was ±60%, too strong)

# match_fn(team_a_ids, team_b_ids, ratings, rng) -> (winner: "A"|"B", impacts)
#   impacts: {player_id: float in [0,1]} (0.5 neutral).
MatchFn = Callable[[list, list, dict, random.Random], tuple]


def band_of(rating: float) -> int:
    return int(rating // BAND_STEP) * BAND_STEP


def soloq_rating_delta(
    rating: float,
    opponent_avg: float,
    won: bool,
    impact: float = 0.5,
    k: float = K_FACTOR,
    spread: float = IMPACT_SPREAD,
) -> float:
    """Elo delta vs the enemy team average, modulated by impact up to ±spread.

    ``impact`` in [0,1] (0.5 neutral): high impact amplifies a win / softens a
    loss; low impact does the opposite. The win/loss term stays dominant.
    """
    expected = expected_score(rating, opponent_avg)
    actual = 1.0 if won else 0.0
    base = k * (actual - expected)
    impact = max(0.0, min(1.0, impact))
    modulation = 1.0 + spread * (2.0 * impact - 1.0)  # 1±spread
    return base * modulation


def _balanced_two_teams(group: list, ratings: dict) -> tuple[list, list]:
    """Split 6 players into two teams of 3 with the closest average ratings
    (greedy: assign rating-desc players to the currently-lighter team)."""
    ordered = sorted(group, key=lambda pid: -ratings[pid])
    a: list = []
    b: list = []
    sum_a = sum_b = 0.0
    for pid in ordered:
        if len(a) < TEAM_SIZE and (sum_a <= sum_b or len(b) >= TEAM_SIZE):
            a.append(pid)
            sum_a += ratings[pid]
        else:
            b.append(pid)
            sum_b += ratings[pid]
    return a, b


def _default_match_fn(team_a, team_b, ratings, rng) -> tuple:
    """Rating-based stochastic outcome + per-player impact, all seeded."""
    avg_a = sum(ratings[p] for p in team_a) / len(team_a)
    avg_b = sum(ratings[p] for p in team_b) / len(team_b)
    p_a = expected_score(avg_a, avg_b)
    winner = "A" if rng.random() < p_a else "B"
    win_team = team_a if winner == "A" else team_b
    impacts = {}
    for pid in team_a + team_b:
        base = rng.uniform(0.25, 0.75)
        if pid in win_team:
            base = min(1.0, base + 0.15)  # carries skew higher on the winning side
        impacts[pid] = base
    return winner, impacts


def _round_matches(ratings: dict, rng: random.Random) -> list[tuple[list, list]]:
    """Form this round's matches: per band, shuffle and chunk into 6, balance
    each chunk into two teams. Leftovers (and bands < 6) skip the round."""
    bands: dict[int, list] = {}
    for pid, r in ratings.items():
        bands.setdefault(band_of(r), []).append(pid)

    matches: list[tuple[list, list]] = []
    for band in sorted(bands):
        members = bands[band]
        rng.shuffle(members)
        full = (len(members) // (2 * TEAM_SIZE)) * (2 * TEAM_SIZE)
        for i in range(0, full, 2 * TEAM_SIZE):
            chunk = members[i:i + 2 * TEAM_SIZE]
            matches.append(_balanced_two_teams(chunk, ratings))
    return matches


def _match_seed(base_seed: int, rnd: int, idx: int) -> int:
    """Deterministic, ORDER-INDEPENDENT per-match seed from (base_seed, round, match index). Used for
    the parallel path so match i always gets the same seed regardless of how many ran before it."""
    import hashlib
    h = hashlib.blake2b(f"{base_seed}:{rnd}:{idx}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") % (2 ** 31 - 1) + 1


def simulate_soloq(
    ratings: dict,
    *,
    rounds: int = DEFAULT_ROUNDS,
    seed: int = 0,
    match_fn: Optional[MatchFn] = None,
    record: bool = False,
    match_player=None,
    workers: int = 1,
) -> tuple[dict, list[dict]]:
    """Run SoloQ over ``ratings`` (pid -> rating). Returns (new_ratings, log).

    Pure: does not mutate the input dict. ``log`` is per-round stats (matches played, who skipped).
    With ``record=True`` each round entry also carries ``player_matches`` = {pid: {teammates,
    opponents, won, delta}} so the world can build a per-player game history.

    Speed (safe): within ONE round every match has DISJOINT teams and deltas are computed off the
    PRE-round ratings, so the matches are independent. When a picklable, seeded ``match_player``
    (``net_match.SoloQSeriesPlayer``) is supplied, a round's matches are played with PRE-ASSIGNED
    per-(round,index) seeds — over a process pool when ``workers > 1`` — then deltas are applied
    sequentially in match order. Bit-identical to sequential. ROUNDS stay sequential (round r+1's
    matchmaking bands read round r's updated ratings)."""
    from src.world import parallel

    match_fn = match_fn or _default_match_fn
    rng = random.Random(seed)
    cur = dict(ratings)
    log: list[dict] = []

    for rnd in range(rounds):
        matches = _round_matches(cur, rng)
        played = {pid for m in matches for pid in (m[0] + m[1])}
        pm: dict[str, dict] = {}
        # PLAY the round's matches: parallel via the seeded player, else the sequential shared-rng fn.
        if match_player is not None:
            tasks = [(list(ta), list(tb), _match_seed(seed, rnd, i)) for i, (ta, tb) in enumerate(matches)]
            outcomes = parallel.run_batch(match_player, tasks, workers)
        else:
            outcomes = [match_fn(ta, tb, cur, rng) for ta, tb in matches]
        # APPLY in match order: teams are disjoint within a round, so deltas off the pre-round snapshot
        # are order-independent (identical to applying inline).
        for (team_a, team_b), (winner, impacts) in zip(matches, outcomes):
            avg_a = sum(cur[p] for p in team_a) / len(team_a)
            avg_b = sum(cur[p] for p in team_b) / len(team_b)
            # All 6 are in the SAME rating band (strict 100-step matchmaking). Snapshot their ratings
            # AT MATCH TIME so the match detail stays band-consistent even after ratings drift later.
            at_time = {p: round(cur[p]) for p in team_a + team_b}
            mband = band_of(cur[team_a[0]])          # the matchmaking band (floor) — shared by all 6
            a_won = winner == "A"
            # compute every player's delta off the PRE-match ratings, then apply (teams are disjoint)
            deltas = {pid: round(soloq_rating_delta(cur[pid], avg_b, a_won, impacts.get(pid, 0.5)), 2)
                      for pid in team_a}
            deltas.update({pid: round(soloq_rating_delta(cur[pid], avg_a, not a_won, impacts.get(pid, 0.5)), 2)
                           for pid in team_b})
            for pid in team_a + team_b:
                cur[pid] += deltas[pid]
            if record:                                  # store ALL 6 ratings+deltas in each entry so the
                for pid in team_a:                      # detail page resolves every player (round ids differ)
                    pm[pid] = {"teammates": [p for p in team_a if p != pid], "opponents": list(team_b),
                               "won": a_won, "delta": deltas[pid], "ratings": at_time, "deltas": deltas,
                               "band": mband}
                for pid in team_b:
                    pm[pid] = {"teammates": [p for p in team_b if p != pid], "opponents": list(team_a),
                               "won": not a_won, "delta": deltas[pid], "ratings": at_time, "deltas": deltas,
                               "band": mband}
        entry = {"round": rnd, "matches": len(matches), "skipped": sorted(set(cur) - played)}
        if record:
            entry["player_matches"] = pm
        log.append(entry)
    # Round for stability/serialization.
    cur = {pid: round(r, 2) for pid, r in cur.items()}
    return cur, log


def apply_soloq(world: dict, *, rounds: int = DEFAULT_ROUNDS, seed: int = 0,
                match_fn: Optional[MatchFn] = None, match_player=None, workers: int = 1) -> list[dict]:
    """Run SoloQ over the world's players and write back ratings + history.
    Returns the per-round log. Only SoloQ persists during Season 0.

    ``match_player`` (seeded, picklable) + ``workers > 1`` play each round's matches in parallel."""
    players = world["players"]
    # Reserves are held OUT of season 0 (not yet debuted) -> they don't play SoloQ until introduced.
    ratings = {pid: p["soloq_rating"] for pid, p in players.items() if p.get("status") != "reserve"}
    new_ratings, log = simulate_soloq(ratings, rounds=rounds, seed=seed, match_fn=match_fn, record=True,
                                      match_player=match_player, workers=workers)
    season = world.get("season", 0)
    for pid, r in new_ratings.items():
        p = players[pid]
        p["soloq_rating"] = r
        p["soloq_history"].append(r)
    # per-match history (with/against whom + points delta) — capped to the recent games. The stored
    # round is offset past the player's already-recorded games for this season, so several
    # pre-seasons in one Season 0 (the bootstrap loop) keep (season, round) UNIQUE — the SoloQ
    # match-detail page navigates by that key. (The returned log keeps its 0..N-1 numbering.)
    base = {pid: 1 + max((x["round"] for x in (players[pid].get("soloq_matches") or [])
                          if x.get("season") == season), default=-1)
            for pid in new_ratings}
    for entry in log:
        for pid, m in (entry.get("player_matches") or {}).items():
            players[pid].setdefault("soloq_matches", []).append(
                {"season": season, "round": base.get(pid, 0) + entry["round"], **m})
    for pid in new_ratings:
        sm = players[pid].get("soloq_matches") or []
        if len(sm) > SOLOQ_MATCH_CAP:
            players[pid]["soloq_matches"] = sm[-SOLOQ_MATCH_CAP:]
    return log
