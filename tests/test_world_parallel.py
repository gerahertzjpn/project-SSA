"""Parallel match execution is BIT-IDENTICAL to sequential.

The regular season and each SoloQ round are independent matches keyed by PRE-ASSIGNED seeds, so a
parallel run (process pool) must produce exactly the same season/ratings as a sequential one. These
tests prove the order-independence of the seed assignment + result fold-back using deterministic,
picklable fake players (no torch / no real nets). ``run_batch`` falls back to sequential on any pool
error, so the requested-parallel run is always at least as correct as the sequential one.
"""
import hashlib

from src.world import live_season, season, soloq
from src.world.parallel import run_batch
from src.world.world import new_world

FRANCHISES = [{"franchise_id": f"fr{i:02d}", "name": f"Team {i}", "abbrev": f"T{i:02d}"}
              for i in range(22)]


class FakeSeriesPlayer:
    """Picklable Bo-N player: outcome depends ONLY on (home, away, best_of, seed)."""
    def __call__(self, home, away, best_of, series_seed):
        need = best_of // 2 + 1
        h = int.from_bytes(hashlib.blake2b(f"{home}|{away}|{series_seed}".encode(),
                                           digest_size=8).digest(), "little")
        home_wins = (h % 2 == 0)
        loser = h % need
        mh, ma = (need, loser) if home_wins else (loser, need)
        winner = home if home_wins else away
        return {"home": home, "away": away, "maps_home": mh, "maps_away": ma, "winner": winner,
                "best_of": best_of, "sequence": [winner] * (mh + ma),
                "map_results": [], "map_lineups": []}


class FakeSoloQPlayer:
    """Picklable SoloQ player: winner depends ONLY on the pre-assigned match seed."""
    def __call__(self, team_a, team_b, match_seed):
        winner = "A" if (match_seed % 2 == 0) else "B"
        return winner, {pid: 0.5 for pid in list(team_a) + list(team_b)}


def _square(x):                      # top-level (picklable) for the run_batch order test
    return x * x


def _ls_snapshot(ls):
    return [(s["gid"], s["home"], s["away"], s["maps_home"], s["maps_away"], s["winner"],
             s.get("win_prob"), s.get("stage"), s.get("group"), s.get("league"), s.get("day"))
            for d in ls["days"] for s in d["series"]]


def test_run_batch_preserves_order_and_matches_sequential():
    tasks = [(i,) for i in range(12)]
    seq = [_square(i) for i in range(12)]
    assert run_batch(_square, tasks, workers=1) == seq
    assert run_batch(_square, tasks, workers=4) == seq      # parallel (or safe fallback) == sequential


def test_play_and_store_parallel_equals_sequential():
    player = FakeSeriesPlayer()
    snaps = []
    for w in (1, 3):
        wd = new_world("w-par", FRANCHISES, seed=5)
        season.bootstrap_world(wd, seed=5)
        ls = live_season.play_and_store(wd, series_player=player, seed=99, workers=w)
        snaps.append(_ls_snapshot(ls))
    assert snaps[0] == snaps[1]
    assert len(snaps[0]) > 50                                # a full 22-team season was played + stored


def test_soloq_parallel_equals_sequential():
    player = FakeSoloQPlayer()
    ratings = {f"p{i:03d}": 1000.0 + (i % 7) * 30 for i in range(60)}
    r1, log1 = soloq.simulate_soloq(ratings, rounds=4, seed=7, match_player=player, workers=1, record=True)
    r2, log2 = soloq.simulate_soloq(ratings, rounds=4, seed=7, match_player=player, workers=3, record=True)
    assert r1 == r2
    assert [e["matches"] for e in log1] == [e["matches"] for e in log2]
