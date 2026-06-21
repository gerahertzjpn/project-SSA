"""Play-once-then-reveal season engine — 1 day = 1 game; spoiler-free reveal; finalise parity."""
import json
import pathlib

import pytest

from src.world import live_season as LS
from src.world import season, views
from src.world.world import new_world

FRANCHISES = [{"franchise_id": f"fr{i:02d}", "name": f"Team {i:02d}", "abbrev": f"T{i:02d}"}
              for i in range(22)]
N_REGULAR = 75      # 45 upper (10-team RR) + 15 + 15 lower groups, ONE game per day
N_DAYS = 85         # + 4 upper-playoff + 6 lower-playoff series, one per day


def _manifest():
    p = pathlib.Path("population/frozen_population_v1.json")
    if not p.exists():
        pytest.skip("frozen population manifest not present")
    return json.loads(p.read_text(encoding="utf-8"))


def _bootstrapped(seed=5):
    w = new_world("w-live", FRANCHISES, seed=seed, population=_manifest())
    season.bootstrap_world(w, seed=seed)
    return w


def _played(world, *, seed=11):
    LS.play_and_store(world, play=season.make_play_fn(world, seed=seed), seed=seed)
    return world


def test_schedule_rounds_each_team_plays_once_per_round_and_all_pairs():
    ids = [f"t{i}" for i in range(10)]
    rounds = LS.schedule_rounds(ids)
    assert len(rounds) == 9
    for rnd in rounds:
        seen = [x for pair in rnd for x in pair]
        assert len(seen) == len(set(seen)) == 10
    assert len({frozenset(p) for rnd in rounds for p in rnd}) == 45


def test_play_and_store_one_game_per_day_no_canonical_mutation():
    w = _bootstrapped()
    elo_before = {tid: t["elo"] for tid, t in w["teams"].items()}
    cal_before, hist_before = len(w.get("calendar", [])), len(w.get("history", []))
    _played(w)
    rep = w["live_replay"]
    assert rep["cursor"] == 0
    assert rep["n_regular_days"] == N_REGULAR and rep["n_days"] == N_DAYS
    assert all(len(d) and "series" in d for d in rep["days"])        # one series per day
    assert w["phase"] == "live"
    # fully played up front, but nothing canonical changed yet (other tabs stay spoiler-free)
    assert {tid: t["elo"] for tid, t in w["teams"].items()} == elo_before
    assert len(w.get("calendar", [])) == cal_before
    assert len(w.get("history", [])) == hist_before


def test_position_stats_do_not_leak_during_reveal():
    w = _bootstrapped()
    def _games(world):
        return sum(v.get("games", 0) for p in world["players"].values()
                   for v in (p.get("position_stats") or {}).values())
    before = _games(w)
    _played(w)
    assert _games(w) == before
    while LS.is_live(w):
        LS.reveal_next_day(w)
    assert _games(w) > before


def test_view_one_game_per_day_and_spoiler_free():
    w = _played(_bootstrapped())
    v0 = LS.view(w)
    assert v0["cursor"] == 0 and v0["day"] == 1 and v0["total_days"] == N_DAYS
    assert len(v0["next_fixtures"]) == 1                          # ONE game per day
    assert "winner" not in v0["next_fixtures"][0]                 # upcoming = matchup only
    assert sum(r["points"] for r in v0["standings"]["upper"]) == 0
    assert v0["bracket"]["upper"] == []
    assert "cores" not in json.dumps(v0)
    LS.reveal_next_day(w)
    v1 = LS.view(w)
    assert v1["cursor"] == 1 and v1["day"] == 2
    assert len(v1["just_played"]) == 1
    assert sum(r["points"] for r in v1["standings"]["upper"]) == 3   # exactly one series revealed


def test_schedule_is_open_from_day_zero():
    # The full regular fixture list is visible from the start (who plays whom on which day),
    # results only for revealed days — that's what a schedule is.
    w = _played(_bootstrapped())
    sched0 = views.schedule_view(w)["regular"]
    assert len(sched0) == N_REGULAR                               # all fixtures shown up front
    assert all(not m["played"] for m in sched0)                  # ...but none with a result yet
    assert all(m["day"] >= 1 for m in sched0)                    # each fixture has its day
    LS.reveal_next_day(w)
    sched1 = views.schedule_view(w)["regular"]
    assert len(sched1) == N_REGULAR                               # still the whole list
    assert sum(1 for m in sched1 if m["played"]) == 1            # exactly one result now


def test_full_reveal_finalizes_with_correct_invariants():
    w = _bootstrapped()
    before = len(w.get("history", []))
    _played(w)
    days = 0
    last = None
    while LS.is_live(w):
        last = LS.reveal_next_day(w)
        days += 1
        assert days <= N_DAYS + 2
    assert days == N_DAYS and last["finished"] is True
    assert w["phase"] == "offseason_due" and not LS.is_live(w)
    h = w["history"][-1]
    assert len(w["history"]) == before + 1
    assert len(h["promoted"]) == 2 and len(h["relegated"]) == 2
    assert sum(r["points"] for r in h["standings"]["upper"].values()) == 135
    assert sum(r["points"] for r in h["standings"]["lower_A"].values()) == 45
    assert sum(r["points"] for r in h["standings"]["lower_Z"].values()) == 45
    played = [e for e in w["calendar"] if e["season"] == h["season"] and e["played"]]
    assert len(played) == N_DAYS                                  # 75 group + 10 playoff series


def test_bracket_reveals_progressively_in_playoffs():
    w = _played(_bootstrapped())
    while LS.view(w)["phase"] == "regular":
        LS.reveal_next_day(w)
    assert LS.view(w)["phase"] == "playoff" and LS.view(w)["bracket"]["upper"]
    assert all(not n["played"] for n in LS.view(w)["bracket"]["upper"])
    LS.reveal_next_day(w)                                         # one playoff game
    played_now = [n for n in LS.view(w)["bracket"]["upper"] if n["played"]]
    assert len(played_now) == 1                                   # exactly one playoff series revealed
    assert {n["key"]: n for n in LS.view(w)["bracket"]["upper"]}["gf"]["played"] is False


def test_all_read_models_track_the_revealed_day():
    w = _played(_bootstrapped())
    assert sum(r["points"] for r in views.league_tables(w)["upper"]) == 0
    assert all(c is None for g in views.league_matrix(w)["groups"] for row in g["cells"] for c in row)
    LS.reveal_next_day(w)
    assert sum(r["points"] for r in views.league_tables(w)["upper"]) == 3   # one series
    while LS.is_live(w):
        LS.reveal_next_day(w)
    assert sum(r["points"] for r in views.league_tables(w)["upper"]) == 135


def test_watch_returns_a_map_sequence_for_the_next_game():
    w = _played(_bootstrapped(seed=6), seed=7)
    fx = LS.view(w)["next_fixtures"][0]
    res = LS.reveal_watch(w, fx["home"], fx["away"])
    assert res["maps_home"] + res["maps_away"] >= 2
    assert len(res["sequence"]) == res["maps_home"] + res["maps_away"]
    assert "cores" not in json.dumps(res)


def test_watch_rejects_a_non_next_fixture():
    w = _played(_bootstrapped(seed=8), seed=9)
    nxt = (LS.view(w)["next_fixtures"][0]["home"], LS.view(w)["next_fixtures"][0]["away"])
    teams = list(w["teams"])
    bogus = next((a, b) for a in teams for b in teams if a != b and (a, b) != nxt)
    with pytest.raises(ValueError):
        LS.reveal_watch(w, *bogus)
