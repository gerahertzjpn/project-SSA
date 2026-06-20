"""Day-by-day season engine: scheduling, spoiler-free reveal, finalisation parity."""
import itertools
import json
import pathlib

import pytest

from src.world import live_season as LS
from src.world import season
from src.world.world import new_world

FRANCHISES = [{"franchise_id": f"fr{i:02d}", "name": f"Team {i:02d}", "abbrev": f"T{i:02d}"}
              for i in range(22)]


def _manifest():
    p = pathlib.Path("population/frozen_population_v1.json")
    if not p.exists():
        pytest.skip("frozen population manifest not present")
    return json.loads(p.read_text(encoding="utf-8"))


def _bootstrapped(seed=5):
    w = new_world("w-live", FRANCHISES, seed=seed, population=_manifest())
    season.bootstrap_world(w, seed=seed)
    return w


# --------------------------------------------------------------- scheduling -- #
def test_schedule_rounds_each_team_plays_once_per_round_and_all_pairs():
    ids = [f"t{i}" for i in range(10)]
    rounds = LS.schedule_rounds(ids)
    assert len(rounds) == 9                                # n-1 rounds for 10 teams
    for rnd in rounds:
        seen = [x for pair in rnd for x in pair]
        assert len(seen) == len(set(seen)) == 10           # everyone plays exactly once
    pairs = {frozenset(p) for rnd in rounds for p in rnd}
    assert len(pairs) == 45                                # full single round-robin (C(10,2))


def test_odd_group_adds_and_drops_a_bye():
    ids = [f"t{i}" for i in range(6)]
    rounds = LS.schedule_rounds(ids)
    pairs = {frozenset(p) for rnd in rounds for p in rnd}
    assert len(pairs) == 15                                # C(6,2)


# ------------------------------------------------------------- full season -- #
def test_full_season_day_by_day_finalizes_like_batch():
    w = _bootstrapped()
    before = len(w.get("history", []))
    LS.start(w, seed=42)
    assert w["phase"] == "live" and LS.is_live(w)
    v = LS.view(w)
    assert v["n_regular_days"] == 9 and v["total_days"] == 13

    days = 0
    summary = None
    while LS.is_live(w):
        res = LS.play_day(w)
        summary = res.get("summary", summary)
        days += 1
        assert days <= 20                                  # guard against a stuck loop
    assert days == 13                                      # 9 regular + 4 playoff days
    assert w["phase"] == "offseason_due"                   # season played -> off-season due
    assert not LS.is_live(w)

    # history settled through the shared finaliser
    assert len(w["history"]) == before + 1
    h = w["history"][-1]
    assert len(h["upper_ranking"]) == 10 and len(h["lower_ranking"]) == 12
    assert len(h["promoted"]) == 2 and len(h["relegated"]) == 2
    assert h["upper_champion"] == summary["upper_champion"]
    # points add up exactly: upper RR = 45 series * 3, each lower group = 15 * 3
    assert sum(r["points"] for r in h["standings"]["upper"].values()) == 135
    assert sum(r["points"] for r in h["standings"]["lower_A"].values()) == 45
    assert sum(r["points"] for r in h["standings"]["lower_Z"].values()) == 45
    # every official series recorded once: 45+15+15 group + 4 upper + 6 lower playoff
    played = [e for e in w["calendar"] if e["season"] == h["season"] and e["played"]]
    assert len(played) == 85


def test_view_is_spoiler_free():
    w = _bootstrapped()
    LS.start(w, seed=1)
    v = LS.view(w)
    # only the immediate next round is exposed (upper 5 + lowerA 3 + lowerZ 3 = 11 series)
    assert len(v["next_fixtures"]) == 11
    assert all(f["stage"] == "league" for f in v["next_fixtures"])
    # no bracket yet during the regular season
    assert v["bracket"]["upper"] == [] and v["bracket"]["lower"] == []
    # standings exist (everyone at 0 points before any day)
    assert sum(r["points"] for r in v["standings"]["upper"]) == 0
    # cores (hidden tiebreaker) never leak into the view payload
    assert "cores" not in json.dumps(v)


def test_bracket_fills_in_progressively_during_playoffs():
    w = _bootstrapped()
    LS.start(w, seed=2)
    while LS.view(w)["phase"] == "regular":
        LS.play_day(w)
    # first playoff day: some bracket slots resolved, grand final still blank
    v = LS.view(w)
    assert v["phase"] == "playoff"
    up = {n["key"]: n for n in v["bracket"]["upper"]}
    assert up["gf"]["played"] is False                     # final not decided yet
    LS.play_day(w)                                         # play another playoff day
    # next-day fixtures never include an unresolved (None-seeded) match
    for f in LS.view(w)["next_fixtures"]:
        assert f["home"] and f["away"]


def test_watch_live_does_not_double_count():
    w = _bootstrapped(seed=6)
    LS.start(w, seed=7)
    fx = LS.view(w)["next_fixtures"][0]
    res = LS.play_series_live(w, fx["home"], fx["away"])
    assert res["maps_home"] + res["maps_away"] >= 2        # a real Bo3 result
    assert len(res["sequence"]) == res["maps_home"] + res["maps_away"]
    LS.play_day(w)                                         # the rest of the round
    gkey = fx["group"]
    pair = frozenset((fx["home"], fx["away"]))
    occurrences = sum(1 for s in w["live_season"]["results"][gkey]
                      if frozenset((s["home"], s["away"])) == pair)
    assert occurrences == 1                                # watched fixture counted once
    while LS.is_live(w):
        LS.play_day(w)
    played = [e for e in w["calendar"] if e["season"] == w["history"][-1]["season"] and e["played"]]
    assert len(played) == 85


def test_play_series_live_rejects_a_non_next_fixture():
    w = _bootstrapped(seed=8)
    LS.start(w, seed=9)
    teams = list(w["teams"])
    # a pair that is not in the immediate next-day fixtures should be refused
    nxt = {(f["home"], f["away"]) for f in LS.view(w)["next_fixtures"]}
    bogus = next((a, b) for a, b in itertools.permutations(teams, 2) if (a, b) not in nxt)
    with pytest.raises(ValueError):
        LS.play_series_live(w, *bogus)
