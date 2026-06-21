"""Play-once-then-reveal season engine: scheduling, spoiler-free reveal, finalisation parity."""
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


def _played(world, *, seed=11):
    """Pre-play the whole season with the fast (deterministic) engine into the replay store."""
    LS.play_and_store(world, play=season.make_play_fn(world, seed=seed), seed=seed)
    return world


# --------------------------------------------------------------- scheduling -- #
def test_schedule_rounds_each_team_plays_once_per_round_and_all_pairs():
    ids = [f"t{i}" for i in range(10)]
    rounds = LS.schedule_rounds(ids)
    assert len(rounds) == 9
    for rnd in rounds:
        seen = [x for pair in rnd for x in pair]
        assert len(seen) == len(set(seen)) == 10
    assert len({frozenset(p) for rnd in rounds for p in rnd}) == 45


# --------------------------------------------- play up front, no canonical mutation -- #
def test_play_and_store_does_not_touch_canonical_state():
    w = _bootstrapped()
    elo_before = {tid: t["elo"] for tid, t in w["teams"].items()}
    cal_before = len(w.get("calendar", []))
    hist_before = len(w.get("history", []))
    _played(w)
    rep = w["live_replay"]
    assert rep["cursor"] == 0
    assert rep["n_regular_days"] == 9 and rep["n_days"] == 13      # 9 regular + 4 playoff
    assert w["phase"] == "live"
    # the season is fully PLAYED but nothing canonical changed yet (no spoilers in other tabs)
    assert {tid: t["elo"] for tid, t in w["teams"].items()} == elo_before
    assert len(w.get("calendar", [])) == cal_before
    assert len(w.get("history", [])) == hist_before


def test_position_stats_do_not_leak_during_reveal():
    # The up-front play records per-game position_stats; they must be held back (restored to
    # pre-season) until the reveal finishes, then applied — otherwise the Ratings tab would show
    # the whole unrevealed season's winrates from day 0.
    w = _bootstrapped()
    def _games(world):
        return sum(v.get("games", 0) for p in world["players"].values()
                   for v in (p.get("position_stats") or {}).values())
    before = _games(w)
    _played(w)
    assert _games(w) == before                       # mid-reveal: no new position games surfaced
    while LS.is_live(w):
        LS.reveal_next_day(w)
    assert _games(w) > before                        # finalised: the season's games are now counted


def test_view_is_spoiler_free_at_each_cursor():
    w = _played(_bootstrapped())
    v0 = LS.view(w)
    assert v0["cursor"] == 0 and v0["day"] == 1 and v0["total_days"] == 13
    assert len(v0["next_fixtures"]) == 11                          # upper 5 + lowerA 3 + lowerZ 3
    assert all("winner" not in f for f in v0["next_fixtures"])     # upcoming = matchup only, no result
    assert sum(r["points"] for r in v0["standings"]["upper"]) == 0  # nothing revealed yet
    assert v0["bracket"]["upper"] == []                            # playoffs not reached
    assert "cores" not in json.dumps(v0)                          # hidden tiebreaker never leaks
    # reveal one day -> standings move, fixtures advance, still no playoff bracket
    LS.reveal_next_day(w)
    v1 = LS.view(w)
    assert v1["cursor"] == 1 and v1["day"] == 2
    assert sum(r["points"] for r in v1["standings"]["upper"]) == 5 * 3   # 5 upper series this round
    assert v1["bracket"]["upper"] == []


def test_full_reveal_finalizes_with_correct_invariants():
    # Not a bit-identical re-run of run_regular_season (per-series RNG is call-order keyed), but the
    # settled season must hold the same structural invariants: 85 series, 135/45/45 points, 2 promo/releg.
    w = _bootstrapped()
    before = len(w.get("history", []))
    _played(w)
    days = 0
    last = None
    while LS.is_live(w):
        last = LS.reveal_next_day(w)
        days += 1
        assert days <= 20
    assert days == 13 and last["finished"] is True
    assert w["phase"] == "offseason_due" and not LS.is_live(w)
    # settled through the shared finaliser
    assert len(w["history"]) == before + 1
    h = w["history"][-1]
    assert len(h["upper_ranking"]) == 10 and len(h["lower_ranking"]) == 12
    assert len(h["promoted"]) == 2 and len(h["relegated"]) == 2
    assert sum(r["points"] for r in h["standings"]["upper"].values()) == 135
    assert sum(r["points"] for r in h["standings"]["lower_A"].values()) == 45
    assert sum(r["points"] for r in h["standings"]["lower_Z"].values()) == 45
    played = [e for e in w["calendar"] if e["season"] == h["season"] and e["played"]]
    assert len(played) == 85                                       # 45+15+15 group + 4 + 6 playoff


def test_bracket_reveals_progressively_in_playoffs():
    w = _played(_bootstrapped())
    while LS.view(w)["phase"] == "regular":          # reveal all regular days; next day = playoff
        LS.reveal_next_day(w)
    assert LS.view(w)["phase"] == "playoff" and LS.view(w)["bracket"]["upper"]
    assert all(not n["played"] for n in LS.view(w)["bracket"]["upper"])   # no playoff revealed yet
    LS.reveal_next_day(w)                            # reveal the first playoff day
    up = {n["key"]: n for n in LS.view(w)["bracket"]["upper"]}
    assert up["ub"]["played"] is True                # first-round series now decided
    assert up["gf"]["played"] is False               # grand final still hidden


def test_all_read_models_track_the_revealed_day():
    # Every tab (league table / H2H matrix / schedule / Power) must reflect ONLY the revealed days.
    from src.world import views
    w = _played(_bootstrapped())
    lt0 = views.league_tables(w)
    assert sum(r["points"] for r in lt0["upper"]) == 0            # day 0: nobody has points
    assert views.schedule_view(w)["regular"] == []                # nothing scheduled-as-played yet
    assert all(c is None for g in views.league_matrix(w)["groups"] for row in g["cells"] for c in row)
    LS.reveal_next_day(w)
    assert sum(r["points"] for r in views.league_tables(w)["upper"]) == 15   # 5 upper series revealed
    assert len(views.schedule_view(w)["regular"]) == 11           # day-1 fixtures now shown
    # fully revealed == the finalised numbers
    while LS.is_live(w):
        LS.reveal_next_day(w)
    assert sum(r["points"] for r in views.league_tables(w)["upper"]) == 135
    assert len(views.schedule_view(w)["regular"]) == 75


def test_watch_returns_a_map_sequence_for_a_next_fixture():
    w = _played(_bootstrapped(seed=6), seed=7)
    fx = LS.view(w)["next_fixtures"][0]
    res = LS.reveal_watch(w, fx["home"], fx["away"])
    assert res["maps_home"] + res["maps_away"] >= 2
    assert len(res["sequence"]) == res["maps_home"] + res["maps_away"]
    assert "cores" not in json.dumps(res)


def test_watch_rejects_a_non_next_fixture():
    w = _played(_bootstrapped(seed=8), seed=9)
    nxt = {(f["home"], f["away"]) for f in LS.view(w)["next_fixtures"]}
    teams = list(w["teams"])
    bogus = next((a, b) for a in teams for b in teams if a != b and (a, b) not in nxt)
    with pytest.raises(ValueError):
        LS.reveal_watch(w, *bogus)
