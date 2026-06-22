"""Play-once-then-reveal season engine.

Per-game reveal model: the regular season runs 2 games per day, the playoffs 1 (upper bracket
fully, THEN lower). Reveal is per GAME (``revealed_gids``) — a single match can be watched and
committed on its own, and every read-model updates the moment it is. Finalise parity with the batch
path holds once the whole season is revealed.
"""
import json
import pathlib

import pytest

from src.world import live_season as LS
from src.world import season, views
from src.world.world import new_world

FRANCHISES = [{"franchise_id": f"fr{i:02d}", "name": f"Team {i:02d}", "abbrev": f"T{i:02d}"}
              for i in range(22)]
N_REGULAR_GAMES = 75    # 45 upper (10-team RR) + 15 + 15 lower groups
N_REGULAR_DAYS = 38     # 2 games per day -> ceil(75/2)
N_PLAYOFF_DAYS = 10     # 4 upper + 6 lower series, ONE per day
N_DAYS = N_REGULAR_DAYS + N_PLAYOFF_DAYS    # 48
N_SERIES = N_REGULAR_GAMES + N_PLAYOFF_DAYS  # 85 series recorded onto the calendar


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


def test_play_and_store_two_games_per_day_no_canonical_mutation():
    w = _bootstrapped()
    elo_before = {tid: t["elo"] for tid, t in w["teams"].items()}
    cal_before, hist_before = len(w.get("calendar", [])), len(w.get("history", []))
    _played(w)
    rep = w["live_replay"]
    assert rep["revealed_gids"] == []
    assert rep["n_regular_days"] == N_REGULAR_DAYS and rep["n_days"] == N_DAYS
    # regular days hold up to 2 games; playoff days exactly 1
    reg_days = rep["days"][:N_REGULAR_DAYS]
    pl_days = rep["days"][N_REGULAR_DAYS:]
    assert all(d["phase"] == "regular" and 1 <= len(d["series"]) <= 2 for d in reg_days)
    assert sum(len(d["series"]) for d in reg_days) == N_REGULAR_GAMES
    assert all(d["phase"] == "playoff" and len(d["series"]) == 1 for d in pl_days)
    # every game has a unique gid spanning the whole season
    gids = [s["gid"] for d in rep["days"] for s in d["series"]]
    assert sorted(gids) == list(range(N_SERIES))
    assert w["phase"] == "live"
    # fully played up front, but nothing canonical changed yet (other tabs stay spoiler-free)
    assert {tid: t["elo"] for tid, t in w["teams"].items()} == elo_before
    assert len(w.get("calendar", [])) == cal_before
    assert len(w.get("history", [])) == hist_before


def test_playoff_days_are_upper_bracket_first_then_lower():
    w = _played(_bootstrapped())
    rep = w["live_replay"]
    leagues = [d["series"][0]["league"] for d in rep["days"][N_REGULAR_DAYS:]]
    assert leagues == ["upper"] * 4 + ["lower"] * 6        # all upper days, then all lower


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


def test_view_two_cards_per_regular_day_and_spoiler_free():
    w = _played(_bootstrapped())
    v0 = LS.view(w)
    assert v0["cursor"] == 0 and v0["day"] == 1 and v0["total_days"] == N_DAYS
    assert v0["phase"] == "regular"
    assert len(v0["day_matches"]) == 2                           # TWO games per regular day
    assert all(not c["revealed"] for c in v0["day_matches"])     # upcoming = matchup only
    assert all(c.get("league_label") for c in v0["day_matches"]) # league label present on the card
    assert sum(r["points"] for r in v0["standings"]["upper"]) == 0
    assert v0["bracket"]["upper"] == []
    assert "cores" not in json.dumps(v0)
    LS.reveal_next_day(w)
    v1 = LS.view(w)
    assert v1["cursor"] == 1 and v1["day"] == 2
    # day 0's two games are both upper-league round-1 fixtures -> 2 series = 6 points
    assert sum(r["points"] for r in v1["standings"]["upper"]) == 6


def test_lower_bracket_seeds_when_lower_regular_done_even_if_upper_isnt():
    # the LOWER playoff seeds the moment the LOWER regular is done, without waiting for the UPPER
    # league to finish (and vice-versa) — per-league gating, not all-or-nothing (#1).
    w = _played(_bootstrapped())
    ls = w["live_replay"]
    found = None
    for _ in range(ls["n_regular_days"]):
        LS.reveal_next_day(w)
        rev = LS.revealed(w)
        if rev and rev["regular_done_by_league"]["lower"] and not rev["regular_done_by_league"]["upper"]:
            found = rev
            break
    assert found is not None, "expected a window where lower regular is done but upper isn't"
    sv = views.schedule_view(w)
    lo_seeds = [m for col in sv["bracket"]["lower"] for m in col["matches"] if not (m.get("from") or [])]
    up_seeds = [m for col in sv["bracket"]["upper"] for m in col["matches"] if not (m.get("from") or [])]
    assert lo_seeds and all(m["home_abbrev"] != "TBD" for m in lo_seeds)   # lower seeds REVEALED
    assert up_seeds and all(m["home_abbrev"] == "TBD" for m in up_seeds)   # upper still hidden (spoiler-safe)


def test_view_day_navigation_rewinds_spoiler_free():
    # reveal several days, then rewind: a past day shows finished RESULTS (no broadcast) and never
    # leaks games from later days (standings frozen at that day's end).
    w = _played(_bootstrapped())
    for _ in range(5):
        LS.reveal_next_day(w)
    cur = LS.view(w)
    assert cur["is_current_day"] and cur["viewing_day"] == 5 and cur["can_prev"] and not cur["can_next"]
    past = LS.view(w, day=1)
    assert past["viewing_day"] == 1 and past["day"] == 2
    assert not past["is_current_day"]                            # a rewound day is not "live"
    assert past["can_prev"] and past["can_next"]                 # can step both ways within played days
    assert past["day_matches"] and all(c["revealed"] for c in past["day_matches"])  # results, not a broadcast
    cur_pts = sum(r["points"] for g in cur["standings"].values() for r in g)
    past_pts = sum(r["points"] for g in past["standings"].values() for r in g)
    assert past_pts < cur_pts                                    # no future games leaked into a past day
    assert LS.view(w, day=0)["can_prev"] is False               # first day has no "previous"
    assert LS.view(w, day=999)["viewing_day"] == 5              # cannot view beyond the live cursor


def test_schedule_is_open_from_day_zero():
    # The full regular fixture list is visible from the start (who plays whom on which day),
    # results only for revealed games — that's what a schedule is.
    w = _played(_bootstrapped())
    sched0 = views.schedule_view(w)["regular"]
    assert len(sched0) == N_REGULAR_GAMES                        # all fixtures shown up front
    assert all(not m["played"] for m in sched0)                 # ...but none with a result yet
    assert all(m["day"] >= 1 for m in sched0)                   # each fixture has its day
    LS.reveal_next_day(w)
    sched1 = views.schedule_view(w)["regular"]
    assert len(sched1) == N_REGULAR_GAMES                        # still the whole list
    assert sum(1 for m in sched1 if m["played"]) == 2           # exactly one day (2 games) now


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
    assert len(played) == N_SERIES                               # 75 group + 10 playoff series


def test_bracket_reveals_progressively_in_playoffs():
    w = _played(_bootstrapped())
    while LS.view(w)["phase"] == "regular":
        LS.reveal_next_day(w)
    assert LS.view(w)["phase"] == "playoff" and LS.view(w)["bracket"]["upper"]
    assert all(not n["played"] for n in LS.view(w)["bracket"]["upper"])
    LS.reveal_next_day(w)                                        # one playoff game
    played_now = [n for n in LS.view(w)["bracket"]["upper"] if n["played"]]
    assert len(played_now) == 1                                  # exactly one playoff series revealed
    assert {n["key"]: n for n in LS.view(w)["bracket"]["upper"]}["gf"]["played"] is False


def test_all_read_models_track_the_revealed_games():
    w = _played(_bootstrapped())
    assert sum(r["points"] for r in views.league_tables(w)["upper"]) == 0
    assert all(c is None for g in views.league_matrix(w)["groups"] for row in g["cells"] for c in row)
    LS.reveal_next_day(w)
    assert sum(r["points"] for r in views.league_tables(w)["upper"]) == 6   # one day = two series
    while LS.is_live(w):
        LS.reveal_next_day(w)
    assert sum(r["points"] for r in views.league_tables(w)["upper"]) == 135


def test_watch_commits_one_game_and_updates_stats_immediately():
    w = _played(_bootstrapped(seed=6), seed=7)
    card = LS.view(w)["day_matches"][0]
    home, away = card["home"]["team_id"], card["away"]["team_id"]
    upper_before = sum(r["points"] for r in views.league_tables(w)["upper"])
    res = LS.reveal_watch(w, home, away)["result"]
    assert res["maps_home"] + res["maps_away"] >= 2
    assert len(res["sequence"]) == res["maps_home"] + res["maps_away"]
    assert "cores" not in json.dumps(res)
    # the single watched game is committed -> read-models reflect it right away, before "next day"
    after = LS.view(w)
    assert any(c["revealed"] for c in after["day_matches"])      # that card now shows a score
    assert any(not c["revealed"] for c in after["day_matches"])  # the day's OTHER game is still pending
    assert sum(r["points"] for r in views.league_tables(w)["upper"]) >= upper_before


def test_watching_all_games_does_not_advance_the_day():
    # The day advances ONLY on «Следующий день» — watching every game of a day leaves you on it.
    w = _played(_bootstrapped(seed=6), seed=7)
    v0 = LS.view(w)
    assert v0["cursor"] == 0 and len(v0["day_matches"]) == 2
    for c in list(v0["day_matches"]):
        res = LS.reveal_watch(w, c["home"]["team_id"], c["away"]["team_id"])
        assert res["finished"] is False                   # watching never finalises
    v = LS.view(w)
    assert v["cursor"] == 0                                # STILL day 0 after watching both
    assert all(c["revealed"] for c in v["day_matches"])    # ...but both now show their score
    assert LS.is_live(w)                                   # season not finished
    LS.reveal_next_day(w)                                  # only this advances
    assert LS.view(w)["cursor"] == 1


def test_server_watch_returns_unwrapped_match_for_the_broadcast():
    # The LIVE broadcast reads response["result"].sequence/maps DIRECTLY, so the server endpoint must
    # UN-wrap reveal_watch's nested {result, finished, view}. Regression guard for the double-wrap bug
    # that made the real-app broadcast render "undefined:undefined".
    import shutil
    import viewer.server as srv
    from src.world.world import save_world, world_path
    w = _played(_bootstrapped(seed=6), seed=7)
    w["world_id"] = "world-pytest-live01"
    save_world(w)
    try:
        view = srv.world_live_view_payload(w["world_id"])["live"]
        card = view["day_matches"][0]
        resp = srv.world_live_watch(w["world_id"], {"home": card["home"]["team_id"],
                                                    "away": card["away"]["team_id"]})
        assert "sequence" in resp["result"] and resp["result"]["maps_home"] is not None
        assert resp["result"]["winner"] and "day_matches" in resp["live"]      # not double-wrapped
        day = srv.world_live_reveal_day(w["world_id"])
        assert "finished" in day["result"] and "live" in day                   # /live/day shape intact
    finally:
        shutil.rmtree(world_path(w["world_id"]).parent, ignore_errors=True)


def test_watch_rejects_a_game_not_in_the_current_day():
    w = _played(_bootstrapped(seed=8), seed=9)
    current = {(c["home"]["team_id"], c["away"]["team_id"]) for c in LS.view(w)["day_matches"]}
    teams = list(w["teams"])
    bogus = next((a, b) for a in teams for b in teams if a != b and (a, b) not in current)
    with pytest.raises(ValueError):
        LS.reveal_watch(w, *bogus)
