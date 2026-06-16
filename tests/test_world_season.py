"""World foundation: brackets, calendar, season orchestration, Season-0 bootstrap."""
from src.world import brackets, calendar as cal, season
from src.world.world import lower_teams, new_world, upper_teams

FRANCHISES = [{"franchise_id": f"fr{i:02d}", "name": f"Team {i}", "abbrev": f"T{i:02d}"}
              for i in range(22)]


def _priority_play(order):
    """A deterministic ``play`` where the team earlier in ``order`` always wins 2-0."""
    rank = {tid: i for i, tid in enumerate(order)}

    def play(home, away, best_of):
        need = best_of // 2 + 1
        if rank.get(home, 1e9) < rank.get(away, 1e9):
            return {"home": home, "away": away, "maps_home": need, "maps_away": 0, "winner": home}
        return {"home": home, "away": away, "maps_home": 0, "maps_away": need, "winner": away}
    return play


# -------------------------------------------------------------- brackets --- #
def test_standings_orders_by_wins_then_mapdiff():
    teams = ["x", "y", "z"]
    results = [
        {"home": "x", "away": "y", "maps_home": 2, "maps_away": 0, "winner": "x"},
        {"home": "x", "away": "z", "maps_home": 2, "maps_away": 1, "winner": "x"},
        {"home": "y", "away": "z", "maps_home": 2, "maps_away": 0, "winner": "y"},
    ]
    rows = brackets.standings(teams, results)
    assert [r["team_id"] for r in rows] == ["x", "y", "z"]
    assert rows[0]["place"] == 1


def test_lower_playoff_promotes_two_grand_finalists():
    a = ["A1", "A2", "A3", "A4", "A5", "A6"]
    z = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]
    play = _priority_play(a + z)  # all A teams outrank all Z teams
    res = brackets.lower_playoff(a, z, play)
    assert len(res["promoted"]) == 2
    assert res["promoted"] == res["grand_finalists"]
    assert res["champion"] in res["promoted"]
    # grand final + ub + 2x lb_r1 + lb_r2 + lb_final = 6 series
    assert len(res["series"]) == 6


def test_upper_playoff_stub_resolves_champion():
    seeds = [f"U{i}" for i in range(10)]
    play = _priority_play(seeds)
    res = brackets.upper_playoff(seeds, play)
    assert res["champion"] in res["grand_finalists"]
    assert len(res["series"]) == 4  # ub + lb + lb_final + grand_final


# -------------------------------------------------------------- calendar --- #
def test_calendar_records_and_skips_month():
    w = new_world("w-cal", FRANCHISES, seed=1)
    cal.add_match(w, season=1, stage="ub", league="upper", home="a", away="b",
                  best_of=5, maps_home=3, maps_away=1, winner="a")
    assert cal.results_table(w, season=1)
    start = w["month"]
    cal.skip_month(w, 1)
    assert w["month"] == start + 1


# ----------------------------------------------------------- bootstrap ----- #
def test_bootstrap_fills_leagues_and_seeds_season_one():
    w = new_world("w-boot", FRANCHISES, seed=11)
    season.bootstrap_world(w, seed=11)
    assert len(upper_teams(w)) == 10
    assert len(lower_teams(w)) == 12
    groups = {"A": 0, "Z": 0}
    for t in lower_teams(w):
        groups[t["group"]] += 1
    assert groups == {"A": 6, "Z": 6}
    # every team drafted to >=3
    assert all(len(t["roster"]) >= 3 for t in w["teams"].values())
    # Season 0 persists ONLY SoloQ: no money, no salaries, no calendar history, Elo reset
    assert all(t["elo"] == 1000.0 and t["elo_history"] == [] for t in w["teams"].values())
    assert w["calendar"] == []
    assert all(p["salary"] == 0.0 for p in w["players"].values())
    # ready for season 1 with seeded treasuries
    assert w["season"] == 1 and w["phase"] == "regular"
    assert all(t["treasury"] == 600 for t in upper_teams(w))
    assert all(t["treasury"] == 400 for t in lower_teams(w))
    assert w["bootstrap"]["upper_filled"] is True


def test_bootstrap_deterministic():
    # same world_id + same seed -> identical outcome (ids/seeding are stable)
    w1 = new_world("w-det", FRANCHISES, seed=20)
    w2 = new_world("w-det", FRANCHISES, seed=20)
    season.bootstrap_world(w1, seed=20)
    season.bootstrap_world(w2, seed=20)
    up1 = {t["franchise_id"] for t in upper_teams(w1)}
    up2 = {t["franchise_id"] for t in upper_teams(w2)}
    assert up1 == up2


# ------------------------------------------------------- regular season ---- #
def test_regular_season_keeps_league_sizes_and_settles():
    w = new_world("w-reg", FRANCHISES, seed=12)
    season.bootstrap_world(w, seed=12)
    summary = season.run_regular_season(w)
    # promotion/relegation conserves league sizes
    assert len(upper_teams(w)) == 10
    assert len(lower_teams(w)) == 12
    assert len(summary["promoted"]) == 2 and len(summary["relegated"]) == 2
    # promoted came from lower, relegated were 9th/10th upper
    assert summary["relegated"] == summary["upper_ranking"][8:10]
    # economy ran: a champion banked a prize, calendar populated, history grew
    assert w["calendar"]
    assert w["history"]
    champ = w["teams"][summary["upper_champion"]]
    assert champ["titles"] >= 1
    # Elo moved from official matches
    assert any(t["elo"] != 1000.0 for t in w["teams"].values())


def test_two_seasons_run():
    w = new_world("w-2s", FRANCHISES, seed=13)
    season.bootstrap_world(w, seed=13)
    s1 = season.run_regular_season(w)
    w["season"] = 2
    cal.skip_month(w, 1)
    s2 = season.run_regular_season(w)
    assert s1["season"] == 1 and s2["season"] == 2
    assert len(upper_teams(w)) == 10 and len(lower_teams(w)) == 12
