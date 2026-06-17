"""Positions, scouting/intel (B), measured counter matrix (C), recorded calls (A),
per-opponent conditions (D) — the data-grounding layer."""
from src.world import calendar as cal, playbook, scouting, season
from src.world.world import get_player, new_world

FRANCHISES = [{"franchise_id": f"fr{i:02d}", "name": f"Team {i}", "abbrev": f"T{i:02d}"}
              for i in range(22)]


def _played_world():
    w = new_world("w-sc", FRANCHISES, seed=4)
    season.bootstrap_world(w, seed=4)
    season.run_regular_season(w, seed=4)     # populates calendar (calls) + position_stats
    return w


# --------------------------------------------------------------- positions -- #
def test_position_affinity_in_schema():
    w = new_world("w-pos", FRANCHISES, seed=1)
    p = next(iter(w["players"].values()))
    assert set(p["position_affinity"]) == {"A1", "A2", "A3"}
    assert p["position_stats"] == {}
    assert max(p["position_affinity"].values()) > 1.0   # clearly best at one position


def test_lineup_strength_depends_on_placement():
    w = new_world("w-pl", FRANCHISES, seed=1)
    tid = list(w["teams"])[0]
    pids = list(w["players"])[:3]
    for pid in pids:
        w["players"][pid].update(status="rostered", team_id=tid, soloq_rating=1000.0)
        w["teams"][tid]["roster"].append(pid)
    # make pid0 a clear A1 specialist, pid1 a clear A3 specialist
    w["players"][pids[0]]["position_affinity"] = {"A1": 1.2, "A2": 0.9, "A3": 0.85}
    w["players"][pids[1]]["position_affinity"] = {"A1": 0.85, "A2": 0.9, "A3": 1.2}
    w["players"][pids[2]]["position_affinity"] = {"A1": 1.0, "A2": 1.0, "A3": 1.0}
    good = {"A1": pids[0], "A2": pids[2], "A3": pids[1]}   # specialists on their spots
    bad = {"A1": pids[1], "A2": pids[2], "A3": pids[0]}    # swapped -> off-position
    assert playbook.lineup_strength(w, good) > playbook.lineup_strength(w, bad)
    # the optimal default finds the good placement
    assert playbook.optimal_lineup(w, tid) == good


def test_record_position_results_accumulates():
    w = new_world("w-rec", FRANCHISES, seed=1)
    pids = list(w["players"])[:3]
    lineup = {"A1": pids[0], "A2": pids[1], "A3": pids[2]}
    playbook.record_position_results(w, lineup, won=True)
    playbook.record_position_results(w, lineup, won=False)
    st = get_player(w, pids[0])["position_stats"]["A1"]
    assert st == {"games": 2, "wins": 1}
    assert playbook.position_winrate(get_player(w, pids[0]), "A1") == 0.5


# --------------------------------------------------- A: recorded calls ------ #
def test_calendar_records_strategy_calls():
    w = _played_world()
    played = [e for e in w["calendar"] if e["played"]]
    assert played
    # every played official match recorded BOTH managers' calls (strategies)
    assert all(e["home_strategy"] in playbook.STRATEGY_IDS
               and e["away_strategy"] in playbook.STRATEGY_IDS for e in played)


def test_bootstrap_does_not_record_positions():
    # Season 0 persists only SoloQ -> no position_stats yet
    w = new_world("w-b", FRANCHISES, seed=2)
    season.bootstrap_world(w, seed=2)
    assert all(not get_player(w, pid).get("position_stats")
               for pid in w["players"])


def test_regular_season_records_positions():
    w = _played_world()
    assert any(get_player(w, pid).get("position_stats") for pid in w["players"])


# --------------------------------------------------- B: scouting intel ------ #
def test_intel_report_has_history_after_a_season():
    w = _played_world()
    tid = list(w["teams"])[0]
    rep = scouting.intel_report(w, tid, opponents=list(w["teams"])[1:4])
    assert rep["league_meta"]                       # leaguewide strategy winrates
    assert rep["own_strategy_record"]               # the team's own past calls
    text = scouting.render_intel(rep)
    assert "Мета лиги" in text and text != "истории игр пока нет (первый сезон)"


def test_league_meta_winrates_are_consistent():
    w = _played_world()
    meta = scouting.league_meta(w)
    for r in meta.values():
        assert 0.0 <= r["winrate"] <= 1.0 and r["games"] > 0


# --------------------------------------------------- C: measured matrix ----- #
def test_measured_winrate_shrinks_from_prior_toward_data():
    w = new_world("w-m", FRANCHISES, seed=1)
    # no games yet -> pure designed prior (core_rush counters macro ~0.585)
    p0, n0 = playbook.measured_winrate(w, "core_rush", "macro")
    assert n0 == 0 and abs(p0 - playbook.PRIOR_WIN_COUNTER) < 1e-9
    # inject lopsided history: macro beats core_rush 30x -> measured pulls DOWN
    for i in range(30):
        cal.add_match(w, season=1, stage="g", league="upper", home="A", away="B",
                      best_of=3, maps_home=0, maps_away=2, winner="B",
                      home_strategy="core_rush", away_strategy="macro")
    p1, n1 = playbook.measured_winrate(w, "core_rush", "macro")
    assert n1 == 30 and p1 < p0                      # data overrides the prior
    # antisymmetry of the rating points
    assert abs(playbook.counter_points(w, "core_rush", "macro")
               + playbook.counter_points(w, "macro", "core_rush")) < 1e-6


# --------------------------------------------------- D: per-opp conditions -- #
def test_opp_last_strategy_and_vs_team_conditions():
    plan = playbook._plan if hasattr(playbook, "_plan") else None  # noqa: F841
    base = {"base": {"strategy": "standard", "lineup": None}, "overrides": [
        {"when": {"opp_last_strategy": "macro"}, "strategy": "core_rush", "note": "x"},
        {"when": {"vs_team": "rivals"}, "strategy": "turtle", "note": "y"}]}
    a = playbook.resolve_active(base, {"stage": "group"}, {"last_strategy": "macro"})
    assert a["strategy"] == "core_rush"
    b = playbook.resolve_active(base, {"stage": "group"}, {"team_id": "rivals"})
    assert b["strategy"] == "turtle"
    c = playbook.resolve_active(base, {"stage": "group"}, {"last_strategy": "turtle"})
    assert c["strategy"] == "standard"               # base when nothing matches


def test_opponent_context_exposes_last_call():
    w = _played_world()
    tid = list(w["teams"])[0]
    ctx = playbook.opponent_context(w, tid)
    assert "last_strategy" in ctx                    # what they actually called last
    assert ctx["last_strategy"] in (None, *playbook.STRATEGY_IDS)
