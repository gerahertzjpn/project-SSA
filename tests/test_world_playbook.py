"""Playbook — the manager's single tactical brain. Proves the authored choice has a
REAL effect on match outcomes (the owner's key question)."""
from src.world import playbook, season
from src.world.world import new_world

FRANCHISES = [{"franchise_id": f"fr{i:02d}", "name": f"Team {i}", "abbrev": f"T{i:02d}"}
              for i in range(4)]


# ------------------------------------------------------------- counters --- #
def test_counter_cycle_is_consistent_and_antisymmetric():
    assert playbook.counter_sign("core_rush", "macro") == 1     # rush punishes greed
    assert playbook.counter_sign("macro", "core_rush") == -1    # antisymmetric
    assert playbook.counter_sign("turtle", "all_in") == 1       # defense outlasts overcommit
    assert playbook.counter_sign("baron", "core_rush") == 1     # objective answers rush
    assert playbook.counter_sign("standard", "macro") == 0      # standard is neutral
    assert playbook.counter_sign("core_rush", "core_rush") == 0


# ----------------------------------------------------------- execution --- #
def test_player_exec_prefers_measured_then_attributes():
    rusher = {"attributes": {"attack": 95}, "strategy_effectiveness": {}}
    assert playbook.player_exec(rusher, "core_rush") > 0.9      # high attack -> runs rush
    weak = {"attributes": {"attack": 20}, "strategy_effectiveness": {}}
    assert playbook.player_exec(weak, "core_rush") < 0.2
    measured = {"attributes": {"attack": 20}, "strategy_effectiveness": {"core_rush": 0.99}}
    assert playbook.player_exec(measured, "core_rush") > 0.9    # measured wins over attrs


# ----------------------------------------------------------- resolve --- #
def _plan(base, overrides=None):
    return {"base": {"strategy": base, "lineup": None}, "overrides": overrides or []}


def test_resolve_base_when_no_override():
    a = playbook.resolve_active(_plan("turtle"), {"stage": "group"}, {})
    assert a["strategy"] == "turtle" and a["note"] == "base"


def test_resolve_opponent_conditional_substitution():
    # "opponent robot majority -> core_rush" — the headline owner example
    plan = _plan("macro", [{"when": {"opp_faction_majority": "robots"},
                            "strategy": "core_rush", "lineup": None, "note": "rush vs robots"}])
    vs_robots = playbook.resolve_active(plan, {"stage": "group"}, {"faction_majority": "robots"})
    assert vs_robots["strategy"] == "core_rush"
    vs_humans = playbook.resolve_active(plan, {"stage": "group"}, {"faction_majority": "humans"})
    assert vs_humans["strategy"] == "macro"            # base when condition not met


def test_resolve_playoff_contingencies():
    plan = _plan("standard", [
        {"when": {"stage": "playoff", "playoff_losing_streak": 2}, "strategy": "all_in", "note": "x"},
        {"when": {"stage": "playoff", "series_ahead": True}, "strategy": "turtle", "note": "y"}])
    behind = playbook.resolve_active(plan, {"stage": "playoff", "playoff_losing_streak": 2}, {})
    assert behind["strategy"] == "all_in"
    ahead = playbook.resolve_active(plan, {"stage": "playoff", "series_won": 1, "series_lost": 0}, {})
    assert ahead["strategy"] == "turtle"


# ------------------------------------------------------------- EFFECT --- #
def _equal_two_team_world():
    """A world where teams A and B have IDENTICAL rosters (same SoloQ, attrs, faction)
    so the ONLY thing that can move a result is the playbook."""
    w = new_world("w-pb", FRANCHISES, seed=1)
    tids = list(w["teams"])[:2]
    pids = list(w["players"])
    idx = 0
    for tid in tids:
        w["teams"][tid]["roster"] = []
        for _ in range(3):
            p = w["players"][pids[idx]]; idx += 1
            p["soloq_rating"] = 1000.0
            p["faction"] = "humans"                    # avoid the robot-counter override
            p["attributes"] = {k: 50 for k in p["attributes"]}
            p["status"], p["team_id"] = "rostered", tid
            w["teams"][tid]["roster"].append(p["player_id"])
    return w, tids[0], tids[1]


def _winrate(w, home, away, n=400, seed=42):
    play = season.make_play_fn(w, seed=seed)           # explicit seed -> reproducible
    return sum(1 for _ in range(n) if play(home, away, 1)["winner"] == home) / n


def test_counter_strategy_meaningfully_raises_winrate():
    w, a, b = _equal_two_team_world()
    # control: identical rosters, identical (standard) strategy -> coin flip
    playbook.set_plan(w, a, _plan("standard"))
    playbook.set_plan(w, b, _plan("standard"))
    base = _winrate(w, a, b)
    assert 0.42 < base < 0.58                          # no edge from anything but variance

    # A authors a hard counter (core_rush) to B's macro -> A wins much more
    playbook.set_plan(w, a, _plan("core_rush"))
    playbook.set_plan(w, b, _plan("macro"))
    countered = _winrate(w, a, b)
    assert countered > 0.60                            # the choice REALLY changes outcomes
    assert countered - base > 0.12


def test_being_countered_lowers_winrate_symmetrically():
    w, a, b = _equal_two_team_world()
    playbook.set_plan(w, a, _plan("macro"))            # A is now the one being countered
    playbook.set_plan(w, b, _plan("core_rush"))
    assert _winrate(w, a, b) < 0.40                    # symmetric disadvantage


def test_matchup_meta_reports_what_each_side_runs():
    w, a, b = _equal_two_team_world()
    playbook.set_plan(w, a, _plan("core_rush"))
    playbook.set_plan(w, b, _plan("macro"))
    dh, da, meta = playbook.matchup_modifiers(w, a, b, best_of=1)
    assert meta["home"]["strategy"] == "core_rush" and meta["away"]["strategy"] == "macro"
    assert meta["counter"] == 1 and dh > da            # home counters away


def test_default_plan_built_when_none_set():
    w, a, b = _equal_two_team_world()
    # no plan set on either -> get_plan builds a heuristic default, sim still works
    assert w["teams"][a]["tournament_plan"] is None
    dh, da, meta = playbook.matchup_modifiers(w, a, b, best_of=3)
    assert meta["home"]["strategy"] in playbook.STRATEGY_IDS
