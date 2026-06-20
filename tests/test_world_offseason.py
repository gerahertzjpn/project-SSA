"""Off-season orchestration + training preview + feature gating + 800k floor."""
import contextlib
import pytest

from src.world import (
    backends, feature_flags, freeagents, offseason, rookies, season,
    training_auction, training_ops,
)
from src.world.world import new_world

FRANCHISES = [{"franchise_id": f"fr{i:02d}", "name": f"Team {i}", "abbrev": f"T{i:02d}"}
              for i in range(22)]


@contextlib.contextmanager
def _flag(name, value):
    orig = feature_flags.FLAGS[name]["enabled"]
    feature_flags.FLAGS[name]["enabled"] = value
    try:
        yield
    finally:
        feature_flags.FLAGS[name]["enabled"] = orig


# ----------------------------------------------------------- feature flags- #
def test_all_features_unlocked():
    # Owner flipped everything ON ("включи всё"): all four flags are live/selectable.
    for name in ("real_training_backend", "llm_manager", "ui_world_tab", "live_auction_l2"):
        st = feature_flags.ui_state(name)
        assert st["locked"] is False and st["selectable"] is True
    feature_flags.assert_enabled("real_training_backend")   # no longer raises


# ------------------------------------------------------------- backend wired #
def test_real_backend_training_ops_state():
    be = backends.get_backend(require_flag=False)   # bypass the dev gate for the validation check
    # All ops are REAL now; they validate input first (missing target/net_ids -> ValueError) so
    # the unit test never kicks off training. continue_ppo / fork / evaluate need a net_id;
    # cotrain needs net_ids.
    for method in ("continue_ppo", "fork", "evaluate"):
        with pytest.raises(ValueError):
            getattr(be, method)({"target": {"kind": "player"}})
    with pytest.raises(ValueError):
        be.cotrain({"target": {"kind": "team"}})        # no net_ids


def test_manager_training_decisions_are_net_bound_and_weakest_first():
    import json
    import pathlib
    from src.world import manager
    mpath = pathlib.Path("population/frozen_population_v1.json")
    if not mpath.exists():
        pytest.skip("frozen population manifest not present")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    w = new_world("w-train", FRANCHISES, seed=7, population=manifest)
    season.bootstrap_world(w, seed=7)
    season.run_regular_season(w)
    jobs = manager.run_training_decisions(w)
    assert jobs, "managers should buy some training on a net-bound roster"
    assert all(j["spend_type"] in training_ops.ENABLED_OPS for j in jobs)
    # the backend must be able to resolve which net(s) to train
    for j in jobs:
        t = j["target"]
        assert ("net_id" in t) or ("net_ids" in t)
        assert j["total_steps"] >= training_ops.MIN_TRAINING_STEPS
    pr = [j["priority"] for j in jobs]
    assert pr == sorted(pr, reverse=True)        # weakest teams first


def test_train_one_net_builds_a_valid_faction_triple():
    # train_one_net must assemble one sparring partner per OTHER faction so LeagueTrainer._triples
    # yields exactly one faction-balanced triple (the bug that made 1-3 nets produce 0 plans).
    from src.training.population_roster import build_population, FACTIONS
    pop = build_population()
    target = next(d for d in pop if d.net_id == "humans_1__skirmisher")
    partners = []
    for fac in FACTIONS:
        if fac == target.faction:
            continue
        partners.append(next(d for d in pop if d.faction == fac and d.net_id != target.net_id))
    descendants = [target, *partners]
    by_fac = {f: [d.net_id for d in descendants if d.faction == f] for f in FACTIONS}
    assert all(len(by_fac[f]) >= 1 for f in FACTIONS)    # -> _triples() returns one triple, not zero


# ------------------------------------------------------------- 800k floor -- #
def test_absolute_min_steps_floor():
    with pytest.raises(ValueError):
        training_ops.validate_request("skill_up", {"kind": "player"}, total_steps=500_000)
    # at the floor it is accepted
    training_ops.validate_request("skill_up", {"kind": "player"}, total_steps=800_000)


def test_all_cards_respect_floor():
    for op in training_ops.ENABLED_OPS:
        assert training_ops.card(op)["min_steps"] >= training_ops.MIN_TRAINING_STEPS


# ------------------------------------------------------- training preview --- #
def test_training_preview_reports_who_what_eta():
    cfg = training_auction.TrainingConfig(steps_per_slot=1_000_000, throughput_steps_per_min=50_000)
    jobs = [
        {"team_id": "TA", "spend_type": "skill_up",
         "target": {"kind": "player", "player_id": "p1"}, "total_steps": 1_000_000},
        {"team_id": "TB", "spend_type": "create_player",
         "target": {"kind": "parent", "parent_id": "p9"}, "total_steps": 4_000_000},
    ]
    pv = offseason.training_preview(jobs, cfg, extra_minutes=30)
    assert pv["n_requests"] == 2 and pv["total_steps"] == 5_000_000
    # 5M / 50k = 100 min + 30 extra
    assert pv["total_minutes"] == 130.0
    assert pv["requests"][0]["est_minutes"] == 20.0
    assert any("skill_up" in line for line in offseason.render_training_preview(pv))


# ----------------------------------------------------------- run offseason - #
def _bootstrapped():
    w = new_world("w-off", FRANCHISES, seed=11)
    season.bootstrap_world(w, seed=11)
    season.run_regular_season(w)        # play a season so there is history
    return w


def test_run_offseason_sequences_stages_and_advances():
    w = _bootstrapped()
    before_season = w["season"]
    before_month = w["month"]
    rep = offseason.run_offseason(w, rookie_budget=rookies.RookieBudget(n_rookies=3), seed=11)
    assert set(rep["stages"]) == set(offseason.OFFSEASON_STAGES)
    assert rep["stages"]["rookies"]["n"] == 3
    assert w["season"] == before_season + 1         # advanced
    assert w["month"] == before_month + 1           # off-season month skipped
    # training previewed but NOT executed (no backend supplied)
    assert rep["stages"]["training"]["execution"]["executed"] is False


def test_combined_launch_merges_team_rookies_and_free_agents():
    w = _bootstrapped()
    team_jobs = [{"team_id": list(w["teams"])[0], "spend_type": "skill_up",
                  "target": {"kind": "player", "player_id": "p"}, "total_steps": 1_000_000}]
    rep = offseason.run_offseason(
        w, training_jobs=team_jobs,
        rookie_budget=rookies.RookieBudget(n_rookies=2, steps_per_rookie=1_000_000),
        fa_budget=freeagents.FreeAgentBudget(steps_per_agent=800_000, max_agents=3),
        seed=5)
    by_kind = rep["stages"]["training"]["preview"]["by_kind"]
    assert by_kind["team"]["n"] == 1
    assert by_kind["rookie"]["n"] == 2
    assert by_kind["free_agent"]["n"] == 3            # all three sources in ONE launch
    assert rep["stages"]["free_agents"]["n"] == 3


def test_free_agents_owner_selected_and_optional():
    w = _bootstrapped()
    fas = freeagents.free_agents(w)
    assert len(fas) >= 2
    # explicit owner choice is honored exactly
    chosen = fas[:2]
    jobs = freeagents.maintenance_jobs(w, freeagents.FreeAgentBudget(agent_ids=chosen))
    assert {j["target"]["player_id"] for j in jobs} == set(chosen)
    # no explicit list and no cap -> owner opted out -> nobody trains
    assert freeagents.maintenance_jobs(w, freeagents.FreeAgentBudget()) == []
    # fully optional: no budget at all -> stage trains nobody
    rep = offseason.run_offseason(w, seed=1)
    assert rep["stages"]["free_agents"]["n"] == 0


def test_free_agent_budget_smaller_than_rookies():
    # owner estimate: total free-agent compute < total rookie compute
    w = _bootstrapped()
    rk = rookies.RookieBudget(n_rookies=8, steps_per_rookie=1_000_000)
    fa = freeagents.FreeAgentBudget(steps_per_agent=800_000, max_agents=3)
    fa_jobs = freeagents.maintenance_jobs(w, fa)
    assert freeagents.total_steps(fa_jobs) < rk.total_steps()


def test_run_offseason_with_locked_backend_records_locked():
    with _flag("real_training_backend", False):       # admin-locked training path still works
        w = _bootstrapped()
        jobs = [{"team_id": list(w["teams"])[0], "spend_type": "skill_up",
                 "target": {"kind": "player", "player_id": "p"}, "total_steps": 1_000_000}]
        rep = offseason.run_offseason(w, training_jobs=jobs,
                                      backend=backends.get_backend(require_flag=True), seed=1)
        exe = rep["stages"]["training"]["execution"]
        assert exe["executed"] is False
        assert exe["results"][0]["status"] == "locked"   # FeatureLocked caught, reported honestly
