"""Off-season orchestration + training preview + feature gating + 800k floor."""
import pytest

from src.world import (
    backends, feature_flags, freeagents, offseason, rookies, season,
    training_auction, training_ops,
)
from src.world.world import new_world

FRANCHISES = [{"franchise_id": f"fr{i:02d}", "name": f"Team {i}", "abbrev": f"T{i:02d}"}
              for i in range(22)]


# ----------------------------------------------------------- feature flags- #
def test_flags_locked_and_marked_in_development():
    for name in ("real_training_backend", "llm_manager", "ui_world_tab"):
        st = feature_flags.ui_state(name)
        assert st["locked"] is True and st["selectable"] is False
        assert st["status"] == "in_development"
    # the optional L2 live auction is a secondary, locked-by-default feature
    assert feature_flags.flag("live_auction_l2")["enabled"] is False


def test_assert_enabled_raises_when_locked():
    with pytest.raises(feature_flags.FeatureLocked):
        feature_flags.assert_enabled("real_training_backend")


# ------------------------------------------------------------- backend lock- #
def test_real_backend_is_wired_but_locked():
    be = backends.get_backend()
    for method, ctx in (("fork", {"target": {"kind": "parent"}}),
                        ("continue_ppo", {"target": {"kind": "player"}})):
        with pytest.raises(feature_flags.FeatureLocked):
            getattr(be, method)({**ctx, "total_steps": 1_000_000})


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
    w = _bootstrapped()
    jobs = [{"team_id": list(w["teams"])[0], "spend_type": "skill_up",
             "target": {"kind": "player", "player_id": "p"}, "total_steps": 1_000_000}]
    rep = offseason.run_offseason(w, training_jobs=jobs, backend=backends.get_backend(), seed=1)
    exe = rep["stages"]["training"]["execution"]
    assert exe["executed"] is False
    assert exe["results"][0]["status"] == "locked"   # FeatureLocked caught, reported honestly
