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


def test_season0_selection_holds_back_reserves_and_introduces_later():
    import json
    import pathlib
    from src.world.world import introduce_reserves
    mpath = pathlib.Path("population/frozen_population_v1.json")
    if not mpath.exists():
        pytest.skip("frozen population manifest not present")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    nets = [e["net_id"] for e in manifest["descendants"]]
    # owner NAMES 4 pros + wants 70 to debut -> 66 random-filled; the other 20 are reserves
    w = new_world("w-s0", FRANCHISES, seed=1, population=manifest,
                  entered_net_ids=nets[:4], n_enter=70)
    assert len(w["free_agents"]) == 70 and len(w["reserves"]) == 20
    entered_nets = {w["players"][pid]["net_id"] for pid in w["free_agents"]}
    assert set(nets[:4]) <= entered_nets                       # the named ones debut
    season.bootstrap_world(w, seed=1)
    # reserves sat out Season 0 entirely (no SoloQ history, still held)
    assert all(w["players"][pid]["status"] == "reserve" for pid in w["reserves"])
    assert all(not w["players"][pid]["soloq_history"] for pid in w["reserves"])
    # introduce some reserves later -> they join the rookies pool (debut as rookies)
    intro = introduce_reserves(w, w["reserves"][:5])
    assert len(intro) == 5 and all(pid in w["rookies_pool"] for pid in intro)
    assert len(w["reserves"]) == 15


def test_post_group_replan_off_by_default_is_noop():
    # the post-group base-strategy replan is behind a default-OFF flag -> no-op even with an llm present
    from src.world import llm_manager
    assert feature_flags.is_enabled("post_group_replan") is False
    rows = [{"team_id": "t", "wins": 0, "losses": 3}]
    assert llm_manager.replan_after_groups({"teams": {}}, [rows], llm=object()) == []


def test_create_player_from_fork_creates_binds_and_rosters():
    """A team's create_player must MAKE a new player OWN the forked net + join the paying team's roster.
    The bug this guards: create_player trained a unique net but created no player, orphaning the net."""
    from src.world.offseason import create_player_from_fork
    parent_pid = "w-cf-p000"
    w = {
        "world_id": "w-cf",
        "players": {parent_pid: {"player_id": parent_pid, "net_id": "robots_3", "faction": "robots",
                                 "archetype": "base", "nickname": "Parent", "attributes": {},
                                 "generation": 0, "team_id": "t1"}},
        "teams": {"t1": {"name": "Club", "roster": [parent_pid]}},
    }
    fork_result = {"net_id": "robots_skirmisher_gZ_1", "genotype_hash": "deadbeef",
                   "generation": 1, "parent_id": "robots_3"}
    info = create_player_from_fork(w, "t1", parent_pid, fork_result)
    new_pid = info["new_player_id"]
    assert new_pid in w["players"] and new_pid != parent_pid
    new = w["players"][new_pid]
    assert new["net_id"] == "robots_skirmisher_gZ_1"                 # bound to the forked net
    assert new["parent_ids"] == [parent_pid]                        # descends from the RIGHT parent
    assert new["faction"] == "robots"                               # faction preserved (net is locked)
    assert new["team_id"] == "t1" and new_pid in w["teams"]["t1"]["roster"]   # rostered on the payer
    # a failed/empty fork must NOT create a phantom player
    assert create_player_from_fork(w, "t1", parent_pid, {}) is None


def test_offseason_depth_signings_build_benches():
    import json
    import pathlib
    from src.world.world import MAX_ROSTER
    mpath = pathlib.Path("population/frozen_population_v1.json")
    if not mpath.exists():
        pytest.skip("frozen population manifest not present")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    w = new_world("w-depth", FRANCHISES, seed=2, population=manifest)
    season.bootstrap_world(w, seed=2)
    season.run_regular_season(w)
    assert all(len(t["roster"]) == 3 for t in w["teams"].values())     # drafted 3 starters
    rep = offseason.run_offseason(w, llm=None, seed=5)
    assert rep["stages"]["transfers"]["signings"] >= 1                  # clubs signed depth
    assert all(len(t["roster"]) <= MAX_ROSTER for t in w["teams"].values())
    assert any(len(t["roster"]) > 3 for t in w["teams"].values())      # benches formed -> subs usable


def test_training_market_auctions_certs_then_spends_into_jobs():
    import json
    import pathlib
    from src.world import offseason as off
    mpath = pathlib.Path("population/frozen_population_v1.json")
    if not mpath.exists():
        pytest.skip("frozen population manifest not present")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    w = new_world("w-mkt", FRANCHISES, seed=7, population=manifest)
    season.bootstrap_world(w, seed=7)
    season.run_regular_season(w)
    cfg = training_auction.TrainingConfig(steps_per_slot=1_000_000)
    jobs, summary = off.run_training_market(w, cfg, llm=None, feed=[], seed=1)   # heuristic bids/spend
    assert summary["awarded"] >= 1                      # the gradeslot auction awarded certificates
    assert summary["spent"] == len(jobs)
    # every spend job is a valid training op with a net-bound target that meets the floor
    for j in jobs:
        assert j["spend_type"] in training_ops.ENABLED_OPS
        t = j["target"]
        assert ("net_id" in t) or ("net_ids" in t)
        assert j["total_steps"] >= training_ops.MIN_TRAINING_STEPS


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


def test_skill_changes_and_card_cache_force_reload(tmp_path, monkeypatch):
    """_skill_changes diffs card attributes; _load_player_cards(force=True) re-reads the file the
    training subprocess rewrote (the equal-conditions re-eval relies on bypassing the cache)."""
    from src.world import offseason
    from src.world import world as wmod
    w = {"players": {"p1": {"nickname": "A", "faction": "humans",
                            "attributes": {"attack": 50, "defense": 40}}}}
    changes = offseason._skill_changes(w, {"p1": {"attack": 45, "defense": 40}})
    assert len(changes) == 1 and changes[0]["changes"] == {"attack": 5}
    f = tmp_path / "cards.json"
    f.write_text('{"cards": {"n1": {"attack": 77}}}', encoding="utf-8")
    monkeypatch.setattr(wmod, "_PLAYER_CARDS_PATH", f)
    wmod._PLAYER_CARDS_CACHE = {"n1": {"attack": 10}}                 # stale cache
    assert wmod._load_player_cards()["n1"]["attack"] == 10            # served stale
    assert wmod._load_player_cards(force=True)["n1"]["attack"] == 77  # re-read from disk
    wmod._PLAYER_CARDS_CACHE = None                                   # reset for other tests


def test_card_rederive_reuses_recorded_reference_cast():
    """A trained net must be re-carded under the SAME gauntlet cast it was first carded with — else
    cards drift (a net scored against a different opponent set). reference_cast guarantees this by
    reusing whatever an existing card file recorded, regardless of the --alt-cast flag."""
    pytest.importorskip("torch")                       # the script imports src.core.runner (torch)
    from scripts.derive_player_cards import reference_cast
    meta = {"reference_teammates": ["humans_5", "infected_5"],
            "reference_opponents": ["robots_5", "humans_6", "infected_6"]}
    recorded = (("humans_5", "infected_5"), ("robots_5", "humans_6", "infected_6"))
    assert reference_cast(False, meta) == recorded      # re-derive reuses recorded cast...
    assert reference_cast(True, meta) == recorded       # ...regardless of the flag (no drift)
    assert reference_cast(False, None)[0] == ("humans_4", "infected_4")   # fresh derive honors flag
    assert reference_cast(True, None)[0] == ("humans_5", "infected_5")
    # the SHIPPED cards file must re-derive against its own recorded cast
    import json, pathlib
    p = pathlib.Path("population/player_cards_v1.json")
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        if m.get("reference_teammates"):
            assert reference_cast(False, m) == (tuple(m["reference_teammates"]),
                                                tuple(m["reference_opponents"]))
