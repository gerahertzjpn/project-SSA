"""Tests for the 18-net population self-play training system."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.bots.policy_bot import cached_policy_state, checkpoint_is_compatible
from src.core.constants import FACTION_TYPES
from src.training.population import (
    NETS_PER_FACTION,
    DEFAULT_RECIPES,
    PopulationConfig,
    PopulationTrainer,
    NetSpec,
    assert_population_unique,
    build_default_population,
    build_episode_plans,
    make_net,
    parameter_count,
    write_frozen_pool,
)
from src.training.team_env import TeamTrainingEnv, TeamTrainingEnvConfig


# --------------------------------------------------------------------------- #
# Population spec + uniqueness
# --------------------------------------------------------------------------- #
def test_population_has_18_unique_nets_six_per_faction():
    nets = build_default_population()
    assert len(nets) == 18
    for faction in FACTION_TYPES:
        assert sum(1 for net in nets if net.faction == faction) == NETS_PER_FACTION
    assert len({net.net_id for net in nets}) == 18
    assert len({net.seed for net in nets}) == 18
    assert len({net.genotype_hash() for net in nets}) == 18
    assert_population_unique(nets)  # does not raise


def test_genotype_hash_is_stable_and_sensitive():
    nets = build_default_population()
    base = nets[0]
    # Stable for an identical spec.
    same = NetSpec(**{**base.__dict__})
    assert same.genotype_hash() == base.genotype_hash()
    # Sensitive to faction, recipe and seed independently.
    assert NetSpec(**{**base.__dict__, "faction": "robots"}).genotype_hash() != base.genotype_hash()
    assert NetSpec(**{**base.__dict__, "seed": base.seed + 1}).genotype_hash() != base.genotype_hash()
    assert NetSpec(**{**base.__dict__, "entropy_coef": base.entropy_coef + 0.001}).genotype_hash() != base.genotype_hash()


def test_assert_population_unique_rejects_duplicate_genotype():
    nets = build_default_population()
    # Same genotype as nets[0] but a fresh, unique net_id: must still be rejected
    # because one seed alone (a unique id) is not genuine uniqueness.
    clone = NetSpec(**{**nets[0].__dict__, "net_id": "dup_clone_1"})
    assert clone.genotype_hash() == nets[0].genotype_hash()
    duped = nets + [clone]
    try:
        assert_population_unique(duped)
    except ValueError:
        return
    raise AssertionError("Expected duplicate genotype to be rejected")


def test_make_net_is_deterministic_per_seed_and_differs_across_seeds():
    nets = build_default_population()
    a = make_net(nets[0])
    b = make_net(nets[0])
    assert all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))
    c = make_net(nets[1])
    assert any(not torch.equal(pa, pc) for pa, pc in zip(a.parameters(), c.parameters()))
    assert parameter_count(a) > 0


# --------------------------------------------------------------------------- #
# Faction wiring in the training env
# --------------------------------------------------------------------------- #
def test_team_env_assigns_per_player_factions():
    cfg = TeamTrainingEnvConfig(
        controlled_players=("A1", "A2", "A3"),
        opponent_bot="random",
        seed=7,
        player_factions={"A1": "humans", "A2": "infected", "A3": "robots",
                         "B1": "robots", "B2": "humans", "B3": "infected"},
    )
    env = TeamTrainingEnv(cfg)
    env.reset(seed=7)
    assert env.state.players["A1"].faction == "humans"
    assert env.state.players["A2"].faction == "infected"
    assert env.state.players["A3"].faction == "robots"
    # Starting rosters differ by faction.
    assert "VANGUARD" in env.state.nodes["A1"].units
    assert "SWARMLING" in env.state.nodes["A2"].units
    assert "WALKER" in env.state.nodes["A3"].units


def test_team_env_defaults_to_humans_without_factions():
    env = TeamTrainingEnv(TeamTrainingEnvConfig(controlled_players=("A1", "A2", "A3"), opponent_bot="random"))
    env.reset(seed=1)
    assert all(env.state.players[pid].faction == "humans" for pid in ("A1", "A2", "A3", "B1", "B2", "B3"))


# --------------------------------------------------------------------------- #
# Frozen pool
# --------------------------------------------------------------------------- #
def test_frozen_pool_round_trips_each_net(tmp_path: Path):
    nets = build_default_population()[:4]
    specs = {spec.net_id: spec for spec in nets}
    states = {spec.net_id: make_net(spec).state_dict() for spec in nets}
    pool_path = tmp_path / "frozen_pool_v1.pt"
    write_frozen_pool(pool_path, specs, states, update=1)
    assert checkpoint_is_compatible(pool_path)
    for spec in nets:
        state, hidden = cached_policy_state(pool_path, "B1", policy_state_id=spec.net_id)
        assert hidden == spec.hidden_size
        assert state["body.0.weight"].shape[1] == 5201  # OBSERVATION_DIM (v4 orders)


# --------------------------------------------------------------------------- #
# Scheduling coverage
# --------------------------------------------------------------------------- #
def test_episode_plans_cover_every_net_as_controlled():
    config = PopulationConfig(run_dir="unused", nets=build_default_population(),
                              episodes_per_net_per_update=3, workers=1)
    plans = build_episode_plans(config, update=1)
    counts: dict[str, int] = {net.net_id: 0 for net in config.nets}
    for plan in plans:
        assert len(plan.controlled) == 3
        assert len(plan.opponents) == 3
        assert len(set(plan.opponents)) == 3  # opponents distinct within an episode
        for net_id in plan.controlled:
            counts[net_id] += 1
    assert all(count == 3 for count in counts.values())


# --------------------------------------------------------------------------- #
# End-to-end smoke (single process, tiny)
# --------------------------------------------------------------------------- #
def test_population_trainer_one_update_smoke(tmp_path: Path):
    config = PopulationConfig(
        run_dir=str(tmp_path / "pop_smoke"),
        nets=build_default_population(),
        master_seed=123,
        updates=1,
        episodes_per_net_per_update=1,
        workers=1,
        ppo_epochs=1,
        minibatch_size=256,
        max_ticks=40,
        snapshot_interval=1,
        checkpoint_interval=1,
    )
    trainer = PopulationTrainer(config)
    trainer.save_all(update=0, milestone=False)
    pool_path = trainer.refresh_frozen_pool(version=1)
    summary = trainer.run_update(update=1, frozen_pool_path=pool_path, pool=None)

    assert summary["episodes"] == 6  # 18 nets * 1 episode / 3 controlled
    # Every controlled episode resolves to exactly one of win/loss/draw.
    win = summary["controlled_win_rate"]
    loss = summary["controlled_loss_rate"]
    draw = summary["controlled_draw_rate"]
    assert abs((win + loss + draw) - 1.0) < 1e-6
    assert summary["min_net_appearances"] >= 1
    assert summary["steps_this_update"] > 0
    # Every net trained and is checkpoint-compatible.
    trainer.save_all(update=1, milestone=True)
    for spec in config.nets:
        latest = trainer.net_dir(spec.net_id) / "latest.pt"
        assert latest.exists()
        assert checkpoint_is_compatible(latest)
