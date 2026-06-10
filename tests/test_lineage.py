"""Tests for per-lineage branching and selection (src/training/lineage.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.lineage import (
    BranchingConfig,
    _perturb_state_dict,
    _quick_state_hash,
    _collect_episodes_for_net,
    _mini_train_branch,
    run_branch_select_cycle,
)
from src.training.population import (
    build_default_population,
    make_net,
    PopulationConfig,
    PopulationTrainer,
    write_frozen_pool,
    save_net_checkpoint,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def population():
    return build_default_population()


@pytest.fixture(scope="module")
def spec0(population):
    return population[0]


@pytest.fixture(scope="module")
def net0(spec0):
    return make_net(spec0)


# --------------------------------------------------------------------------- #
# Perturbation tests
# --------------------------------------------------------------------------- #

def test_perturb_changes_weights(spec0, net0):
    state = {k: v.clone() for k, v in net0.state_dict().items()}
    perturbed = _perturb_state_dict(state, scale=0.01, seed=42)

    diffs = [
        (perturbed[k] - state[k]).abs().max().item()
        for k in state
        if state[k].is_floating_point()
    ]
    assert max(diffs) > 0, "Perturbation should change at least some weights"
    assert all(not torch.isnan(perturbed[k]).any() for k in perturbed if perturbed[k].is_floating_point()), \
        "Perturbed weights must not contain NaN"


def test_perturb_is_deterministic(spec0, net0):
    state = {k: v.clone() for k, v in net0.state_dict().items()}
    p1 = _perturb_state_dict(state, scale=0.005, seed=7)
    p2 = _perturb_state_dict(state, scale=0.005, seed=7)
    for k in state:
        assert torch.equal(p1[k], p2[k]), f"Same seed must produce same perturbation for {k}"


def test_perturb_different_seeds_differ(spec0, net0):
    state = {k: v.clone() for k, v in net0.state_dict().items()}
    p1 = _perturb_state_dict(state, scale=0.01, seed=1)
    p2 = _perturb_state_dict(state, scale=0.01, seed=2)
    float_keys = [k for k in state if state[k].is_floating_point()]
    diffs = [(p1[k] - p2[k]).abs().max().item() for k in float_keys]
    assert max(diffs) > 0, "Different seeds should produce different perturbations"


def test_state_hash_stable(spec0, net0):
    state = {k: v.clone() for k, v in net0.state_dict().items()}
    h1 = _quick_state_hash(state)
    h2 = _quick_state_hash(state)
    assert h1 == h2, "Hash must be stable for same weights"


def test_state_hash_sensitive_to_perturbation(spec0, net0):
    state = {k: v.clone() for k, v in net0.state_dict().items()}
    perturbed = _perturb_state_dict(state, scale=0.01, seed=99)
    assert _quick_state_hash(state) != _quick_state_hash(perturbed), \
        "Hash must differ after perturbation"


# --------------------------------------------------------------------------- #
# BranchingConfig defaults
# --------------------------------------------------------------------------- #

def test_branching_config_defaults():
    cfg = BranchingConfig()
    assert cfg.n_branches == 3
    assert cfg.mini_updates >= 1
    assert cfg.eval_episodes >= 2
    assert cfg.branch_interval >= 1
    assert 0 < cfg.mutation_scale < 0.1


# --------------------------------------------------------------------------- #
# Full branch-select cycle (tiny smoke)
# --------------------------------------------------------------------------- #

def test_branch_select_cycle_smoke(tmp_path, population):
    """One full branch-select cycle: creates branches, trains briefly, selects winner.
    Uses the smallest possible settings to keep the test fast.
    """
    nets_list = population[:18]
    config = PopulationConfig(
        run_dir=str(tmp_path),
        nets=nets_list,
        updates=1,
        episodes_per_net_per_update=1,
        workers=1,
        max_ticks=60,
        snapshot_interval=1,
        checkpoint_interval=1,
    )
    trainer = PopulationTrainer(config)
    trainer.save_all(update=0, milestone=False)

    # Write a frozen pool that branching can load.
    from src.training.policy import to_cpu_state_dict
    states = {spec.net_id: to_cpu_state_dict(trainer.nets[spec.net_id]) for spec in nets_list}
    pool_path = tmp_path / "frozen_pool" / "frozen_pool_v0.pt"
    spec_by_id = {spec.net_id: spec for spec in nets_list}
    write_frozen_pool(pool_path, spec_by_id, states, update=0)

    branching_cfg = BranchingConfig(
        n_branches=2,
        mini_updates=1,
        mini_episodes_per_update=1,
        eval_episodes=2,
        mutation_scale=0.005,
        branch_interval=1,
    )

    # Record parent hashes before branching.
    pre_hashes = {
        nid: _quick_state_hash(to_cpu_state_dict(net))
        for nid, net in trainer.nets.items()
    }

    summary = run_branch_select_cycle(
        trainer=trainer,
        branching_config=branching_cfg,
        frozen_pool_path=str(pool_path),
        cycle_num=1,
        update=1,
    )

    # Structural checks.
    assert summary["kind"] == "branch_select_cycle"
    assert summary["n_players"] == 18
    assert summary["n_branches"] == 2
    assert len(summary["player_results"]) == 18
    assert "avg_winner_win_rate" in summary
    assert "winner_branch_counts" in summary
    assert len(summary["winner_branch_counts"]) == 2
    assert sum(summary["winner_branch_counts"]) == 18

    # All 18 nets must still be in trainer (roots preserved).
    assert set(trainer.nets.keys()) == {spec.net_id for spec in nets_list}

    # latest.pt must exist for all 18.
    for spec in nets_list:
        assert (tmp_path / "nets" / spec.net_id / "latest.pt").exists(), \
            f"latest.pt missing for {spec.net_id}"

    # Player results structure.
    for r in summary["player_results"]:
        assert "net_id" in r
        assert "winner_branch_idx" in r
        assert "winner_win_rate" in r
        assert r["winner_branch_idx"] in (0, 1)
        assert 0.0 <= r["winner_win_rate"] <= 1.0
        assert "branch_win_rates" in r
        assert len(r["branch_win_rates"]) == 2
