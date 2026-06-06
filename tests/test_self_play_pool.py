from pathlib import Path

import pytest

from scripts.build_self_play_pool import build_pool
from src.training.spaces import ACTION_DIM, OBSERVATION_DIM, OBSERVATION_VERSION
from src.training.team_ppo_trainer import (
    TeamPPOConfig,
    make_team_optimizers,
    make_team_policies,
    save_team_checkpoint,
)


def test_build_self_play_pool_manifest_from_compatible_team_checkpoints(tmp_path: Path):
    run_dir = tmp_path / "training_runs" / "sprint3_candidate"
    config = TeamPPOConfig(seed=1, hidden_size=32)
    policies = make_team_policies(config)
    optimizers = make_team_optimizers(policies, config)
    checkpoint_path = run_dir / "checkpoints" / "checkpoint_update_7.pt"
    latest_path = run_dir / "checkpoints" / "latest.pt"
    save_team_checkpoint(checkpoint_path, policies, optimizers, config, update=7, total_steps=123)
    save_team_checkpoint(latest_path, policies, optimizers, config, update=7, total_steps=123)

    pool = build_pool(run_dir, ["7", "latest"])

    assert pool["kind"] == "self_play_policy_pool"
    assert pool["observation_version"] == OBSERVATION_VERSION
    assert pool["observation_dim"] == OBSERVATION_DIM
    assert pool["action_dim"] == ACTION_DIM
    assert len(pool["entries"]) == 2
    assert {entry["update"] for entry in pool["entries"]} == {7}
    assert all(entry["path"].endswith(".pt") for entry in pool["entries"])


def test_build_self_play_pool_rejects_missing_checkpoint(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_pool(tmp_path / "missing_run", ["latest"])
