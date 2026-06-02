from pathlib import Path

import numpy as np
import torch

from src.training.policy import MaskedActorCritic
from src.training.ppo_trainer import (
    PPOConfig,
    load_checkpoint,
    save_checkpoint,
    summarize_action_distribution,
    summarize_training_episodes,
)
from src.training.spaces import ACTION_DIM, OBSERVATION_DIM


def test_masked_policy_never_selects_illegal_action():
    policy = MaskedActorCritic(hidden_size=32)
    obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    mask[3] = True

    for deterministic in (False, True):
        for _ in range(20):
            step = policy.act(obs, mask, deterministic=deterministic)
            assert step.action == 3


def test_checkpoint_roundtrip_restores_policy_and_optimizer(tmp_path: Path):
    config = PPOConfig(workers=1, updates=1, rollout_steps=4, hidden_size=32)
    policy = MaskedActorCritic(hidden_size=config.hidden_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    with torch.no_grad():
        for param in policy.parameters():
            param.add_(0.123)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, policy, optimizer, config, update=7, total_steps=1234)

    loaded_policy = MaskedActorCritic(hidden_size=config.hidden_size)
    loaded_optimizer = torch.optim.Adam(loaded_policy.parameters(), lr=config.learning_rate)
    checkpoint = load_checkpoint(checkpoint_path, loaded_policy, loaded_optimizer)

    assert checkpoint["update"] == 7
    assert checkpoint["total_steps"] == 1234
    for key, value in policy.state_dict().items():
        assert torch.equal(value, loaded_policy.state_dict()[key])


def test_default_eval_episodes_are_large_enough_for_win_rate():
    assert PPOConfig().eval_episodes == 50


def test_action_distribution_summary_exposes_argmax_collapse():
    summary = summarize_action_distribution([0, 0, 0, 1, 4])

    assert summary["total_actions"] == 5
    assert summary["unique_actions"] == 3
    assert summary["invalid_actions"] == 0
    assert summary["invalid_action_rate"] == 0.0
    assert summary["dominant_action"] == "0:HOLD"
    assert summary["dominant_action_rate"] == 0.6
    assert summary["kind_counts"]["HOLD"]["count"] == 3
    assert summary["kind_counts"]["PRODUCE"]["count"] == 1
    assert summary["kind_counts"]["MOVE_CAPTURE"]["count"] == 1


def test_action_distribution_reports_invalid_masked_actions():
    masks = np.zeros((3, ACTION_DIM), dtype=np.bool_)
    masks[0, 0] = True
    masks[1, 1] = True
    masks[2, 3] = True

    summary = summarize_action_distribution([0, 1, 4], masks)

    assert summary["invalid_actions"] == 1
    assert summary["invalid_action_rate"] == 0.333333


def test_training_episode_summary_can_include_action_distribution():
    masks = np.zeros((2, ACTION_DIM), dtype=np.bool_)
    masks[:, 0] = True
    summary = summarize_training_episodes([], [0, 0], masks)

    assert summary["episodes"] == 0
    assert summary["action_distribution"]["dominant_action"] == "0:HOLD"
    assert summary["action_distribution"]["invalid_action_rate"] == 0.0
