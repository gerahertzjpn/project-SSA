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
from src.training.spaces import ACTION_DIM, BUILD_ACTION_START, MOVE_ACTION_START, OBSERVATION_DIM, action_is_legal


def test_masked_policy_never_selects_illegal_action():
    policy = MaskedActorCritic(hidden_size=32)
    obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    mask[3] = True
    mask[MOVE_ACTION_START] = True
    mask[BUILD_ACTION_START] = True

    for deterministic in (False, True):
        for _ in range(20):
            step = policy.act(obs, mask, deterministic=deterministic)
            np.testing.assert_array_equal(step.action, np.array([3, 0, 0], dtype=np.int64))


def test_action_only_path_matches_full_policy_action_for_same_rng():
    policy = MaskedActorCritic(hidden_size=32)
    rng = np.random.default_rng(123)
    obs = rng.normal(size=OBSERVATION_DIM).astype(np.float32)
    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    for start, stop in ((0, MOVE_ACTION_START), (MOVE_ACTION_START, BUILD_ACTION_START), (BUILD_ACTION_START, ACTION_DIM)):
        local_legal = rng.choice(np.arange(start, stop), size=min(10, stop - start), replace=False)
        mask[local_legal] = True

    for deterministic in (False, True):
        for seed in range(20):
            torch.manual_seed(seed)
            full_action = policy.act(obs, mask, deterministic=deterministic).action
            torch.manual_seed(seed)
            action_only = policy.act_action_only(obs, mask, deterministic=deterministic)
            np.testing.assert_array_equal(action_only, full_action)
            assert action_is_legal(action_only, mask)


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
    summary = summarize_action_distribution([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ])

    assert summary["total_actions"] == 5
    assert summary["unique_actions"] == 3
    assert summary["invalid_actions"] == 0
    assert summary["invalid_action_rate"] == 0.0
    assert summary["dominant_action"] == "(0, 0, 0):HOLD"
    assert summary["dominant_action_rate"] == 0.6
    assert summary["kind_counts"]["HOLD"]["count"] == 3
    assert summary["kind_counts"]["PRODUCE"]["count"] == 1
    assert summary["kind_counts"]["MOVE_CAPTURE"]["count"] == 1


def test_action_distribution_reports_invalid_masked_actions():
    masks = np.zeros((3, ACTION_DIM), dtype=np.bool_)
    masks[0, 0] = True
    masks[0, MOVE_ACTION_START] = True
    masks[0, BUILD_ACTION_START] = True
    masks[1, 1] = True
    masks[1, MOVE_ACTION_START] = True
    masks[1, BUILD_ACTION_START] = True
    masks[2, 3] = True
    masks[2, MOVE_ACTION_START] = True
    masks[2, BUILD_ACTION_START] = True

    summary = summarize_action_distribution([[0, 0, 0], [1, 0, 0], [4, 0, 0]], masks)

    assert summary["invalid_actions"] == 1
    assert summary["invalid_action_rate"] == 0.333333


def test_training_episode_summary_can_include_action_distribution():
    masks = np.zeros((2, ACTION_DIM), dtype=np.bool_)
    masks[:, 0] = True
    masks[:, MOVE_ACTION_START] = True
    masks[:, BUILD_ACTION_START] = True
    summary = summarize_training_episodes([], [[0, 0, 0], [0, 0, 0]], masks)

    assert summary["episodes"] == 0
    assert summary["action_distribution"]["dominant_action"] == "(0, 0, 0):HOLD"
    assert summary["action_distribution"]["invalid_action_rate"] == 0.0
