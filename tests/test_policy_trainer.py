from pathlib import Path

import numpy as np
import torch

from src.training.policy import MaskedActorCritic
from src.training.ppo_trainer import PPOConfig, load_checkpoint, save_checkpoint
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
