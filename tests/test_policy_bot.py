from pathlib import Path

import pytest
import torch

from src.bots.policy_bot import (
    PolicyBot,
    _POLICY_STATE_CACHE,
    checkpoint_is_compatible,
    policy_model_id_for_checkpoint,
    policy_state_id_from_model_id,
)
from src.core.runner import make_bot
from src.training.policy import MaskedActorCritic
from src.training.spaces import ACTION_DIM, OBSERVATION_DIM, OBSERVATION_VERSION


def test_policy_bot_loads_team_checkpoint_for_player_and_mirrored_slot(tmp_path: Path):
    _POLICY_STATE_CACHE.clear()
    policy = MaskedActorCritic(hidden_size=32)
    checkpoint_path = tmp_path / "team_checkpoint.pt"
    torch.save(
        {
            "policy_states": {
                "A1": policy.state_dict(),
                "A2": policy.state_dict(),
                "A3": policy.state_dict(),
            },
            "config": {"hidden_size": 32},
            "observation_version": OBSERVATION_VERSION,
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
        },
        checkpoint_path,
    )

    a_bot = PolicyBot("A1", "TEAM_A", checkpoint_path=checkpoint_path)
    b_bot = PolicyBot("B1", "TEAM_B", checkpoint_path=checkpoint_path)

    assert a_bot.policy.hidden_size == 32
    assert b_bot.policy.hidden_size == 32
    for key, value in policy.state_dict().items():
        assert torch.equal(a_bot.policy.state_dict()[key].cpu(), value)
        assert torch.equal(b_bot.policy.state_dict()[key].cpu(), value)


def test_policy_bot_can_load_named_policy_state_on_any_player_slot(tmp_path: Path, monkeypatch):
    _POLICY_STATE_CACHE.clear()
    policy = MaskedActorCritic(hidden_size=32)

    def marked_state(marker: float) -> dict:
        state = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        state["body.0.bias"][0] = marker
        return state

    checkpoint = tmp_path / "training_runs" / "state_pick" / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "policy_states": {
                "A1": marked_state(1.0),
                "A2": marked_state(2.0),
                "A3": marked_state(3.0),
            },
            "config": {"hidden_size": 32},
            "observation_version": OBSERVATION_VERSION,
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
        },
        checkpoint,
    )

    monkeypatch.setattr("src.bots.policy_bot.PROJECT_ROOT", tmp_path)
    model_id = "policy:training_runs/state_pick/checkpoints/latest.pt#A2"
    bot = make_bot(model_id, "B3", "TEAM_B")

    assert policy_state_id_from_model_id(model_id) == "A2"
    assert policy_model_id_for_checkpoint(checkpoint, "A2").endswith("#A2")
    assert bot.player_id == "B3"
    assert bot.policy_state_id == "A2"
    assert float(bot.policy.state_dict()["body.0.bias"][0]) == pytest.approx(2.0)


def test_policy_bot_rejects_checkpoint_with_old_observation_dim(tmp_path: Path):
    _POLICY_STATE_CACHE.clear()
    policy = MaskedActorCritic(observation_dim=OBSERVATION_DIM - 1, hidden_size=32)
    checkpoint_path = tmp_path / "old_obs_checkpoint.pt"
    torch.save(
        {
            "policy_state": policy.state_dict(),
            "config": {"hidden_size": 32},
            "observation_version": "train_obs_old",
            "observation_dim": OBSERVATION_DIM - 1,
            "action_dim": ACTION_DIM,
        },
        checkpoint_path,
    )

    assert checkpoint_is_compatible(checkpoint_path) is False
    with pytest.raises(ValueError, match="incompatible with current policy interface"):
        PolicyBot("A1", "TEAM_A", checkpoint_path=checkpoint_path)
