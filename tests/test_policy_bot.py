from pathlib import Path

import torch

from src.bots.policy_bot import PolicyBot, _POLICY_STATE_CACHE
from src.training.policy import MaskedActorCritic


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
