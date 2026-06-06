from pathlib import Path

import numpy as np
import torch

from src.training.team_ppo_trainer import (
    TeamPPOConfig,
    _role_assignment,
    collect_team_rollouts,
    evaluate_random_legal_team,
    evaluate_team_policies,
    load_single_policy_into_team,
    load_team_checkpoint,
    make_team_optimizers,
    make_team_policies,
    save_team_checkpoint,
    summarize_team_rollout,
    update_team_policies,
)
from src.training.spaces import ACTION_DIM, OBSERVATION_DIM, OBSERVATION_VERSION


def test_collect_team_rollouts_has_one_buffer_per_controlled_player():
    config = TeamPPOConfig(
        seed=101,
        workers=1,
        rollout_steps=6,
        hidden_size=32,
        max_ticks=30,
    )
    policies = make_team_policies(config)

    rollout = collect_team_rollouts(policies, config, update=1)

    assert set(rollout["agents"]) == {"A1", "A2", "A3"}
    for agent in rollout["agents"].values():
        assert agent["observations"].shape[0] == 6
        assert agent["masks"].shape[0] == 6
        assert agent["actions"].shape == (6,)
        assert agent["advantages"].shape == (6,)
        assert np.isfinite(agent["returns"]).all()


def test_role_assignment_shuffle_can_train_each_policy_on_other_slots():
    config = TeamPPOConfig(seed=101, role_assignment_mode="shuffle")
    rng = np.random.default_rng(123)

    assignments = [_role_assignment(config, rng) for _ in range(20)]

    assert any(assignment["A1"] != "A1" for assignment in assignments)
    assert all(set(assignment) == {"A1", "A2", "A3"} for assignment in assignments)
    assert all(set(assignment.values()) == {"A1", "A2", "A3"} for assignment in assignments)


def test_collect_team_rollouts_role_shuffle_keeps_policy_buffers_aligned():
    config = TeamPPOConfig(
        seed=101,
        workers=1,
        rollout_steps=6,
        hidden_size=32,
        max_ticks=30,
        role_assignment_mode="shuffle",
    )
    policies = make_team_policies(config)

    rollout = collect_team_rollouts(policies, config, update=1)

    assert set(rollout["agents"]) == {"A1", "A2", "A3"}
    assert rollout["role_assignment_counts"]
    for agent in rollout["agents"].values():
        assert agent["observations"].shape[0] == 6
        assert agent["masks"].shape[0] == 6
        assert agent["actions"].shape == (6,)
        assert agent["advantages"].shape == (6,)
        assert np.isfinite(agent["returns"]).all()


def test_team_policy_update_and_summary_are_finite():
    config = TeamPPOConfig(
        seed=102,
        workers=1,
        rollout_steps=8,
        hidden_size=32,
        max_ticks=30,
        minibatch_size=4,
    )
    device = torch.device("cpu")
    policies = make_team_policies(config, device=device)
    optimizers = make_team_optimizers(policies, config)
    rollout = collect_team_rollouts(policies, config, update=1)

    loss = update_team_policies(policies, optimizers, rollout, config, device)
    summary = summarize_team_rollout(rollout, config)

    assert set(loss) == {"A1", "A2", "A3"}
    assert all(np.isfinite(value) for agent_loss in loss.values() for value in agent_loss.values())
    assert set(summary["agents"]) == {"A1", "A2", "A3"}
    assert all(summary["agents"][pid]["action_distribution"]["total_actions"] == 8 for pid in summary["agents"])


def test_team_checkpoint_roundtrip_restores_all_policies(tmp_path: Path):
    config = TeamPPOConfig(seed=103, workers=1, rollout_steps=4, hidden_size=32)
    policies = make_team_policies(config)
    optimizers = make_team_optimizers(policies, config)
    checkpoint_path = tmp_path / "team_checkpoint.pt"

    save_team_checkpoint(checkpoint_path, policies, optimizers, config, update=7, total_steps=123)

    loaded_policies = make_team_policies(config)
    loaded_optimizers = make_team_optimizers(loaded_policies, config)
    checkpoint = load_team_checkpoint(checkpoint_path, loaded_policies, loaded_optimizers)

    assert checkpoint["update"] == 7
    assert checkpoint["total_steps"] == 123
    for player_id in config.controlled_players:
        for key, value in policies[player_id].state_dict().items():
            assert torch.equal(value, loaded_policies[player_id].state_dict()[key])


def test_single_checkpoint_can_warm_start_team_policies(tmp_path: Path):
    config = TeamPPOConfig(seed=104, hidden_size=32)
    single = make_team_policies(config)["A1"]
    checkpoint_path = tmp_path / "single.pt"
    torch.save(
        {
            "policy_state": single.state_dict(),
            "observation_version": OBSERVATION_VERSION,
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
        },
        checkpoint_path,
    )

    policies = make_team_policies(config)
    load_single_policy_into_team(checkpoint_path, policies)

    for player_id in config.controlled_players:
        for key, value in single.state_dict().items():
            assert torch.equal(value, policies[player_id].state_dict()[key])


def test_team_eval_and_random_legal_smoke_are_finite():
    config = TeamPPOConfig(seed=105, workers=1, rollout_steps=4, hidden_size=32, max_ticks=20)
    policies = make_team_policies(config)

    policy_eval = evaluate_team_policies(
        policies,
        config,
        episodes=2,
        seed=105,
        deterministic=True,
        device=torch.device("cpu"),
    )
    random_eval = evaluate_random_legal_team(config, episodes=2, seed=205)

    assert policy_eval["episodes"] == 2
    assert random_eval["episodes"] == 2
    assert set(policy_eval["agents"]) == {"A1", "A2", "A3"}
    assert set(random_eval["agents"]) == {"A1", "A2", "A3"}
