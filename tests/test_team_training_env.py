import json

import numpy as np
import pytest
import torch

from src.bots.scripted_bot import ScriptedBot
from src.core.engine import GameEngine
from src.core.state import get_team, make_initial_state
from src.training.env import ExternalActionBot
from src.bots.policy_bot import PolicyBot, _POLICY_STATE_CACHE
from src.core.constants import PLAYER_IDS, TICK_LIMIT
from src.training import ACTION_DIM, OBSERVATION_DIM, TeamTrainingEnv, TeamTrainingEnvConfig
from src.training.policy import MaskedActorCritic
from src.training.spaces import OBSERVATION_VERSION


def test_team_training_env_reset_step_and_masks_are_valid():
    env = TeamTrainingEnv(TeamTrainingEnvConfig(seed=11, max_ticks=30))
    observations, info = env.reset()

    assert set(observations) == {"A1", "A2", "A3"}
    assert info["controlled_team"] == "TEAM_A"
    assert info["tick"] == 0
    assert all(vector.shape == (OBSERVATION_DIM,) for vector in observations.values())
    assert all(np.isfinite(vector).all() for vector in observations.values())

    masks = env.action_masks()
    assert set(masks) == set(observations)
    assert all(mask.shape == (ACTION_DIM,) for mask in masks.values())
    assert all(mask[0] for mask in masks.values())

    result = env.step(env.sample_legal_actions())

    assert result.info["tick"] == 1
    assert env.state.tick == 1
    assert set(result.observations) == set(observations)
    assert all(np.isfinite(vector).all() for vector in result.observations.values())
    assert set(result.rewards) == set(observations)
    assert all(np.isfinite(reward) for reward in result.rewards.values())
    assert result.terminal in {True, False}
    assert result.truncated in {True, False}
    assert set(result.info["decoded_actions"]) == set(observations)
    assert env.engine.replay_ticks == []


def test_team_training_env_step_is_synchronized_once_for_all_players():
    env = TeamTrainingEnv(TeamTrainingEnvConfig(seed=12, max_ticks=30))
    env.reset()

    result = env.step({"A1": 0, "A2": 0, "A3": 0})

    assert env.state.tick == 1
    assert result.info["tick"] == 1
    assert result.info["action_indices"] == {"A1": 0, "A2": 0, "A3": 0}
    assert all(action["type"] == "HOLD" for action in result.info["decoded_actions"].values())


def test_team_training_env_invalid_actions_are_counted_and_fall_back():
    env = TeamTrainingEnv(TeamTrainingEnvConfig(seed=13, max_ticks=30))
    env.reset()

    result = env.step({"A1": ACTION_DIM + 10, "A2": 0, "A3": -1})

    assert result.info["invalid_actions"] == {"A1": 1, "A2": 0, "A3": 1}
    assert result.info["decoded_actions"]["A1"]["type"] == "HOLD"
    assert result.info["decoded_actions"]["A3"]["type"] == "HOLD"


def test_team_training_env_dead_player_mask_is_hold_only():
    env = TeamTrainingEnv(TeamTrainingEnvConfig(seed=14, max_ticks=30))
    env.reset()
    env.state.players["A2"].alive = False

    mask = env.action_mask("A2")

    assert mask[0]
    assert int(mask.sum()) == 1


def test_team_training_env_finalizes_death_time_at_real_tick_limit():
    env = TeamTrainingEnv(TeamTrainingEnvConfig(seed=15, max_ticks=TICK_LIMIT))
    env.reset()
    assert env.engine is not None
    assert env.state is not None

    env.engine.bots = {}
    env.state.tick = TICK_LIMIT - 1
    env.state.core_combat_damage = 1.0
    env.state.core_damage_by_team = {"TEAM_A": 1.0, "TEAM_B": 0.0}
    for player_id in PLAYER_IDS:
        env.state.players[player_id].alive = True
        env.state.players[player_id].core_hp = 50.0

    result = env.step({"A1": 0, "A2": 0, "A3": 0})

    assert result.terminal is True
    assert result.truncated is False
    assert result.info["winner"] == "TEAM_A"
    assert result.info["winner_type"] == "death_time"
    assert all(reward == 0.5 for reward in result.rewards.values())


def test_team_training_env_short_max_tick_limit_is_truncated():
    env = TeamTrainingEnv(TeamTrainingEnvConfig(seed=16, max_ticks=1))
    env.reset()

    result = env.step({"A1": 0, "A2": 0, "A3": 0})

    assert result.terminal is False
    assert result.truncated is True
    assert result.info["winner"] == "DRAW"
    assert result.info["winner_type"] == "truncated"


def test_team_training_env_requires_same_team_controlled_players():
    with pytest.raises(ValueError):
        TeamTrainingEnvConfig(controlled_players=("A1", "B1"))


def test_team_training_env_can_use_frozen_policy_opponent(tmp_path):
    _POLICY_STATE_CACHE.clear()
    policy = MaskedActorCritic(hidden_size=32)
    checkpoint_path = tmp_path / "team_policy.pt"
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
            "update": 1,
            "total_steps": 1,
        },
        checkpoint_path,
    )
    env = TeamTrainingEnv(TeamTrainingEnvConfig(
        seed=17,
        opponent_bot="policy",
        opponent_policy_checkpoint=str(checkpoint_path),
        opponent_policy_deterministic=True,
        max_ticks=30,
    ))

    observations, info = env.reset()
    result = env.step({"A1": 0, "A2": 0, "A3": 0})

    assert info["controlled_players"] == ["A1", "A2", "A3"]
    assert result.info["tick"] == 1
    assert isinstance(env.engine.bots["B1"], PolicyBot)
    assert isinstance(env.engine.bots["B2"], PolicyBot)
    assert isinstance(env.engine.bots["B3"], PolicyBot)
    assert set(observations) == {"A1", "A2", "A3"}


def test_team_training_env_policy_opponent_requires_checkpoint():
    with pytest.raises(ValueError, match="opponent_policy_checkpoint"):
        TeamTrainingEnv(TeamTrainingEnvConfig(
            seed=18,
            opponent_bot="policy",
            max_ticks=30,
        )).reset()


class ObservationHungryExternalBot(ExternalActionBot):
    requires_observation = True

    def __init__(self, player_id: str, team: str):
        super().__init__(player_id, team)
        self.non_empty_observations = 0

    def act(self, obs: dict):
        if obs:
            self.non_empty_observations += 1
        return super().act(obs)


class ObservationSkippingExternalBot(ExternalActionBot):
    requires_observation = False

    def __init__(self, player_id: str, team: str):
        super().__init__(player_id, team)
        self.non_empty_observations = 0

    def act(self, obs: dict):
        if obs:
            self.non_empty_observations += 1
        return super().act(obs)


def make_external_team_engine(external_bot_cls):
    state = make_initial_state(seed=20260605)
    bots = {}
    for player_id in PLAYER_IDS:
        team = get_team(player_id)
        bots[player_id] = (
            external_bot_cls(player_id, team)
            if team == "TEAM_A"
            else ScriptedBot(player_id, team)
        )
    return GameEngine(state, bots, record_trace=False, record_replay=False)


def test_external_action_observation_skip_preserves_engine_state():
    hungry = make_external_team_engine(ObservationHungryExternalBot)
    skipping = make_external_team_engine(ObservationSkippingExternalBot)

    for _ in range(20):
        hungry.tick()
        skipping.tick()

    assert sum(
        bot.non_empty_observations
        for bot in hungry.bots.values()
        if isinstance(bot, ObservationHungryExternalBot)
    ) > 0
    assert sum(
        bot.non_empty_observations
        for bot in skipping.bots.values()
        if isinstance(bot, ObservationSkippingExternalBot)
    ) == 0
    assert json.dumps(hungry.observe("A1"), sort_keys=True) == json.dumps(skipping.observe("A1"), sort_keys=True)
