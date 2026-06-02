import numpy as np

from src.training import ACTION_DIM, OBSERVATION_DIM, TrainingEnv, TrainingEnvConfig
from src.training.rollout import run_random_rollouts
from src.core.constants import PLAYER_IDS, TICK_LIMIT


def test_training_env_reset_step_and_mask_are_valid():
    env = TrainingEnv(TrainingEnvConfig(seed=11, max_ticks=30))
    obs, info = env.reset()

    assert obs.shape == (OBSERVATION_DIM,)
    assert np.isfinite(obs).all()
    assert info["legal_actions"] > 0

    mask = env.action_mask()
    assert mask.shape == (ACTION_DIM,)
    assert mask.any()

    next_obs, reward, terminal, truncated, step_info = env.step(env.sample_legal_action())

    assert next_obs.shape == obs.shape
    assert np.isfinite(next_obs).all()
    assert np.isfinite(reward)
    assert terminal in {True, False}
    assert truncated in {True, False}
    assert step_info["tick"] == 1
    assert "decoded_action" in step_info
    assert "shaping" in step_info["reward_components"]
    assert env.engine.replay_ticks == []


def test_training_env_episode_finishes_without_learning():
    env = TrainingEnv(TrainingEnvConfig(seed=12, max_ticks=20, opponent_bot="random", ally_bot="random"))
    env.reset()

    terminal = False
    truncated = False
    steps = 0
    while not (terminal or truncated):
        _, reward, terminal, truncated, info = env.step(env.sample_legal_action())
        assert np.isfinite(reward)
        assert info["legal_actions"] >= 1
        steps += 1

    assert steps <= 20
    assert info["winner"] in {"TEAM_A", "TEAM_B", "DRAW"}


def test_random_rollout_smoke_is_deterministic_for_same_seed():
    first = run_random_rollouts(episodes=3, seed=123, max_ticks=40)
    second = run_random_rollouts(episodes=3, seed=123, max_ticks=40)

    assert first == second
    assert first["kind"] == "random_rollout_smoke_no_training"
    assert first["episodes"] == 3
    assert all(item["observation_dim"] == OBSERVATION_DIM for item in first["results"])


def test_training_env_finalizes_death_time_at_real_tick_limit():
    env = TrainingEnv(TrainingEnvConfig(seed=123, max_ticks=TICK_LIMIT))
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

    _, _, terminal, truncated, info = env.step(0)

    assert terminal is True
    assert truncated is False
    assert info["winner"] == "TEAM_A"
    assert info["winner_type"] == "death_time"


def test_training_env_short_max_tick_limit_is_truncated():
    env = TrainingEnv(TrainingEnvConfig(seed=124, max_ticks=1))
    env.reset()

    _, _, terminal, truncated, info = env.step(0)

    assert terminal is False
    assert truncated is True
    assert info["winner"] == "DRAW"
    assert info["winner_type"] == "truncated"
