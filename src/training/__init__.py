"""Training utilities for RL experiments.

The package exposes the stable action/observation/reward plumbing plus a small
masked PPO runner used for the first Sprint 2 smoke training runs.
"""

from .spaces import (
    ACTION_DIM,
    ACTION_SPECS,
    OBSERVATION_DIM,
    OBSERVATION_VERSION,
    decode_action,
    encode_observation,
    find_route_obs,
    legal_action_mask,
)
from .env import TrainingEnv, TrainingEnvConfig
from .team_env import TeamTrainingEnv, TeamTrainingEnvConfig
from .policy import MaskedActorCritic
from .ppo_trainer import PPOConfig

__all__ = [
    "ACTION_DIM",
    "ACTION_SPECS",
    "OBSERVATION_DIM",
    "OBSERVATION_VERSION",
    "TrainingEnv",
    "TrainingEnvConfig",
    "TeamTrainingEnv",
    "TeamTrainingEnvConfig",
    "MaskedActorCritic",
    "PPOConfig",
    "decode_action",
    "encode_observation",
    "find_route_obs",
    "legal_action_mask",
]
