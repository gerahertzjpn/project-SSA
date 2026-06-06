"""Training utilities for RL experiments.

The package exposes the stable action/observation/reward plumbing plus a small
masked PPO runner used for the first Sprint 2 smoke training runs.
"""

from .spaces import (
    ACTION_DIM,
    ACTION_HEAD_DIMS,
    ACTION_HEAD_SLICES,
    ACTION_SPECS,
    OBSERVATION_DIM,
    OBSERVATION_VERSION,
    action_is_legal,
    decode_action,
    encode_observation,
    find_route_obs,
    flatten_factorized_action,
    legal_action_mask,
    mirror_action_index,
    mirror_observation,
    normalize_action,
    policy_observation,
    sample_legal_action,
)
from .env import TrainingEnv, TrainingEnvConfig
from .team_env import TeamTrainingEnv, TeamTrainingEnvConfig
from .policy import MaskedActorCritic
from .ppo_trainer import PPOConfig
from .team_ppo_trainer import TeamPPOConfig

__all__ = [
    "ACTION_DIM",
    "ACTION_HEAD_DIMS",
    "ACTION_HEAD_SLICES",
    "ACTION_SPECS",
    "OBSERVATION_DIM",
    "OBSERVATION_VERSION",
    "TrainingEnv",
    "TrainingEnvConfig",
    "TeamTrainingEnv",
    "TeamTrainingEnvConfig",
    "MaskedActorCritic",
    "PPOConfig",
    "TeamPPOConfig",
    "action_is_legal",
    "decode_action",
    "encode_observation",
    "find_route_obs",
    "flatten_factorized_action",
    "legal_action_mask",
    "mirror_action_index",
    "mirror_observation",
    "normalize_action",
    "policy_observation",
    "sample_legal_action",
]
