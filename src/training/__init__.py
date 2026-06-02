"""Training-prep utilities for RL experiments.

This package intentionally does not train a neural network. It provides the
stable action/observation/reward plumbing that a future PPO/CleanRL/SB3 runner
can consume.
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

__all__ = [
    "ACTION_DIM",
    "ACTION_SPECS",
    "OBSERVATION_DIM",
    "OBSERVATION_VERSION",
    "TrainingEnv",
    "TrainingEnvConfig",
    "decode_action",
    "encode_observation",
    "find_route_obs",
    "legal_action_mask",
]
