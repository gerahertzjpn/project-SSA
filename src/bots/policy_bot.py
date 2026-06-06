from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from src.bots.base import Bot
from src.core.constants import TEAM_A_PLAYERS, TEAM_B_PLAYERS
from src.core.engine import Action
from src.training.policy import MaskedActorCritic
from src.training.spaces import (
    ACTION_DIM,
    OBSERVATION_DIM,
    OBSERVATION_VERSION,
    decode_action,
    encode_observation,
    legal_action_mask,
)


_POLICY_STATE_CACHE: dict[str, tuple[dict, int]] = {}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIRRORED_PLAYER_SLOT = {
    **dict(zip(TEAM_A_PLAYERS, TEAM_B_PLAYERS)),
    **dict(zip(TEAM_B_PLAYERS, TEAM_A_PLAYERS)),
}


class PolicyBot(Bot):
    def __init__(
        self,
        player_id: str,
        team: str,
        checkpoint_path: str | Path,
        deterministic: bool = True,
        device: str = "cpu",
    ):
        super().__init__(player_id, team)
        self.checkpoint_path = Path(checkpoint_path)
        self.deterministic = deterministic
        self.device = torch.device(device)
        policy_state, hidden_size = cached_policy_state(self.checkpoint_path, player_id)
        self.policy = MaskedActorCritic(hidden_size=hidden_size).to(self.device)
        self.policy.load_state_dict(policy_state)
        self.policy.eval()

    def act(self, obs: dict) -> Action:
        mask = legal_action_mask(obs, self.player_id)
        encoded = encode_observation(obs, self.player_id)
        action = self.policy.act_action_only(
            encoded,
            mask,
            deterministic=self.deterministic,
            device=self.device,
        )
        return decode_action(action, obs, self.player_id, assume_legal=True)


def cached_policy_state(path: str | Path, player_id: str) -> tuple[dict, int]:
    checkpoint_path = Path(path)
    cache_key = f"{checkpoint_path.resolve()}::{player_id}"
    if cache_key not in _POLICY_STATE_CACHE:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        policy_state = policy_state_from_checkpoint(checkpoint, checkpoint_path, player_id)
        validate_checkpoint_compatible(checkpoint, policy_state, checkpoint_path)
        _POLICY_STATE_CACHE[cache_key] = (
            policy_state,
            checkpoint_hidden_size(checkpoint),
        )
    return _POLICY_STATE_CACHE[cache_key]


def policy_state_from_checkpoint(checkpoint: dict, checkpoint_path: Path, player_id: str) -> dict:
    if "policy_state" in checkpoint:
        return checkpoint["policy_state"]

    policy_states = checkpoint.get("policy_states")
    if isinstance(policy_states, dict):
        if player_id in policy_states:
            return policy_states[player_id]
        mirrored_player_id = MIRRORED_PLAYER_SLOT.get(player_id)
        if mirrored_player_id in policy_states:
            return policy_states[mirrored_player_id]
        available = ", ".join(sorted(policy_states))
        raise ValueError(
            f"Team policy checkpoint {checkpoint_path} has no state for {player_id}; "
            f"available states: {available}"
        )

    raise ValueError(f"Policy checkpoint {checkpoint_path} has no policy_state or policy_states")


def checkpoint_hidden_size(checkpoint: dict) -> int:
    config = checkpoint.get("config")
    if isinstance(config, dict):
        return int(config.get("hidden_size", 128) or 128)
    return 128


def validate_checkpoint_compatible(checkpoint: dict, policy_state: dict, checkpoint_path: Path) -> None:
    checkpoint_version = checkpoint.get("observation_version")
    checkpoint_obs_dim = checkpoint.get("observation_dim")
    checkpoint_action_dim = checkpoint.get("action_dim")
    if (
        checkpoint_version != OBSERVATION_VERSION
        or int(checkpoint_obs_dim or -1) != OBSERVATION_DIM
        or int(checkpoint_action_dim or -1) != ACTION_DIM
    ):
        raise ValueError(
            f"Policy checkpoint {checkpoint_path} is incompatible with current "
            f"policy interface metadata: checkpoint version/obs/action="
            f"({checkpoint_version}, {checkpoint_obs_dim}, {checkpoint_action_dim}), "
            f"current=({OBSERVATION_VERSION}, {OBSERVATION_DIM}, {ACTION_DIM})"
        )
    validate_policy_state_compatible(policy_state, checkpoint_path)


def validate_policy_state_compatible(policy_state: dict, checkpoint_path: Path) -> None:
    observation_dim, action_dim = policy_state_interface(policy_state)
    if observation_dim != OBSERVATION_DIM or action_dim != ACTION_DIM:
        raise ValueError(
            f"Policy checkpoint {checkpoint_path} is incompatible with current "
            f"policy interface: checkpoint obs/action=({observation_dim}, {action_dim}), "
            f"current obs/action=({OBSERVATION_DIM}, {ACTION_DIM})"
        )


def policy_state_interface(policy_state: dict) -> tuple[int, int]:
    first_layer = policy_state.get("body.0.weight")
    if first_layer is None:
        return -1, -1
    actor = policy_state.get("actor.weight")
    if actor is not None:
        return int(first_layer.shape[1]), int(actor.shape[0])
    head_rows = []
    head_index = 0
    while True:
        head = policy_state.get(f"actor_heads.{head_index}.weight")
        if head is None:
            break
        head_rows.append(int(head.shape[0]))
        head_index += 1
    if not head_rows:
        return -1, -1
    return int(first_layer.shape[1]), int(sum(head_rows))


def checkpoint_is_compatible(path: str | Path) -> bool:
    try:
        checkpoint_path = Path(path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("observation_version") != OBSERVATION_VERSION
            or int(checkpoint.get("observation_dim") or -1) != OBSERVATION_DIM
            or int(checkpoint.get("action_dim") or -1) != ACTION_DIM
        ):
            return False
        if "policy_state" in checkpoint:
            states = [checkpoint["policy_state"]]
        else:
            states = list((checkpoint.get("policy_states") or {}).values())
        if not states:
            return False
        return all(policy_state_interface(state) == (OBSERVATION_DIM, ACTION_DIM) for state in states)
    except Exception:
        return False


def default_policy_checkpoint() -> Optional[Path]:
    candidates = [
        Path("training_runs/sprint2_random_first_real_v1/checkpoints/checkpoint_update_180.pt"),
        Path("training_runs/sprint2_random_first_real_v1/checkpoints/latest.pt"),
    ]
    for path in candidates:
        if path.exists() and checkpoint_is_compatible(path):
            return path
    return None


def default_team_policy_checkpoint() -> Optional[Path]:
    candidates = [
        Path("training_runs/sprint3_team_scripted_warm_v1/checkpoints/checkpoint_update_240.pt"),
        Path("training_runs/sprint3_team_scripted_warm_v1/checkpoints/latest.pt"),
    ]
    for path in candidates:
        if path.exists() and checkpoint_is_compatible(path):
            return path
    return None


def policy_model_id_for_checkpoint(path: str | Path) -> str:
    checkpoint_path = Path(path).resolve()
    try:
        relative = checkpoint_path.relative_to(PROJECT_ROOT)
    except ValueError:
        relative = checkpoint_path
    return f"policy:{relative.as_posix()}"


def checkpoint_from_policy_model_id(model_id: str) -> Path:
    if not model_id.startswith("policy:"):
        raise ValueError(f"Not a policy model id: {model_id}")

    raw_path = Path(model_id.split(":", 1)[1])
    if raw_path.is_absolute():
        raise ValueError("Policy model path must be relative")

    checkpoint_path = (PROJECT_ROOT / raw_path).resolve()
    training_root = (PROJECT_ROOT / "training_runs").resolve()
    try:
        checkpoint_path.relative_to(training_root)
    except ValueError as error:
        raise ValueError("Policy checkpoints must live under training_runs") from error

    if checkpoint_path.suffix != ".pt" or not checkpoint_path.exists():
        raise ValueError(f"Policy checkpoint is not available: {raw_path.as_posix()}")
    return checkpoint_path
