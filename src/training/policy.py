from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.training.spaces import ACTION_DIM, OBSERVATION_DIM


MASK_VALUE = -1.0e9


@dataclass
class PolicyStep:
    action: int
    logprob: float
    entropy: float
    value: float


class MaskedActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
        hidden_size: int = 128,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.body = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(obs)
        return self.actor(features), self.critic(features).squeeze(-1)

    def distribution(self, obs: torch.Tensor, action_mask: torch.Tensor) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        logits, values = self.forward(obs)
        masked_logits = logits.masked_fill(~action_mask.bool(), MASK_VALUE)
        return torch.distributions.Categorical(logits=masked_logits), values

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
        device: torch.device | str = "cpu",
    ) -> PolicyStep:
        self.eval()
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
        dist, values = self.distribution(obs_tensor, mask_tensor)
        if deterministic:
            action_tensor = torch.argmax(dist.logits, dim=-1)
        else:
            action_tensor = dist.sample()
        logprob = dist.log_prob(action_tensor)
        entropy = dist.entropy()
        return PolicyStep(
            action=int(action_tensor.item()),
            logprob=float(logprob.item()),
            entropy=float(entropy.item()),
            value=float(values.item()),
        )


def to_cpu_state_dict(model: nn.Module) -> dict:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}
