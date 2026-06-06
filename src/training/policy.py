from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.training.spaces import ACTION_DIM, ACTION_HEAD_DIMS, ACTION_HEAD_SLICES, OBSERVATION_DIM


MASK_VALUE = -1.0e9


@dataclass
class PolicyStep:
    action: np.ndarray
    logprob: float
    entropy: float
    value: float


class FactorizedActionDistribution:
    def __init__(self, logits_by_head: tuple[torch.Tensor, ...], action_mask: torch.Tensor):
        if len(logits_by_head) != len(ACTION_HEAD_DIMS):
            raise ValueError("Policy head count does not match action space")
        mask = action_mask.bool().clone()
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        dists = []
        masked_logits = []
        for logits, action_slice in zip(logits_by_head, ACTION_HEAD_SLICES):
            head_mask = mask[:, action_slice].clone()
            empty_rows = ~head_mask.any(dim=-1)
            if empty_rows.any():
                head_mask[empty_rows, 0] = True
            head_logits = logits.masked_fill(~head_mask, MASK_VALUE)
            masked_logits.append(head_logits)
            dists.append(torch.distributions.Categorical(logits=head_logits))
        self.dists = tuple(dists)
        self.masked_logits_by_head = tuple(masked_logits)

    def sample(self) -> torch.Tensor:
        return torch.stack([dist.sample() for dist in self.dists], dim=-1)

    def argmax(self) -> torch.Tensor:
        return torch.stack([
            torch.argmax(logits, dim=-1)
            for logits in self.masked_logits_by_head
        ], dim=-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        action_tensor = actions.long()
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        parts = [
            dist.log_prob(action_tensor[:, head_index])
            for head_index, dist in enumerate(self.dists)
        ]
        return torch.stack(parts, dim=-1).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([dist.entropy() for dist in self.dists], dim=-1).sum(dim=-1)


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
        self.actor_heads = nn.ModuleList(
            nn.Linear(hidden_size, head_dim)
            for head_dim in ACTION_HEAD_DIMS
        )
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        features = self.body(obs)
        return tuple(actor(features) for actor in self.actor_heads), self.critic(features).squeeze(-1)

    def distribution(self, obs: torch.Tensor, action_mask: torch.Tensor) -> tuple[FactorizedActionDistribution, torch.Tensor]:
        logits_by_head, values = self.forward(obs)
        return FactorizedActionDistribution(logits_by_head, action_mask), values

    def action_distribution(self, obs: torch.Tensor, action_mask: torch.Tensor) -> FactorizedActionDistribution:
        features = self.body(obs)
        logits_by_head = tuple(actor(features) for actor in self.actor_heads)
        return FactorizedActionDistribution(logits_by_head, action_mask)

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
            action_tensor = dist.argmax()
        else:
            action_tensor = dist.sample()
        logprob = dist.log_prob(action_tensor)
        entropy = dist.entropy()
        return PolicyStep(
            action=action_tensor.squeeze(0).detach().cpu().numpy().astype(np.int64),
            logprob=float(logprob.item()),
            entropy=float(entropy.item()),
            value=float(values.item()),
        )

    @torch.no_grad()
    def act_action_only(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
        device: torch.device | str = "cpu",
    ) -> int:
        self.eval()
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
        dist = self.action_distribution(obs_tensor, mask_tensor)
        if deterministic:
            action_tensor = dist.argmax()
        else:
            action_tensor = dist.sample()
        return action_tensor.squeeze(0).detach().cpu().numpy().astype(np.int64)


def to_cpu_state_dict(model: nn.Module) -> dict:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}
