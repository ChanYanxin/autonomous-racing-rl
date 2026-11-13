from __future__ import annotations

import numpy as np
import torch
from torch import nn

RGB_PAYLOAD_INDICES = (7, 8, 9)


def preprocess_observation(observation: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert the racing observation into the policy input tensor.

    The course environment exposes a 456-element observation. Three entries are
    RGB payload containers that are not used by the learned policy, resulting in
    a 453-dimensional policy input.
    """
    if isinstance(observation, torch.Tensor):
        obs = observation.detach().cpu().numpy()
    else:
        obs = np.asarray(observation)

    if obs.ndim != 1:
        raise ValueError(f"Expected a 1D observation, got shape {obs.shape}.")
    if obs.shape[0] <= max(RGB_PAYLOAD_INDICES):
        raise ValueError("Observation is too short for the configured preprocessing.")

    processed = np.delete(obs, RGB_PAYLOAD_INDICES).astype(np.float32, copy=False)
    return torch.from_numpy(processed)


class Actor(nn.Module):
    """Deterministic policy network for bounded continuous actions."""

    def __init__(
        self,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)
        if action_low.shape != action_high.shape:
            raise ValueError("Action bounds must have identical shapes.")

        action_dim = int(np.prod(action_low.shape))
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(action_scale))
        self.register_buffer("action_bias", torch.as_tensor(action_bias))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized_action = self.network(observation)
        return normalized_action * self.action_scale + self.action_bias

    @torch.no_grad()
    def act(self, observation: torch.Tensor) -> np.ndarray:
        return self(observation).detach().cpu().numpy()


class Critic(nn.Module):
    """State-action value network Q(s, a)."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((observation, action), dim=-1))
