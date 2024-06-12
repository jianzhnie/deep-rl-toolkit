from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer


class PolicyNet(nn.Module):
    """Policy Network for Actor-Critic method.

    Args:
        obs_dim (int): Dimension of observation space.
        hidden_dim (int): Dimension of hidden layer.
        action_dim (int): Dimension of action space.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = init_layer_uniform(self.fc2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs (torch.Tensor): Observation tensor.

        Returns:
            torch.Tensor: Action tensor mapped to the range [-1, 1].
        """
        out = self.fc1(obs)
        out = self.relu(out)
        out = self.fc2(out)
        out = torch.tanh(out)  # Map to [-1, 1]
        return out


class ValueNet(nn.Module):
    """Value Network for Critic in Actor-Critic method.

    Args:
        obs_dim (int): Dimension of observation space.
        hidden_dim (int): Dimension of hidden layer.
        action_dim (int): Dimension of action space.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = init_layer_uniform(self.fc2)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs (torch.Tensor): Observation tensor.
            action (torch.Tensor): Action tensor.

        Returns:
            torch.Tensor: Q-value tensor.
        """
        # Concatenate observation and action
        cat = torch.cat([obs, action], dim=1)
        out = self.fc1(cat)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class OUNoise(object):

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 0.3,
        theta: float = 0.15,
        dt: float = 1e-2,
        x0: Optional[Union[float, np.ndarray]] = None,
    ):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)

    def __call__(self, size: Sequence[int]):
        x = (self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt +
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=size))
        self.x_prev = x
        return x
