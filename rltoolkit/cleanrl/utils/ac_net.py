from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer


class ActorNet(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(ActorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.net(obs)
        return logits


class CriticNet(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim):
        super(CriticNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class SACActor(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super(SACActor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.relu = nn.ReLU()
        # set log_std layer
        self.std_layer = nn.Linear(hidden_dim, action_dim)
        # set mean layer
        self.mu_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.fc1(obs)
        out = self.relu(out)
        # get mean
        mu = self.mu_layer(out)
        # get std
        std = F.softplus(self.std_layer(out))
        # sample actions
        dist = Normal(mu, std)
        # Reparameterization trick (mean + std * N(0,1))
        x_t = dist.rsample()

        pi = torch.tanh(x_t)
        # normalize action and log_prob
        log_pi = dist.log_prob(x_t)
        # Enforcing Action Bound
        log_pi = log_pi - torch.log(1 - pi.pow(2) + 1e-7)
        return pi, log_pi


class SACCritic(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super(SACCritic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # 拼接状态和动作
        cat = torch.cat([x, a], dim=1)
        out = self.fc1(cat)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class DDPGActor(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        action_high: float,
        action_low: float,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        # action rescaling
        self.register_buffer(
            'action_scale',
            torch.tensor(
                (action_high - action_low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            'action_bias',
            torch.tensor(
                (action_high + action_low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class DDPGCritic(nn.Module):
    """Value Network for Critic in Actor-Critic method.

    Args:
        obs_dim (int): Dimension of observation space.
        hidden_dim (int): Dimension of hidden layer.
        action_dim (int): Dimension of action space.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(DDPGCritic, self).__init__()
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
