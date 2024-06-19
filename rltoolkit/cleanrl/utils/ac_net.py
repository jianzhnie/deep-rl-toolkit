from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        action_low: float,
        action_high: float,
        log_std_min: float,
        log_std_max: float,
    ):
        super(SACActor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # set log_std layer
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        # set mean layer
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
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
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.fc1(obs))
        out = F.relu(self.fc1(obs))
        # get mean
        mean = self.fc_mean(out)
        # get std
        log_std = self.fc_logstd(out)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max -
                                            self.log_std_min) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs: torch.Tensor):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal_dist = torch.distributions.Normal(mean, std)
        x_t = normal_dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal_dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SACCritic(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super(SACCritic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # 拼接状态和动作
        inputs = torch.cat([obs, action], dim=1)
        logits = self.net(inputs)
        return logits


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
