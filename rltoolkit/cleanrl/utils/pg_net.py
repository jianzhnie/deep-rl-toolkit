import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticNet(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super(ActorCriticNet, self).__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.actor(obs)
        return logits

    def get_action_and_value(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value


class ActorNet(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super(ActorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.net(obs)
        return logits


class ValueNet(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
