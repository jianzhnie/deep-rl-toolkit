import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import build_mlp, layer_init_with_orthogonal


class QNet(nn.Module):
    """Initialization.

    只有一层隐藏层的Q网络.
    """

    def __init__(
        self,
        state_shape: Union[int, Tuple[int]],
        action_shape: Union[int, Tuple[int]],
        hidden_dim: int = 128,
    ):
        super(QNet, self).__init__()
        obs_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape))
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.network(obs)


class DuelingNet(nn.Module):
    """只有一层隐藏层的A网络和V网络."""

    def __init__(
        self,
        state_shape: Union[int, Tuple[int]],
        action_shape: Union[int, Tuple[int]],
        hidden_dim: int = 128,
    ):
        """Initialization."""
        super(DuelingNet, self).__init__()
        obs_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape))
        # set common feature layer
        self.feature_layer = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                           nn.ReLU())
        # set advantage layer
        self.advantage_layer = nn.Linear(hidden_dim, action_dim)
        # set value layer
        self.value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = self.feature_layer(x)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        # Q值由V值和A值计算得到
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


# 定义一个添加噪声的网络层
class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        """Initialization."""
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))

        # 向模块添加持久缓冲区,这通常用于注册不应被视为模型参数的缓冲区。
        # 例如，BatchNorm的running_mean不是一个参数，而是持久状态的一部分。
        # 缓冲区可以使用给定的名称作为属性访问。
        self.register_buffer('weight_epsilon',
                             torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode. It doesn't show
        remarkable difference of performance.
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(
                self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init /
                                     math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init /
                                   math.sqrt(self.out_features))

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())


class NoisyNet(nn.Module):

    def __init__(
        self,
        state_shape: Union[int, Tuple[int]],
        action_shape: Union[int, Tuple[int]],
        hidden_dim: int,
        std_init: float = 0.5,
    ):
        """Initialization."""
        super(NoisyNet, self).__init__()
        obs_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape))
        self.feature = nn.Linear(obs_dim, hidden_dim)
        self.noisy_layer = NoisyLinear(hidden_dim, action_dim, std_init)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        out = self.feature(x)
        out = self.relu(out)
        out = self.noisy_layer(out)
        return out

    def reset_noise(self):
        """Reset all noisy layers."""
        self.noisy_layer.reset_noise()


# ALGO LOGIC: initialize agent here:
class AtariQNetwork(nn.Module):

    def __init__(
        self,
        state_shape: Union[int, Tuple[int]],
        action_shape: Union[int, Tuple[int]],
    ) -> None:
        super().__init__()
        obs_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape))
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class QNetBase(nn.Module):  # nn.Module is a standard PyTorch Network

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.explore_rate = 0.125
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim + action_dim, *dims, 1])

        self.state_avg = nn.Parameter(torch.zeros((state_dim, )),
                                      requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim, )),
                                      requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1, )), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1, )), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std

    def value_re_norm(self, value: Tensor) -> Tensor:
        return value * self.value_std + self.value_avg


class QNetWork(QNetBase):

    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        layer_init_with_orthogonal(self.net[-1], std=0.1)

    def forward(self, state):
        state = self.state_norm(state)
        value = self.net(state)
        value = self.value_re_norm(value)
        return value  # Q values for multiple actions

    def get_action(self, state):
        state = self.state_norm(state)
        if self.explore_rate < torch.rand(1):
            action = self.net(state).argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action


class QNetDuel(QNetBase):  # Dueling DQN

    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_state = build_mlp(dims=[state_dim, *dims])
        self.net_adv = build_mlp(dims=[dims[-1], 1])  # advantage value
        self.net_val = build_mlp(dims=[dims[-1], action_dim])  # Q value

        layer_init_with_orthogonal(self.net_adv[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val[-1], std=0.1)

    def forward(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val(s_enc)  # q value
        q_adv = self.net_adv(s_enc)  # advantage value
        value = q_val - q_val.mean(dim=1,
                                   keepdim=True) + q_adv  # dueling Q value
        value = self.value_re_norm(value)
        return value

    def get_action(self, state):
        state = self.state_norm(state)
        if self.explore_rate < torch.rand(1):
            s_enc = self.net_state(state)  # encoded state
            q_val = self.net_val(s_enc)  # q value
            action = q_val.argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action


class QNetTwin(QNetBase):  # Double DQN

    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_state = build_mlp(dims=[state_dim, *dims])
        self.net_val1 = build_mlp(dims=[dims[-1], action_dim])  # Q value 1
        self.net_val2 = build_mlp(dims=[dims[-1], action_dim])  # Q value 2
        self.soft_max = nn.Softmax(dim=1)

        layer_init_with_orthogonal(self.net_val1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val2[-1], std=0.1)

    def forward(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        return q_val  # one group of Q values

    def get_q1_q2(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val1 = self.net_val1(s_enc)  # q value 1
        q_val1 = self.value_re_norm(q_val1)
        q_val2 = self.net_val2(s_enc)  # q value 2
        q_val2 = self.value_re_norm(q_val2)
        return q_val1, q_val2  # two groups of Q values

    def get_action(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        if self.explore_rate < torch.rand(1):
            action = q_val.argmax(dim=1, keepdim=True)
        else:
            a_prob = self.soft_max(q_val)
            action = torch.multinomial(a_prob, num_samples=1)
        return action


class QNetTwinDuel(QNetBase):  # D3QN: Dueling Double DQN

    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_state = build_mlp(dims=[state_dim, *dims])
        self.net_adv1 = build_mlp(dims=[dims[-1], 1])  # advantage value 1
        self.net_val1 = build_mlp(dims=[dims[-1], action_dim])  # Q value 1
        self.net_adv2 = build_mlp(dims=[dims[-1], 1])  # advantage value 2
        self.net_val2 = build_mlp(dims=[dims[-1], action_dim])  # Q value 2
        self.soft_max = nn.Softmax(dim=1)

        layer_init_with_orthogonal(self.net_adv1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_adv2[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val2[-1], std=0.1)

    def forward(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        q_adv = self.net_adv1(s_enc)  # advantage value
        value = q_val - q_val.mean(dim=1,
                                   keepdim=True) + q_adv  # one dueling Q value
        value = self.value_re_norm(value)
        return value

    def get_q1_q2(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state

        q_val1 = self.net_val1(s_enc)  # q value 1
        q_adv1 = self.net_adv1(s_enc)  # advantage value 1
        q_duel1 = q_val1 - q_val1.mean(dim=1, keepdim=True) + q_adv1
        q_duel1 = self.value_re_norm(q_duel1)

        q_val2 = self.net_val2(s_enc)  # q value 2
        q_adv2 = self.net_adv2(s_enc)  # advantage value 2
        q_duel2 = q_val2 - q_val2.mean(dim=1, keepdim=True) + q_adv2
        q_duel2 = self.value_re_norm(q_duel2)
        return q_duel1, q_duel2  # two dueling Q values

    def get_action(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        if self.explore_rate < torch.rand(1):
            action = q_val.argmax(dim=1, keepdim=True)
        else:
            a_prob = self.soft_max(q_val)
            action = torch.multinomial(a_prob, num_samples=1)
        return action
