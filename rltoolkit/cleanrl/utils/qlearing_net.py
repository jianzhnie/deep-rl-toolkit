from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super(QNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
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
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        """Initialization."""
        super(DuelingNet, self).__init__()
        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # set advantage layer
        self.advantage_layer = nn.Linear(hidden_dim, action_dim)
        # set value layer
        self.value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = self.feature_layer(obs)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        # Q值由V值和A值计算得到
        q_value = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_value


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        noisy_std (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 noisy_std: float = 0.5) -> None:
        super(NoisyLinear, self).__init__()
        """Initialization."""
        self.in_features = in_features
        self.out_features = out_features
        self.noisy_std = noisy_std

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
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.noisy_std /
                                     np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.noisy_std / np.sqrt(self.out_features))

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
        noisy_std: float = 0.5,
    ):
        """Initialization."""
        super(NoisyNet, self).__init__()
        obs_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape))
        self.feature = nn.Linear(obs_dim, hidden_dim)
        self.noisy_layer = NoisyLinear(hidden_dim, action_dim, noisy_std)
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


class RainbowNet(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_atoms: int,
        support: torch.Tensor,
        noisy_std: float = 0.5,
        is_dueling: bool = True,
        is_noisy: bool = True,
    ) -> None:
        """Initialization."""

        super().__init__()

        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.support = support

        def linear(in_features, out_features):
            if is_noisy:
                return NoisyLinear(in_features, out_features, noisy_std)
            else:
                return nn.Linear(in_features, out_features)

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.value_net = nn.Sequential(
            linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            linear(hidden_dim, self.num_atoms),
        )
        self.is_dueling = is_dueling
        if self.is_dueling:
            self.advantage_net = nn.Sequential(
                linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                linear(hidden_dim, self.action_dim * self.num_atoms),
            )
        self.output_dim = self.action_dim * self.num_atoms

    def forward(self,
                obs: torch.Tensor,
                return_qval: bool = True) -> torch.Tensor:
        """Forward method implementation."""
        feature = self.feature_layer(obs)
        value = self.value_net(feature)
        value = value.view(-1, 1, self.num_atoms)

        if self.is_dueling:
            advantage = self.advantage_net(feature)
            advantage = advantage.view(-1, self.action_dim, self.num_atoms)
            logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            logits = advantage

        prob_dist = logits.softmax(dim=2)
        prob_dist = prob_dist.clamp(min=1e-3)  # for avoiding nans

        qval = torch.sum(prob_dist * self.support, dim=2)
        if return_qval:
            return qval
        return prob_dist

    def reset_noise(self):
        """Resets noise of value and advantage networks."""
        for layer in self.value_net:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_net:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


class C51Network(nn.Module):
    """Neural network for the C51 algorithm in reinforcement learning.

    Args:
        obs_dim (int): Dimension of the observation space.
        hidden_dim (int): Dimension of the hidden layers.
        action_dim (int): Dimension of the action space.
        num_atoms (int, optional): Number of atoms for the value distribution. Defaults to 101.
        v_min (float, optional): Minimum value of the value distribution. Defaults to -100.
        v_max (float, optional): Maximum value of the value distribution. Defaults to 100.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_atoms: int = 101,
        v_min: float = -100,
        v_max: float = 100,
    ) -> None:
        super().__init__()
        self.num_atoms = num_atoms
        self.register_buffer('atoms',
                             torch.linspace(v_min, v_max, steps=num_atoms))

        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim * num_atoms),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            obs (torch.Tensor): The observation tensor of shape (batch_size, obs_dim).
            action (Optional[torch.Tensor]): The action tensor of shape (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - The action tensor of shape (batch_size,).
                - The probability mass function (PMF) for each action of shape (batch_size, num_atoms).
        """
        # Compute logits: (batch_size, action_dim * num_atoms)
        logits = self.network(obs)
        # Reshape logits to (batch_size, action_dim, num_atoms)
        logits = logits.view(len(obs), self.action_dim, self.num_atoms)
        # Compute the probability mass function (PMF) using softmax
        pmfs = F.softmax(logits, dim=2)
        # Compute Q-values by summing over the atoms dimension
        q_values = (pmfs * self.atoms).sum(dim=2)

        if action is None:
            # Select the action with the highest Q-value
            action = torch.argmax(q_values, dim=1).to(dtype=torch.long)
        # Return the action and the PMF of the selected action
        return action, pmfs[torch.arange(len(obs)), action]
