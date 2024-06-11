from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rltoolkit.cleanrl.agent.base import BaseAgent
from rltoolkit.cleanrl.rl_args import A2CArguments


def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3) -> None:
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class PolicyNet(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super(PolicyNet, self).__init__()
        self.hidden = nn.Linear(obs_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.mu_layer.apply(initialize_uniformly)
        self.log_std_layer.apply(initialize_uniformly)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.hidden(obs)
        out = self.relu(out)

        mu = torch.tanh(self.mu_layer(out)) * 2
        log_std = F.softplus(self.log_std_layer(out))

        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        return action, dist


class ValueNet(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.fc1(obs)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class A2CAgent(BaseAgent):
    """Agent interacting with environment. The “Critic” estimates the value
    function. This could be the action-value (the Q value) or state-value (the
    V value). The “Actor” updates the policy distribution in the direction
    suggested by the Critic (such as with policy gradients).

    Attribute:
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        device (torch.device): cpu / gpu
    """

    def __init__(
        self,
        args: A2CArguments,
        env: gym.Env,
        state_shape: Union[int, List[int]] = None,
        action_shape: Union[int, List[int]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.args = args
        self.env = env
        self.device = device
        self.obs_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))
        self.global_update_step = 0
        self.target_model_update_step = 0

        # 策略网络
        self.policy_net = PolicyNet(self.obs_dim, self.args.hidden_dim,
                                    self.action_dim).to(device)
        # 价值网络
        self.critic = ValueNet(self.obs_dim, self.args.hidden_dim).to(device)

        # 策略网络优化器
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                                 lr=self.args.actor_lr)
        # 价值网络优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.args.critic_lr)
        # 折扣因子
        self.device = device

    def get_action(self, obs: np.ndarray) -> Tuple[int, float]:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action, dist = self.policy_net(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.item(), log_prob.item()

    def predict(self, obs: np.ndarray) -> Tuple[int, float]:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action, dist = self.policy_net(obs)
        selected_action = dist.mean
        return selected_action.item()

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    def learn(self, transition_dict) -> Tuple[float, float]:
        """Update the model by TD actor-critic."""
        obs = torch.tensor(transition_dict['obs'],
                           dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = (torch.tensor(transition_dict['rewards'],
                                dtype=torch.float).view(-1, 1).to(self.device))
        next_obs = torch.tensor(transition_dict['next_obs'],
                                dtype=torch.float).to(self.device)
        dones = (torch.tensor(transition_dict['dones'],
                              dtype=torch.float).view(-1, 1).to(self.device))

        log_probs = torch.log(self.policy_net(obs).gather(1, actions))

        pred_value = self.critic(obs)
        # 时序差分目标
        td_target = rewards + self.args.gamma * self.critic(next_obs) * (1 -
                                                                         dones)
        # 均方误差损失函数
        value_loss = F.mse_loss(pred_value, td_target.detach())

        # 时序差分误差
        td_delta = td_target - pred_value
        policy_loss = torch.mean(-log_probs * td_delta.detach())

        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()  # 计算策略网络的梯度
        self.policy_optimizer.step()  # 更新策略网络的参数

        return policy_loss.item(), value_loss.item()
