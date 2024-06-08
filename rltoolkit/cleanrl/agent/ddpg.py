import copy
from typing import Dict, List, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rltoolkit.cleanrl.agent.base import BaseAgent
from rltoolkit.cleanrl.rl_args import DDPGArgments
from rltoolkit.utils import soft_target_update


class PolicyNet(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # 连续动作空间，利用 tanh() 函数将特征映射到 [-1, 1],
        # 然后通过变换，得到 [low, high] 的输出
        out = torch.tanh(x)
        return out


class ValueNet(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
    ):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # 拼接状态和动作
        cat = torch.cat([x, a], dim=1)
        out = self.fc1(cat)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class DDPGAgent(BaseAgent):
    """Agent interacting with environment.

    The “Critic” estimates the value function. This could be the action-value (the Q value) or state-value (the V value).

    The “Actor” updates the policy distribution in the direction suggested by the Critic (such as with policy gradients).

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
        args: DDPGArgments,
        env: gym.Env,
        state_shape: Union[int, List[int]] = None,
        action_shape: Union[int, List[int]] = None,
        action_bound: int = None,
        device='cpu',
    ) -> None:
        super().__init__(args)
        self.args = args
        self.env = env
        self.device = device
        self.action_bound = action_bound
        self.obs_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))
        self.global_update_step = 0
        self.target_model_update_step = 0

        # 策略网络
        self.actor_net = PolicyNet(self.obs_dim, self.args.hidden_dim,
                                   self.action_dim).to(device)
        # 价值网络
        self.critic_net = ValueNet(self.obs_dim, self.args.hidden_dim,
                                   self.action_dim).to(device)

        self.target_actor = copy.deepcopy(self.actor_net)
        self.target_critic = copy.deepcopy(self.critic_net)

        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(),
                                                lr=self.args.actor_lr)
        # 价值网络优化器
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(),
                                                 lr=self.args.critic_lr)
        self.device = device

    def get_action(self, obs: np.ndarray):
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action = self.actor_net(obs).detach().cpu().numpy()
        action = self.args.action_bound * np.clip(action, -1.0, 1.0)
        action = action.flatten()
        return action

    def predict(self, obs: np.ndarray):
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action = self.actor_net(obs).detach().cpu().numpy().flatten()
        action *= self.args.action_bound
        return action

    def learn(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update model with an episode data.

        Returns:
            loss (float)
        """
        obs = batch['obs']
        next_obs = batch['next_obs']
        action = batch['action']
        reward = batch['reward']
        done = batch['done']

        pred_q_values = self.critic_net(obs, action)
        with torch.no_grad():
            next_actions = self.target_actor(next_obs)
            next_q_values = self.target_critic(next_obs, next_actions)

        # 时序差分目标
        q_targets = reward + self.args.gamma * next_q_values * (1 - done)
        # 均方误差损失函数
        value_loss = F.mse_loss(pred_q_values, q_targets)
        # update value network
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # cal policy loss
        # For the policy function, our objective is to maximize the expected return
        # To calculate the policy loss, we take the derivative of the objective function with respect to the policy parameter.
        # Keep in mind that the actor (policy) function is differentiable, so we have to apply the chain rule.
        policy_loss = -torch.mean(self.critic_net(obs, self.actor_net(obs)))
        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        if self.global_update_step % self.target_model_update_step == 0:
            # 软更新策略网络
            soft_target_update(self.actor_net,
                               self.target_actor,
                               tau=self.args.soft_update_tau)
            # 软更新价值网络
            soft_target_update(self.critic_net,
                               self.target_critic,
                               tau=self.args.soft_update_tau)

        self.global_update_step += 1
        return policy_loss.item(), value_loss.item()
