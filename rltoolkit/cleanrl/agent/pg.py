from typing import Any, List, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn
from rltoolkit.cleanrl.rl_args import PGArguments
from torch.distributions import Categorical


class PolicyNet(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.fc1(obs)
        out = self.relu(obs)
        out = self.fc2(obs)
        out = self.softmax(out)
        return out


class PGAgent(object):

    def __init__(
        self,
        args: PGArguments,
        env: gym.Env,
        state_shape: Union[int, List[int]],
        action_shape: Union[int, List[int]],
        device: Any = None,
    ) -> None:
        self.args = args
        self.env = env
        self.obs_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))

        self.policy_net = PolicyNet(self.obs_dim, self.args.hidden_dim,
                                    self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=self.args.learning_rate)
        self.device = device

    def get_action(self, obs: np.ndarray) -> Tuple[int, float]:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        prob = self.policy_net(obs)
        action_dist = Categorical(prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def predict(self, obs: np.ndarray) -> int:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        prob = self.policy_net(obs)
        action_dist = Categorical(prob)
        action = action_dist.sample()
        return action.item()

    def learn(self, log_probs: list, returns: list) -> None:
        """REINFORCE algorithm, also known as Monte Carlo Policy Gradients.

        Args:
            - log_probs:
            - returns:

        Return:
            loss (torch.tensor): shape of (1)
        """
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)

        loss = torch.cat(policy_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降
        return loss.item()

    def learn_with_baseline(self, log_probs: list, returns: list) -> float:
        baseline = np.mean(returns)
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * (G - baseline))

        loss = torch.cat(policy_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降
        return loss.item()
