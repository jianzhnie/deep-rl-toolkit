from typing import List, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn
from rltoolkit.cleanrl.rl_args import PGArguments
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    """Neural network that represents the policy for the REINFORCE algorithm.

    Args:
        obs_dim (int): Dimension of the observation space
        hidden_dim (int): Number of hidden units in the network
        action_dim (int): Dimension of the action space
    """

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.fc1(obs)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


class PGAgent(object):

    def __init__(
        self,
        args: PGArguments,
        env: gym.Env,
        state_shape: Union[int, List[int]],
        action_shape: Union[int, List[int]],
        device: Optional[torch.device] = None,
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

    def learn(self,
              log_probs: List[torch.Tensor],
              returns: List[float],
              use_baseline: bool = False) -> float:
        """REINFORCE algorithm implementation with optional baseline.

        Args:
            log_probs (List[torch.Tensor]): Log probabilities of taken actions
            returns (List[float]): Discounted returns for each timestep
            use_baseline (bool): Whether to use baseline subtraction. Defaults to False.

        Returns:
            float: The policy loss value after optimization
        """
        self.optimizer.zero_grad()

        returns_tensor = torch.tensor(returns, device=self.device)

        if use_baseline:
            # Subtract baseline (mean of returns) to reduce variance
            baseline = returns_tensor.mean()
            advantage = returns_tensor - baseline
        else:
            advantage = returns_tensor

        # Calculate policy loss
        policy_loss = []
        for log_prob, adv in zip(log_probs, advantage):
            policy_loss.append(-log_prob * adv)

        loss = torch.stack(policy_loss).sum()

        # Backpropagate and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),
                                       max_norm=0.5)
        self.optimizer.step()

        return loss.item()
