from typing import Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn
from rltoolkit.cleanrl.rl_args import PGArguments
from rltoolkit.data.utils.type_aliases import RolloutBufferSamples
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
        dist = Categorical(prob)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = 0
        return value, action.item(), log_prob.item(), entropy

    def predict(self, obs: np.ndarray) -> int:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = self.policy_net(obs)
            action_dist = Categorical(prob)
            action = action_dist.sample()
        return action.item()

    def learn(self, batch: RolloutBufferSamples) -> Dict[str, float]:
        """Update the model using a batch of sampled experiences.

        Args:
            batch (RolloutBufferSamples): A batch of sampled experiences.
            RolloutBufferSamples contains the following fields:
            - obs (torch.Tensor): The observations from the environment.
            - actions (torch.Tensor): The actions taken by the agent.
            - old_values (torch.Tensor): The value estimates from the critic.
            - old_log_prob (torch.Tensor): The log probabilities of the actions.
            - advantages (torch.Tensor): The advantages of the actions.
            - returns (torch.Tensor): The returns from the environment.

        Returns:
            Dict[str, float]: A dictionary containing the following metrics:
            - value_loss (float): The value loss of the critic.
            - actor_loss (float): The actor loss of the policy.
            - entropy_loss (float): The entropy loss of the policy.
            - approx_kl (float): The approximate KL divergence.
            - clipped_frac (float): The fraction of clipped actions.
        """
        log_probs = batch.old_log_prob
        returns = batch.returns
        use_baseline: bool = self.args.use_baseline

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

        return {'loss': loss.item()}
