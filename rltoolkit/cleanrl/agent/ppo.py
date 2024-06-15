from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from rltoolkit.cleanrl.agent.base import BaseAgent
from rltoolkit.cleanrl.rl_args import PPOArguments
from rltoolkit.cleanrl.utils.pg_net import PPOPolicyNet, PPOValueNet
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence


class PPOAgent(BaseAgent):
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
        args: PPOArguments,
        env: gym.Env,
        state_shape: Union[int, List[int]] = None,
        action_shape: Union[int, List[int]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.args = args
        self.env = env
        self.device = device
        self.obs_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))
        self.learner_update_step = 0
        self.target_model_update_step = 0

        # 策略网络
        self.actor = PPOPolicyNet(self.obs_dim, self.args.hidden_dim,
                                  self.action_dim).to(device)
        # 价值网络
        self.critic = PPOValueNet(self.obs_dim,
                                  self.args.hidden_dim).to(device)

        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.args.actor_lr)
        # 价值网络优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.args.critic_lr)
        self.device = device

    def get_action(self, obs: np.ndarray) -> Tuple[int, float]:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        (action, log_prob, _) = self.actor(obs)
        return action.item()

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    def learn(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """Update the model by TD actor-critic."""
        obs = batch['obs']
        next_obs = batch['next_obs']
        actions = batch['actions']
        rewards = batch['rewards']
        dones = batch['dones']

        current_value = self.critic(obs)
        # 时序差分目标
        td_target = rewards + self.args.gamma * self.critic(next_obs) * (1 -
                                                                         dones)
        # 时序差分误差
        td_delta = td_target - current_value

        advantage = self.compute_advantage(self.args.gamma, self.args.lmbda,
                                           td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(obs).gather(1, actions)).detach()
        old_action_dists = Categorical(self.actor(obs).detach())

        for _ in range(self.args.policy_net_iters):
            log_probs = torch.log(self.actor(obs).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            # 截断
            surr2 = (torch.clamp(ratio, 1 - self.args.clip_param,
                                 1 + self.args.clip_param) * advantage)
            # PPO损失函数
            policy_loss = torch.mean(-torch.min(surr1, surr2))

            # K-L dist
            new_action_dists = Categorical(self.actor(obs))
            # A sample estimate for KL-divergence, easy to compute
            # approx_kl = (old_log_probs - log_probs).mean()
            kl_div = torch.mean(
                kl_divergence(old_action_dists, new_action_dists))
            # Early stopping at step i due to reaching max kl
            if kl_div > 1.5 * self.args.target_kl:
                print(
                    'Early stopping, due to current kl_div: %3f reaching max kl %3f'
                    % (kl_div, self.args.target_kl))
                break
            # update policy
            self.actor_optimizer.zero_grad()
            policy_loss.backward()  # 计算策略网络的梯度
            self.actor_optimizer.step()  # 更新策略网络的参数

        for _ in range(self.args.critic_net_iters):
            # value loss
            value_loss = F.mse_loss(self.critic(obs), td_target.detach())
            # update value
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

        return policy_loss.item(), value_loss.item()
