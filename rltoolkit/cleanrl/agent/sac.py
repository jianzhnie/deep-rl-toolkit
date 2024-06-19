import copy
from typing import List, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn.functional as F
from rltoolkit.cleanrl.agent.base import BaseAgent
from rltoolkit.cleanrl.rl_args import SACArguments
from rltoolkit.cleanrl.utils.ac_net import ActorNet, CriticNet
from rltoolkit.data.utils.type_aliases import ReplayBufferSamples
from rltoolkit.utils import soft_target_update
from torch.distributions import Categorical


class Agent(BaseAgent):
    """Agent interacting with environment.

    The “Critic” estimates the value function. This could be the action-value (the Q value) or state-value (the V value).

    The “Actor” updates the policy distribution in the direction suggested by the Critic (such as with policy gradients).
    """

    def __init__(
        self,
        args: SACArguments,
        env: gym.Env,
        state_shape: Union[int, List[int]],
        action_shape: Union[int, List[int]],
        device: str = 'cpu',
    ) -> None:
        self.args = args
        self.env = env
        self.device = device
        self.obs_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))
        self.learner_update_step = 0
        self.target_model_update_step = 0

        # 策略网络
        self.actor = ActorNet(self.obs_dim, self.args.hidden_dim,
                              self.action_dim).to(device)
        # 价值网络
        self.critic1 = CriticNet(self.obs_dim, self.args.hidden_dim,
                                 self.action_dim).to(device)
        self.critic2 = CriticNet(self.obs_dim, self.args.hidden_dim,
                                 self.action_dim).to(device)

        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.args.actor_lr)
        # 价值网络优化器
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(),
                                                  lr=self.args.critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(),
                                                  lr=self.args.critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01),
                                      dtype=torch.float,
                                      device=device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=self.args.alpha_lr)

    def sample(self, obs: np.ndarray):
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.learner_update_step < self.args.warmup_learn_steps:
            action = self.env.action_space.sample()
        else:
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            logits = self.actor(obs)
            action_dist = Categorical(logits=logits)
            action = action_dist.sample()
        return action.item()

    def predict(self, obs) -> int:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(obs)
            action_dist = Categorical(logits=logits)
            action = action_dist.probs.argmax(dim=1, keepdim=True)
        return action.item()

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(
        self,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.actor(next_obs)
        action_dist = Categorical(logits=logits)
        entropy = action_dist.entropy()

        q1_value = self.target_critic1(next_obs)
        q2_value = self.target_critic2(next_obs)
        min_qvalue = torch.sum(action_dist * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = reward + self.args.gamma * next_value * (1 - done)
        return td_target

    def learn(self, batch: ReplayBufferSamples) -> Tuple[float, float]:
        """Update the model by TD actor-critic."""
        obs = batch.obs
        next_obs = batch.next_obs
        action = batch.actions
        reward = batch.rewards
        done = batch.dones

        action = action.long()
        # 更新两个Q网络
        td_target = self.calc_target(reward, next_obs, done)
        critic1_q_values = self.critic1(obs).gather(1, action)
        critic1_loss = F.mse_loss(critic1_q_values, td_target.detach())
        critic2_q_values = self.critic2(obs).gather(1, action)
        critic2_loss = F.mse_loss(critic2_q_values, td_target.detach())

        # critic1_loss backward
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        # critic1_loss backward
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 更新策略网络
        logits = self.actor(obs)
        action_dist = Categorical(logits=logits)
        # 直接根据概率计算熵
        entropy = action_dist.entropy()
        q1_value = self.critic1(obs)
        q2_value = self.critic2(obs)
        min_qvalue = torch.sum(action_dist * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        # backward
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean((entropy - self.args.target_entropy).detach() *
                                self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if self.learner_update_step % self.args.target_update_frequency == 0:
            soft_target_update(self.critic1, self.target_critic1)
            soft_target_update(self.critic2, self.target_critic2)
            self.target_model_update_step += 1

        self.learner_update_step += 1

        resullt = {
            'actor_loss': actor_loss.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
        }
        return resullt
