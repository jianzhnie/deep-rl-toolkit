import copy
from typing import List, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from rltoolkit.cleanrl.agent.base import BaseAgent
from rltoolkit.cleanrl.rl_args import TD3Arguments
from rltoolkit.cleanrl.utils.ac_net import TD3Actor, TD3Critic
from rltoolkit.data.utils.type_aliases import ReplayBufferSamples
from rltoolkit.utils import soft_target_update


class TD3Agent(BaseAgent):
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
        args: TD3Arguments,
        env: gym.Env,
        state_shape: Union[int, List[int]],
        action_shape: Union[int, List[int]],
        device: str = 'cpu',
    ) -> None:
        super().__init__(args)
        self.args = args
        self.env = env
        self.device = device
        self.obs_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))
        self.action_high = self.env.action_space.high[0]
        self.action_low = self.env.action_space.low[0]
        self.learner_update_step = 0
        self.target_model_update_step = 0

        # 策略网络
        self.actor = TD3Actor(
            self.obs_dim,
            self.args.hidden_dim,
            self.action_dim,
            action_high=self.action_high,
            action_low=self.action_low,
        ).to(device)

        # 价值网络
        self.critic1 = TD3Critic(self.obs_dim, self.args.hidden_dim,
                                 self.action_dim).to(device)
        self.critic2 = TD3Critic(self.obs_dim, self.args.hidden_dim,
                                 self.action_dim).to(device)

        # target network
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.args.actor_lr)
        # 价值网络优化器
        # concat critic parameters to use one optim
        self.critic_parameters = list(self.critic1.parameters()) + list(
            self.critic2.parameters())

        self.critic_optimizer = torch.optim.Adam(self.critic_parameters,
                                                 lr=self.args.critic_lr)

    def get_action(self, obs: np.ndarray):
        if self.learner_update_step < self.args.warmup_learn_steps:
            action = self.env.action_space.sample()
        else:
            action = self.predict(obs)

        return action

    def predict(self, obs: np.ndarray):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            action = self.actor(obs)
            action += torch.normal(
                0, self.actor.action_scale * self.args.exploration_noise)
            action = action.cpu().numpy().clip(self.action_low,
                                               self.action_high)
        return action.flatten()

    def learn(self, batch: ReplayBufferSamples) -> Tuple[float, float]:
        """Update the model by TD actor-critic."""

        obs = batch.obs
        next_obs = batch.next_obs
        action = batch.actions
        reward = batch.rewards
        done = batch.dones

        with torch.no_grad():
            clipped_noise = (torch.randn_like(
                action, device=self.device) * self.args.policy_noise).clamp(
                    -self.args.noise_clip,
                    self.args.noise_clip) * self.target_actor.action_scale

            next_state_action = (self.target_actor(next_obs) +
                                 clipped_noise).clamp(self.action_low,
                                                      self.action_high)
            q1_next_target = self.target_critic1(next_obs, next_state_action)
            q2_next_target = self.target_critic2(next_obs, next_state_action)
            min_q_next_target = torch.min(q1_next_target, q2_next_target)
            next_q_value = reward + self.args.gamma * (
                1 - done) * min_q_next_target

        q1_value = self.critic1(obs, action)
        q2_value = self.critic2(obs, action)

        q1_loss = F.mse_loss(q1_value, next_q_value)
        q2_loss = F.mse_loss(q2_value, next_q_value)
        critic_loss = q1_loss + q2_loss

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # cal policy loss
        actor_loss = torch.tensor(0.0).to(self.device)
        if self.learner_update_step % self.args.policy_update_frequency == 0:
            pi = self.actor(obs)
            q1_pi = self.critic1(obs, pi)
            actor_loss = -torch.mean(q1_pi)

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新策略网络
            soft_target_update(self.actor, self.target_actor)
            # 软更新价值网络
            soft_target_update(self.critic1, self.target_critic1)
            soft_target_update(self.critic2, self.target_critic2)

        self.learner_update_step += 1

        learn_result = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'q1_value': q1_value.mean().item(),
            'q2_value': q2_value.mean().item(),
        }
        return learn_result
