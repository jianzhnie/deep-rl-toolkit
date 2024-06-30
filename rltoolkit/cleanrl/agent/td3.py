import copy
from typing import Dict, List, Optional, Union

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
    """Agent interacting with environment using the Twin Delayed Deep
    Deterministic Policy Gradient (TD3) algorithm.

    Args:
        args (TD3Arguments): Configuration arguments for the TD3 agent.
        env (gym.Env): The environment to interact with.
        state_shape (Union[int, List[int]]): Shape of the state space.
        action_shape (Union[int, List[int]]): Shape of the action space.
        device (torch.device): The device to run the computations on (CPU/GPU).
    """

    def __init__(
        self,
        args: TD3Arguments,
        env: gym.Env,
        state_shape: Union[int, List[int]],
        action_shape: Union[int, List[int]],
        device: Optional[Union[str, torch.device]] = None,
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

        # Initialize the actor network
        self.actor = TD3Actor(
            self.obs_dim,
            self.args.hidden_dim,
            self.action_dim,
            action_high=self.action_high,
            action_low=self.action_low,
        ).to(self.device)

        # Initialize the critic networks
        self.critic1 = TD3Critic(self.obs_dim, self.args.hidden_dim,
                                 self.action_dim).to(self.device)
        self.critic2 = TD3Critic(self.obs_dim, self.args.hidden_dim,
                                 self.action_dim).to(self.device)

        # Initialize the target networks
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.args.actor_lr)
        self.critic_parameters = list(self.critic1.parameters()) + list(
            self.critic2.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic_parameters,
                                                 lr=self.args.critic_lr)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action based on the current observation.

        Args:
            obs (np.ndarray): The current observation.

        Returns:
            np.ndarray: The action to take.
        """
        if self.learner_update_step < self.args.warmup_learn_steps:
            action = self.env.action_space.sample()
        else:
            action = self.predict(obs)
        return action

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict action based on the observation using the actor network.

        Args:
            obs (np.ndarray): The current observation.

        Returns:
            np.ndarray: The predicted action.
        """
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(
                self.device)
            action = self.actor(obs_tensor)
            action += torch.normal(
                0, self.actor.action_scale * self.args.exploration_noise)
            action = action.cpu().numpy().clip(self.action_low,
                                               self.action_high)
        return action.flatten()

    def learn(self, batch: ReplayBufferSamples) -> Dict[str, float]:
        """Update the model using the TD3 algorithm.

        Args:
            batch (ReplayBufferSamples): A batch of experience from the replay buffer.

        Returns:
            Dict[str, float]: Dictionary containing the losses and values from the learning step.
        """
        obs = batch.obs.to(self.device)
        next_obs = batch.next_obs.to(self.device)
        action = batch.actions.to(self.device)
        reward = batch.rewards.to(self.device)
        done = batch.dones.to(self.device)

        with torch.no_grad():
            clipped_noise = (
                torch.randn_like(action) * self.args.policy_noise).clamp(
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

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = torch.tensor(0.0).to(self.device)
        if self.learner_update_step % self.args.actor_update_frequency == 0:
            pi = self.actor(obs)
            q1_pi = self.critic1(obs, pi)
            actor_loss = -q1_pi.mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update the target networks
            soft_target_update(self.actor, self.target_actor,
                               self.args.soft_update_tau)
            soft_target_update(self.critic1, self.target_critic1,
                               self.args.soft_update_tau)
            soft_target_update(self.critic2, self.target_critic2,
                               self.args.soft_update_tau)

        self.learner_update_step += 1

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'q1_value': q1_value.mean().item(),
            'q2_value': q2_value.mean().item(),
        }
