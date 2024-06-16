from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from rltoolkit.cleanrl.agent.base import BaseAgent
from rltoolkit.cleanrl.rl_args import PPOArguments
from rltoolkit.cleanrl.utils.pg_net import PPOPolicyNet, PPOValueNet
from rltoolkit.data.utils.type_aliases import RolloutBufferSamples
from torch.distributions import Categorical


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization (PPO) Agent.

    The agent interacts with the environment using an actor-critic model.
    The actor updates the policy distribution based on the critic's feedback.

    Args:
        args (PPOArguments): Configuration arguments for PPO.
        env (gym.Env): Environment to interact with.
        state_shape (Union[int, List[int]]): Shape of the state space.
        action_shape (Union[int, List[int]]): Shape of the action space.
        device (Optional[Union[str, torch.device]]): Device for computations.
    """

    def __init__(
        self,
        args: PPOArguments,
        env: gym.Env,
        state_shape: Union[int, List[int]],
        action_shape: Union[int, List[int]],
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:

        self.args = args
        self.env = env
        self.device = device if device is not None else torch.device('cpu')
        self.obs_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))

        # Initialize actor and critic networks
        self.actor = PPOPolicyNet(self.obs_dim, self.args.hidden_dim,
                                  self.action_dim).to(self.device)
        self.critic = PPOValueNet(self.obs_dim,
                                  self.args.hidden_dim).to(self.device)

        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.args.critic_lr)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.args.learning_rate,
        )

    def get_action(self, obs: np.ndarray) -> Tuple[float, int, float, float]:
        """Sample an action from the policy given an observation.

        Args:
            obs (np.ndarray): The observation from the environment.

        Returns:
            Tuple[float, int, float, float]: A tuple containing:
                - value (float): The value estimate from the critic.
                - action (int): The sampled action.
                - log_prob (float): The log probability of the sampled action.
                - entropy (float): The entropy of the action distribution.
        """
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        value = self.critic(obs_tensor)
        logits = self.actor(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return value.item(), action.item(), log_prob.item(), entropy.item()

    def get_value(self, obs: np.ndarray) -> float:
        """Use the critic model to predict the value of an observation.

        Args:
            obs (np.ndarray): The observation from the environment.

        Returns:
            float: The predicted value of the observation.
        """
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        value = self.critic(obs_tensor)
        return value.item()

    def predict(self, obs: np.ndarray) -> int:
        """Predict the action with the highest probability given an
        observation.

        Args:
            obs (np.ndarray): The observation from the environment.

        Returns:
            int: The predicted action.
        """
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.probs.argmax(dim=1, keepdim=True)
        return action.item()

    def learn(self, batch: RolloutBufferSamples) -> Tuple[float, float]:
        """Update the model using a batch of sampled experiences.

        Args:
            batch (RolloutBufferSamples): A batch of sampled experiences.

        Returns:
            Tuple[float, float]: A tuple containing:
                - value_loss (float): The loss for the value function.
                - actor_loss (float): The loss for the policy function.
        """
        obs = batch.obs
        actions = batch.actions
        old_values = batch.old_values
        old_log_probs = batch.old_log_prob
        advantages = batch.advantages
        returns = batch.returns

        # Compute new values, log probs, and entropy
        new_values = self.critic(obs)
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # Compute entropy loss
        entropy_loss = entropy.mean()

        # Compute actor loss
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        surr1 = ratio * advantages
        surr2 = (torch.clamp(ratio, 1.0 - self.args.clip_param,
                             1.0 + self.args.clip_param) * advantages)
        actor_loss = -torch.min(surr1, surr2).mean()

        # Compute value loss
        if self.args.clip_vloss:
            assert self.args.clip_param is not None, 'clip_param must be set'
            value_pred_clipped = old_values + torch.clamp(
                new_values - old_values, -self.args.clip_param,
                self.args.clip_param)
            value_losses_unclipped = (new_values - returns).pow(2)
            value_losses_clipped = (value_pred_clipped - returns).pow(2)
            value_loss = (
                0.5 *
                torch.max(value_losses_unclipped, value_losses_clipped).mean())
        else:
            value_loss = 0.5 * (new_values - returns).pow(2).mean()

        # Total loss
        loss = (value_loss * self.args.value_loss_coef + actor_loss -
                entropy_loss * self.args.entropy_coef)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        if self.args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                           self.args.max_grad_norm)
        self.optimizer.step()

        # Calculate KL divergence metrics
        with torch.no_grad():
            old_approx_kl = (-log_ratio).mean()
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clipfrac = (abs(ratio - 1.0) >
                        self.args.clip_param).float().mean().item()

        return {
            'value_loss': value_loss.item(),
            'actor_loss': actor_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'old_approx_kl': old_approx_kl.item(),
            'approx_kl': approx_kl.item(),
            'clipfrac': clipfrac,
        }
