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

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.args.learning_rate,
        )
        self.device = device

    def get_action(self, obs: np.ndarray) -> Tuple[int, float]:
        """Define the sampling process. This function returns the action
        according to action distribution.

        Args:
            obs (torch tensor): observation, shape([batch_size] + obs_shape)
        Returns:
            value (torch tensor): value, shape([batch_size, 1])
            action (torch tensor): action, shape([batch_size] , action_shape)
            action_log_probs (torch tensor): action log probs, shape([batch_size])
            action_entropy (torch tensor): action entropy, shape([batch_size])
        """
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        value = self.critic(obs)
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return (
            value.item(),
            action.item(),
            log_prob.item(),
            entropy.item(),
        )

    def get_value(self, obs: np.ndarray) -> float:
        """Use the critic model to predict obs values.

        Args:
            obs (torch tensor): observation, shape([batch_size] + obs_shape)
        Returns:
            value (torch tensor): value of obs, shape([batch_size])
        """
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        value = self.critic(obs)
        return value

    def predict(self, obs: np.array):
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        action = dist.probs.argmax(dim=1, keepdim=True)
        return action.item()

    def learn(self, batch: RolloutBufferSamples) -> Tuple[float, float]:
        """Update the model by TD actor-critic."""
        obs = batch.obs
        actions = batch.actions
        old_values = batch.old_values
        old_log_probs = batch.old_log_prob
        advantages = batch.advantages
        returns = batch.returns

        # Compute the value, action log probs, and entropy
        new_values = self.critic(obs)
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # entropy Loss
        entropy_loss = entropy.mean()

        # actor Loss
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        surr1 = ratio * advantages
        surr2 = (torch.clamp(ratio, 1.0 - self.args.clip_param,
                             1.0 + self.args.clip_param) * advantages)
        actor_loss = -torch.min(surr1, surr2).mean()

        # value Loss
        if self.args.clip_vloss:
            value_pred_clipped = old_values + torch.clamp(
                new_values - old_values,
                -self.args.clip_param,
                self.args.clip_param,
            )
            value_losses_unclipped = (new_values - returns).pow(2)
            value_losses_clipped = (value_pred_clipped - returns).pow(2)
            value_loss = (
                0.5 *
                torch.max(value_losses_unclipped, value_losses_clipped).mean())
        else:
            value_loss = 0.5 * (new_values - returns).pow(2).mean()

        loss = (value_loss * self.args.value_loss_coef + actor_loss -
                entropy_loss * self.args.entropy_coef)
        self.optimizer.zero_grad()
        loss.backward()
        if self.args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                           self.args.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            old_approx_kl = (-log_ratio).mean()
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clipfrac = (abs(
                (ratio - 1.0)) > self.args.clip_param).float().mean().item()

        result = {
            'value_loss': value_loss.item(),
            'actor_loss': actor_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'old_approx_kl': old_approx_kl.item(),
            'approx_kl': approx_kl.item(),
            'clipfrac': clipfrac,
        }

        return result
