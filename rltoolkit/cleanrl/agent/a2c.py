from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from rltoolkit.cleanrl.agent.base import BaseAgent
from rltoolkit.cleanrl.rl_args import A2CArguments
from rltoolkit.cleanrl.utils.pg_net import ActorCriticNet
from rltoolkit.data.utils.type_aliases import RolloutBufferSamples
from torch.distributions import Categorical


class A2CAgent(BaseAgent):
    """Agent interacting with environment. The “Critic” estimates the value
    function. This could be the action-value (the Q value) or state-value (the
    V value). The “Actor” updates the policy distribution in the direction
    suggested by the Critic (such as with policy gradients).

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
        args: A2CArguments,
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

        # Initialize actor and critic networks
        self.actor_critic = ActorCriticNet(self.obs_dim, self.args.hidden_dim,
                                           self.action_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(),
                                          lr=self.args.learning_rate)
        self.device = device

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
        value = self.actor_critic.get_value(obs_tensor)
        logits = self.actor_critic.get_action(obs_tensor)
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
        value = self.actor_critic.get_value(obs_tensor)
        return value.item()

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
            - approx_kl (float): The approximate KL divergence.
            - clipped_frac (float): The fraction of clipped actions.
        """
        obs = batch.obs
        actions = batch.actions
        advantages = batch.advantages
        returns = batch.returns

        # Compute new values, log probs, and entropy
        new_values = self.actor_critic.get_value(obs)
        logits = self.actor_critic.get_action(obs)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # actor loss
        actor_loss = torch.mean(-new_log_probs * advantages)

        # value loss
        value_loss = F.mse_loss(returns, new_values)

        # entropy loss
        entropy_loss = entropy.mean()

        loss = (actor_loss + value_loss * self.args.value_loss_coef -
                entropy_loss * self.args.entropy_coef)

        self.optimizer.zero_grad()
        loss.backward()
        if self.args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(),
                max_norm=self.args.max_grad_norm,
            )
        self.optimizer.step()
        self.learner_update_step += 1

        return {
            'loss': loss.item(),
            'value_loss': value_loss.item(),
            'actor_loss': actor_loss.item(),
            'entropy_loss': entropy_loss.item(),
        }
