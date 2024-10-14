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
    """A2C (Advantage Actor-Critic) agent for interacting with an environment.

    This agent uses an actor-critic model:
    - The Actor: updates the policy based on the action probabilities.
    - The Critic: estimates the value of a state (or action) and provides feedback to the Actor.

    Args:
        args (A2CArguments): Configuration arguments for the agent.
        env (gym.Env): The environment the agent interacts with.
        state_shape (Union[int, List[int]]): The shape of the state/observation space.
        action_shape (Union[int, List[int]]): The shape of the action space.
        device (Optional[Union[str, torch.device]]): The device to run computations on (CPU or GPU).
    """

    def __init__(
        self,
        args: A2CArguments,
        env: gym.Env,
        state_shape: Union[int, List[int]],
        action_shape: Union[int, List[int]],
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(args)
        self.args = args
        self.env = env
        # Default to CPU if no device provided
        # Define the dimensions of the state and action spaces
        self.obs_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))
        self.learner_update_step = 0
        self.device = device or torch.device('cpu')

        # Initialize the actor-critic network
        self.actor_critic = ActorCriticNet(self.obs_dim, self.args.hidden_dim,
                                           self.action_dim).to(self.device)

        # All Parameters
        self.all_parameters = self.actor_critic.parameters()
        # Set up optimizer
        self.optimizer = torch.optim.Adam(
            self.all_parameters,
            lr=self.args.learning_rate,
            eps=self.args.epsilon,
        )

    def get_action(self, obs: np.ndarray) -> Tuple[float, int, float, float]:
        """Sample an action based on the given observation from the
        environment.

        Args:
            obs (np.ndarray): The observation from the environment.

        Returns:
            Tuple[float, int, float, float]:
                - value (float): The value estimate of the state from the critic.
                - action (int): The sampled action.
                - log_prob (float): The log-probability of the sampled action.
                - entropy (float): The entropy of the action distribution (a measure of randomness).
        """
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        value = self.actor_critic.get_value(obs_tensor)
        logits = self.actor_critic.get_action(obs_tensor)
        dist = Categorical(logits=logits)

        # Sample action and calculate log probability and entropy
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return value.item(), action.item(), log_prob.item(), entropy

    def get_value(self, obs: np.ndarray) -> float:
        """Get the value estimate of the given observation using the critic.

        Args:
            obs (np.ndarray): The observation from the environment.

        Returns:
            float: The estimated value of the state.
        """
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        value = self.actor_critic.get_value(obs_tensor)
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
            logits = self.actor_critic.get_action(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.probs.argmax(dim=1, keepdim=True)
        return action.item()

    def learn(self, batch: RolloutBufferSamples) -> Dict[str, float]:
        """Perform a learning step using a batch of sampled experiences.

        Args:
            batch (RolloutBufferSamples): A batch of experiences containing:
                - obs (torch.Tensor): Observations.
                - actions (torch.Tensor): Actions taken.
                - old_values (torch.Tensor): Old value estimates.
                - old_log_prob (torch.Tensor): Log probabilities of the actions.
                - advantages (torch.Tensor): Advantages for each action.
                - returns (torch.Tensor): Estimated returns.

        Returns:
            Dict[str, float]: A dictionary with various loss metrics:
                - value_loss (float): Loss from the value function.
                - actor_loss (float): Loss from the policy (actor).
                - entropy_loss (float): Loss from the entropy term.
                - total_loss (float): The combined total loss.
        """
        # Unpack batch data
        obs = batch.obs
        actions = batch.actions
        advantages = batch.advantages
        returns = batch.returns

        # Convert discrete action from float to long
        actions = actions.long()

        # Compute new value estimates and log probabilities
        new_values = self.actor_critic.get_value(obs)
        logits = self.actor_critic.get_action(obs)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # Normalize advantage (not present in the original implementation)
        if self.args.norm_advantages:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8).to(self.device)

        log_prob = new_log_probs.reshape(len(advantages), -1).transpose(0, 1)
        # Actor (policy) loss: Maximize log_probs weighted by advantages
        actor_loss = -(log_prob * advantages).mean()

        # Critic (value) loss: Mean squared error between returns and new values
        value_loss = F.mse_loss(returns, new_values)

        # Entropy loss: Mean entropy of the action distribution
        entropy_loss = -torch.mean(entropy)

        # Total loss
        total_loss = (actor_loss + self.args.value_loss_coef * value_loss -
                      self.args.entropy_coef * entropy_loss)

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping (if specified)
        if self.args.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.all_parameters,
                                           self.args.max_grad_norm)

        self.optimizer.step()

        # Track learner update step
        self.learner_update_step += 1

        # Return loss metrics
        return {
            'total_loss': total_loss.item(),
            'value_loss': value_loss.item(),
            'actor_loss': actor_loss.item(),
            'entropy_loss': entropy_loss.item(),
        }
