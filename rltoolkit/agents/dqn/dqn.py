import copy
import warnings
from typing import List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from rltoolkit.agents.base_agent import BaseAgent
from rltoolkit.agents.configs import BaseConfig
from rltoolkit.agents.network import QNetwork
from rltoolkit.utils import LinearDecayScheduler
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from torch.optim.lr_scheduler import LinearLR


class DQNAgent(BaseAgent):
    """Deep Q-Network algorithm.

    “Human-Level Control Through Deep Reinforcement Learning”. Mnih V. et al..
    2015.

    Args:
        config (Namespace): Configuration object for the agent.
        envs (Union[None]): Environment object.
    """

    def __init__(
        self,
        config: BaseConfig,
        envs: Union[gym.Env, BaseVectorEnv],
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        if isinstance(envs, gym.Env) and not hasattr(envs, '__len__'):
            warnings.warn(
                'Single environment detected, wrap to DummyVectorEnv.',
                stacklevel=2,
            )
            self.envs = DummyVectorEnv([lambda: envs])  # type: ignore
        else:
            self.envs = envs  # type: ignore
        self.action_space = self.envs.action_space

        self.device = device
        self.gradient_steps: int = config.gradient_steps

        self.q_network = QNetwork(state_shape=config.state_shape,
                                  action_shape=config.action_shape).to(device)
        self.q_target = copy.deepcopy(self.q_network)

        self.optimizer = torch.optim.Adam(params=self.q_network.parameters(),
                                          lr=self.config.learning_rate)
        self.lr_scheduler = LinearLR(
            optimizer=self.optimizer,
            start_factor=self.config.learning_rate,
            end_factor=self.config.min_learning_rate,
            total_iters=self.config.max_timesteps,
        )
        self.eps_greedy_scheduler = LinearDecayScheduler(
            config.eps_greedy_start,
            config.eps_greedy_end,
            max_steps=config.max_timesteps,
        )

    def get_action(self, obs: torch.Tensor, eps_greedy: float) -> torch.Tensor:
        """Get action from the actor network.

        Sample an action when given an observation, based on the current
        epsilon value, either a greedy action or a random action will be
        returned.

        Args:
            obs: Current observation

        Returns:
            actions (np.array): Action
        """
        # Choose a random action with probability epsilon

        if np.random.rand() < eps_greedy:
            try:
                actions = [
                    self.action_space[i].sample()
                    for i in range(self.config.num_envs)
                ]
            except TypeError:  # envpool's action space is not for per-env
                actions = [
                    self.action_space.sample()
                    for _ in range(self.config.num_envs)
                ]
        else:
            actions = self.predict(obs)
        return actions

    def predict(self, obs: torch.Tensor) -> Union[int, List[int]]:
        """Predict an action when given an observation, a greedy action will be
        returned.

        Args:
            obs (torch.Tensor): Current observation

        Returns:
            actions (Union[int, List[int]]): Action
        """
        if obs.ndim == 1:
            # If obs is 1-dimensional, we need to expand it to have batch_size = 1
            obs = np.expand_dims(obs, axis=0)

        obs = torch.Tensor(obs).to(self.device)
        q_values = self.q_network(
            obs)  # Ensure self.q_network is a callable object
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        return actions

    def learn(self, batch: dict[str, torch.Tensor]) -> float:
        """DQN learner.

        Returns:
            dict[str, Union[float, int]]: Information about the learning process.
        """
        # Unpack the batch
        obs: torch.Tensor = batch.observations
        action: torch.Tensor = batch.actions
        reward: torch.Tensor = batch.rewards
        next_obs: torch.Tensor = batch.next_observations
        dones: torch.Tensor = batch.dones

        # Prediction Q(s)
        current_q_values = self.q_network(obs)
        # Retrieve the q-values for the actions from the replay buffer
        current_q_values = torch.gather(current_q_values,
                                        dim=1,
                                        index=action.long())

        # Target for Q regression
        with torch.no_grad():
            next_q_values = self.q_target(next_obs)
            # Follow greedy policy: use the one with the highest value
            next_q_values, _ = next_q_values.max(dim=1)
            # Avoid potential broadcast issue
            next_q_values = next_q_values.reshape(-1, 1)

        # TD target
        target_q_values = reward + (1 -
                                    dones) * self.config.gamma * next_q_values
        # TD loss
        loss = F.mse_loss(current_q_values, target_q_values)
        # Set the gradients to zero
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norm
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(),
                                       self.config.max_grad_norm)
        # Backward propagation to update parameters
        self.optimizer.step()

        return loss.item()
