import copy
from typing import Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from rltoolkit.utils import LinearDecayScheduler, soft_target_update
from torch.optim.lr_scheduler import LinearLR

from cleanrl.base_agent import BaseAgent
from cleanrl.network import QNet
from cleanrl.rl_args import RLArguments


class DQNAgent(BaseAgent):
    """Deep Q-Network algorithm.

    "Human-Level Control Through Deep Reinforcement Learning" - Mnih V. et al., 2015.

    Args:
        args (RLArguments): Configuration object for the agent.
        env (gym.Env): Environment object.
        state_shape (Optional[Union[int, List[int]]]): Shape of the state.
        action_shape (Optional[Union[int, List[int]]]): Shape of the action.
        device (Optional[Union[str, torch.device]]): Device to use for computation.
    """

    def __init__(
        self,
        args: RLArguments,
        env: gym.Env,
        state_shape: Optional[Union[int, List[int]]] = None,
        action_shape: Optional[Union[int, List[int]]] = None,
        double_dqn: Optional[bool] = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(args)
        self.args = args
        self.env = env
        self.double_dqn = double_dqn
        self.device = device
        self.global_update_step = 0
        self.target_model_update_step = 0
        self.eps_greedy = args.eps_greedy_start
        self.learning_rate = args.learning_rate

        # Initialize networks
        self.q_network = QNet(state_shape=state_shape,
                              action_shape=action_shape).to(device)
        self.q_target = copy.deepcopy(self.q_network)

        # Initialize optimizer and schedulers
        self.optimizer = torch.optim.Adam(params=self.q_network.parameters(),
                                          lr=self.learning_rate)
        self.lr_scheduler = LinearLR(
            optimizer=self.optimizer,
            start_factor=self.learning_rate,
            end_factor=args.min_learning_rate,
            total_iters=args.max_timesteps,
        )
        self.eps_greedy_scheduler = LinearDecayScheduler(
            args.eps_greedy_start,
            args.eps_greedy_end,
            max_steps=args.max_timesteps * 0.8,
        )

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from the actor network.

        Args:
            obs (np.ndarray): Current observation.

        Returns:
            np.ndarray: Selected action.
        """
        if np.random.rand() < self.eps_greedy:
            action = self.env.action_space.sample()
        else:
            action = self.predict(obs)

        self.eps_greedy = max(self.eps_greedy_scheduler.step(),
                              self.args.eps_greedy_end)
        return action

    def predict(self, obs: np.ndarray) -> int:
        """Predict an action given an observation.

        Args:
            obs (np.ndarray): Current observation.

        Returns:
            int: Selected action.
        """
        if obs.ndim == 1:
            # Expand to have batch_size = 1
            obs = np.expand_dims(obs, axis=0)

        obs = torch.Tensor(obs).to(self.device)
        q_values = self.q_network(obs)
        action = torch.argmax(q_values, dim=1).item()
        return action

    def learn(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform a learning step.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of experience.

        Returns:
            float: Loss value.
        """
        obs = batch['obs'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        done = batch['done'].to(self.device)

        # Compute current Q values
        current_q_values = self.q_network(obs).gather(1, action.long())

        # Compute target Q values
        if self.double_dqn:
            greedy_action = self.q_network(next_obs).max(dim=1,
                                                         keepdim=True)[1]
            next_q_values = self.q_target(next_obs).gather(1, greedy_action)
        else:
            next_q_values = self.q_target(next_obs).max(1, keepdim=True)[0]

        target_q_values = reward + (1 - done) * self.args.gamma * next_q_values

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        if self.args.clip_weights and self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(),
                                           self.args.max_grad_norm)
        self.optimizer.step()

        # Soft update target network
        if self.global_update_step % self.args.target_update_frequency == 0:
            soft_target_update(self.q_network, self.q_target,
                               self.args.soft_update_tau)
            self.target_model_update_step += 1

        self.global_update_step += 1
        return loss.item()
