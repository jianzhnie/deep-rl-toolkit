import copy
from typing import List, Optional, Union

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

    “Human-Level Control Through Deep Reinforcement Learning”. Mnih V. et al..
    2015.

    Args:
        args (RLArguments): Configuration object for the agent.
        env (Env): Environment object.
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
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(args)
        self.args = args
        self.env = env
        self.device = device
        self.global_update_step: int = 0
        self.gradient_steps: int = args.gradient_steps
        self.eps_greedy = args.eps_greedy_start
        self.learning_rate = args.learning_rate
        self.q_network = QNet(state_shape=state_shape,
                              action_shape=action_shape).to(device)
        self.q_target = copy.deepcopy(self.q_network)

        self.optimizer = torch.optim.Adam(params=self.q_network.parameters(),
                                          lr=self.args.learning_rate)
        self.lr_scheduler = LinearLR(
            optimizer=self.optimizer,
            start_factor=self.args.learning_rate,
            end_factor=self.args.min_learning_rate,
            total_iters=self.args.max_timesteps,
        )
        self.eps_greedy_scheduler = LinearDecayScheduler(
            args.eps_greedy_start,
            args.eps_greedy_end,
            max_steps=args.max_timesteps,
        )

    def get_action(self, obs: np.array) -> np.ndarray:
        """Get action from the actor network.

        Args:
            obs (torch.Tensor): Current observation

        Returns:
            np.ndarray: Selected action
        """
        if np.random.rand() < self.eps_greedy:
            action = self.env.action_space.sample()
        else:
            action = self.predict(obs)

        self.eps_greedy = max(self.eps_greedy_scheduler.step(),
                              self.args.eps_greedy_end)
        return action

    def predict(self, obs: np.array) -> int:
        """Predict an action given an observation.

        Args:
            obs (torch.Tensor): Current observation

        Returns:
            int: Selected action
        """
        if obs.ndim == 1:
            # If obs is 1-dimensional, we need to expand it to have batch_size = 1
            obs = np.expand_dims(obs, axis=0)

        obs = torch.Tensor(obs).to(self.device)
        q_values = self.q_network(obs)
        actions = torch.argmax(q_values, dim=1).item()
        return actions

    def learn(self, batch: dict[str, torch.Tensor]) -> float:
        """Perform a learning step.

        Args:
            batch (dict[str, torch.Tensor]): Batch of experience

        Returns:
            float: Loss value
        """
        obs = batch['obs'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        done = batch['done'].to(self.device)

        current_q_values = self.q_network(obs).gather(1, action.long())
        with torch.no_grad():
            next_q_values = self.q_target(next_obs).max(1, keepdim=True)[0]
        target_q_values = reward + (1 - done) * self.args.gamma * next_q_values

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(),
                                       self.args.max_grad_norm)
        self.optimizer.step()

        if self.global_update_step % self.args.target_update_frequency == 0:
            soft_target_update(self.q_network, self.q_target,
                               self.args.soft_update_tau)

        self.global_update_step += 1
        return loss.item()
