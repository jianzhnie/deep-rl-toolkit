import copy
from typing import List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from rltoolkit.utils import LinearDecayScheduler, soft_target_update
from torch.optim.lr_scheduler import LinearLR

from cleanrl.base_agent import BaseAgent
from cleanrl.network import QNetwork
from cleanrl.rl_args import RLArguments


class DQNAgent(BaseAgent):
    """Deep Q-Network algorithm.

    “Human-Level Control Through Deep Reinforcement Learning”. Mnih V. et al..
    2015.

    Args:
        args (Namespace): argsuration object for the agent.
        envs (Union[None]): Environment object.
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

        self.q_network = QNetwork(state_shape=state_shape,
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

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
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

        if np.random.rand() < self.eps_greedy:
            action = self.env.action_space.sample()
        else:
            action = self.predict(obs)

        # update exploration
        self.eps_greedy = max(self.eps_greedy_scheduler.step(),
                              self.args.eps_greedy_end)
        return action

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
        q_values = self.q_network(obs)
        # Ensure self.q_network is a callable object
        actions = torch.argmax(q_values, dim=1).item()
        return actions

    def learn(self, batch: dict[str, torch.Tensor]) -> float:
        """DQN learner.

        Returns:
            dict[str, Union[float, int]]: Information about the learning process.
        """
        # Unpack the batch
        obs: torch.Tensor = batch.get('obs')
        next_obs: torch.Tensor = batch.get('next_obs')
        action: torch.Tensor = batch.get('action')
        reward: torch.Tensor = batch.get('reward')
        done: torch.Tensor = batch.get('done')

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
        target_q_values = reward + (1 - done) * self.args.gamma * next_q_values
        # TD loss
        loss = F.mse_loss(current_q_values, target_q_values)
        # Set the gradients to zero
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norm
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(),
                                       self.args.max_grad_norm)
        # Backward propagation to update parameters
        self.optimizer.step()

        # Update target network
        if self.global_update_step % self.args.target_update_frequency == 0:
            soft_target_update(
                src_model=self.q_network,
                tgt_model=self.q_target,
                tau=self.args.soft_update_tau,
            )
        self.global_update_step += 1
        return loss.item()
