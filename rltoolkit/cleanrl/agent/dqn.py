import copy
from typing import Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from rltoolkit.cleanrl.agent.base import BaseAgent
from rltoolkit.cleanrl.rl_args import DQNArguments
from rltoolkit.cleanrl.utils.qlearing_net import DuelingNet, NoisyNet, QNet
from rltoolkit.data.utils.type_aliases import ReplayBufferSamples
from rltoolkit.utils import LinearDecayScheduler, soft_target_update


class DQNAgent(BaseAgent):
    """Implementation of the Deep Q-Network (DQN) algorithm.

    Based on the paper "Human-Level Control Through Deep Reinforcement Learning" by Mnih V. et al., 2015.

    This class also supports variants such as Double DQN, Dueling DQN, and Noisy DQN.

    Args:
        args (DQNArguments): Configuration object for the agent.
        env (gym.Env): The environment object.
        state_shape (Union[int, List[int]]): Shape of the state space.
        action_shape (Union[int, List[int]]): Shape of the action space.
        device (Optional[Union[str, torch.device]]): The device to use for computation (CPU/GPU).
    """

    def __init__(
        self,
        args: DQNArguments,
        env: gym.Env,
        state_shape: Union[int, List[int]],
        action_shape: Union[int, List[int]],
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(args)

        # Validate N-step is a positive integer
        assert (isinstance(args.n_steps, int)
                and args.n_steps > 0), 'N-step should be a positive integer.'

        self.args = args
        self.env = env
        self.device = device or torch.device('cpu')
        # Default to CPU if not specified
        self.learner_update_step = 0
        self.target_model_update_step = 0
        self.eps_greedy = args.eps_greedy_start
        self.learning_rate = args.learning_rate

        # Flatten state and action shapes
        self.obs_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))

        # Initialize Q-network and Target network
        if args.dueling_dqn:
            self.qnet = DuelingNet(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=args.hidden_dim,
            ).to(self.device)
        elif args.noisy_dqn:
            self.qnet = NoisyNet(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=args.hidden_dim,
            ).to(self.device)
        else:
            self.qnet = QNet(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=args.hidden_dim,
            ).to(self.device)

        # Target network is a copy of the Q-network
        self.target_qnet = copy.deepcopy(self.qnet)
        self.target_qnet.eval()  # Target network is in evaluation mode

        # Optimizer and learning rate scheduler
        self.optimizer = torch.optim.Adam(params=self.qnet.parameters(),
                                          lr=args.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer,
            start_factor=args.learning_rate,
            end_factor=args.min_learning_rate,
            total_iters=args.max_timesteps,
        )

        # Epsilon-greedy scheduler for exploration
        self.eps_greedy_scheduler = LinearDecayScheduler(
            start_value=args.eps_greedy_start,
            end_value=args.eps_greedy_end,
            max_steps=int(args.max_timesteps * 0.8),
        )

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Select an action using epsilon-greedy policy or NoisyNet.

        Args:
            obs (np.ndarray): The current observation.

        Returns:
            np.ndarray: The selected action.
        """
        if self.args.noisy_dqn:
            # NoisyNet bypasses epsilon-greedy exploration
            action = self.predict(obs)
        else:
            # Epsilon-greedy policy: random action with probability epsilon
            if np.random.rand() <= self.eps_greedy:
                action = self.env.action_space.sample()  # Explore
            else:
                action = self.predict(obs)  # Exploit

            # Decay epsilon over time
            self.eps_greedy = max(self.eps_greedy_scheduler.step(),
                                  self.args.eps_greedy_end)

        return action

    def predict(self, obs: np.ndarray) -> int:
        """Predict the action with the highest Q-value for a given observation.

        Args:
            obs (np.ndarray): The current observation.

        Returns:
            int: The action with the highest predicted Q-value.
        """
        # If the observation is a single state, expand its dimensions
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

        # Forward pass through the Q-network to get the action with the highest Q-value
        with torch.no_grad():
            action = self.qnet(obs_tensor).argmax().item()

        return action

    def learn(self, batch: ReplayBufferSamples) -> Dict[str, float]:
        """Perform a learning step by updating the Q-network based on the given
        batch of experiences.

        Args:
            batch (ReplayBufferSamples): A batch of experience samples from the replay buffer.

        Returns:
            Dict[str, float]: A dictionary containing the loss value.
        """
        # Move tensors to the correct device
        obs = batch.obs.to(self.device)
        next_obs = batch.next_obs.to(self.device)
        action = batch.actions.to(self.device, dtype=torch.long)
        reward = batch.rewards.to(self.device)
        done = batch.dones.to(self.device)

        # Compute current Q-values
        current_q_values = self.qnet(obs).gather(1, action)

        # Compute target Q-values based on the next state
        with torch.no_grad():
            if self.args.noisy_dqn:
                next_q_values = self.target_qnet(next_obs).max(dim=1,
                                                               keepdim=True)[0]
            elif self.args.double_dqn:
                # Double DQN: use the Q-network to select actions, and the target network to evaluate them
                greedy_action = self.qnet(next_obs).max(dim=1, keepdim=True)[1]
                next_q_values = self.target_qnet(next_obs).gather(
                    1, greedy_action)
            else:
                # Vanilla DQN
                next_q_values = self.target_qnet(next_obs).max(dim=1,
                                                               keepdim=True)[0]

        # Calculate target Q-values using the Bellman equation
        target_q_values = (
            reward +
            (1 - done) * self.args.gamma**self.args.n_steps * next_q_values)

        # Compute loss (Mean Squared Error between current and target Q-values)
        loss = F.mse_loss(current_q_values, target_q_values, reduction='mean')

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (if enabled)
        if self.args.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(),
                                           self.args.max_grad_norm)

        self.optimizer.step()

        # If NoisyNet is enabled, reset noise after each update
        if self.args.noisy_dqn:
            self.qnet.reset_noise()
            self.target_qnet.reset_noise()

        # Soft update of the target network at regular intervals
        if self.learner_update_step % self.args.target_update_frequency == 0:
            soft_target_update(self.qnet, self.target_qnet,
                               self.args.soft_update_tau)
            self.target_model_update_step += 1

        self.learner_update_step += 1

        learn_result = {
            'loss': loss.item(),
        }
        # Return the loss as a dictionary for logging
        return learn_result
