import copy
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from rltoolkit.cleanrl.agent.base import BaseAgent
from rltoolkit.cleanrl.rl_args import PERArguments
from rltoolkit.cleanrl.utils.qlearing_net import DuelingNet, QNet
from rltoolkit.data.utils.type_aliases import PrioritizedReplayBufferSamples
from rltoolkit.utils import LinearDecayScheduler, soft_target_update


class PERAgent(BaseAgent):
    """Prioritized Experience Replay (PER) agent implementing DQN, Double DQN,
    and Dueling DQN.

    Args:
        args (PERArguments): Configuration object with agent hyperparameters.
        env (gym.Env): The environment the agent will interact with.
        state_shape (Union[int, List[int]]): Shape of the environment's state space.
        action_shape (Union[int, List[int]]): Shape of the environment's action space.
        device (Optional[Union[str, torch.device]]): Device (CPU/GPU) to run the computations on.
    """

    def __init__(
        self,
        args: PERArguments,
        env: gym.Env,
        state_shape: Union[int, List[int]],
        action_shape: Union[int, List[int]],
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(args)

        # Check for valid argument values
        assert (isinstance(args.n_steps, int) and args.n_steps > 0
                ), 'N-step should be an integer greater than 0.'

        self.args = args
        self.env = env
        self.device = device or torch.device('cpu')
        # Default to CPU if no device is provided
        self.obs_dim = int(np.prod(state_shape))
        # Flatten state shape for network input
        self.action_dim = int(np.prod(action_shape))
        # Flatten action shape

        # Initialize learning variables
        self.learner_update_step = 0
        self.target_model_update_step = 0
        self.eps_greedy = args.eps_greedy_start
        self.learning_rate = args.learning_rate

        # Initialize networks
        if args.dueling_dqn:
            self.qnet = DuelingNet(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=self.args.hidden_dim,
            ).to(self.device)
        else:
            self.qnet = QNet(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=self.args.hidden_dim,
            ).to(self.device)

        self.target_qnet = copy.deepcopy(self.qnet)
        # Target network is a copy of the Q-network
        self.target_qnet.eval()
        # Set target network to evaluation mode

        # Initialize optimizer and learning rate scheduler
        self.optimizer = torch.optim.Adam(params=self.qnet.parameters(),
                                          lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer,
            start_factor=self.learning_rate,
            end_factor=self.args.min_learning_rate,
            total_iters=self.args.max_timesteps,
        )

        # Epsilon-greedy decay scheduler
        self.eps_greedy_scheduler = LinearDecayScheduler(
            self.args.eps_greedy_start,
            self.args.eps_greedy_end,
            max_steps=int(self.args.max_timesteps * 0.9),
        )

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get an action using epsilon-greedy policy.

        Args:
            obs (np.ndarray): The current observation from the environment.

        Returns:
            np.ndarray: The chosen action.
        """
        # With probability epsilon, sample a random action (exploration)
        if np.random.rand() <= self.eps_greedy:
            action = self.env.action_space.sample()
        else:
            # Otherwise, predict the best action (exploitation)
            action = self.predict(obs)

        # Decay epsilon over time
        self.eps_greedy = max(self.eps_greedy_scheduler.step(),
                              self.args.eps_greedy_end)
        return action

    def predict(self, obs: np.ndarray) -> int:
        """Predict the action based on the current Q-network.

        Args:
            obs (np.ndarray): The current observation.

        Returns:
            int: The action with the highest Q-value.
        """
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)  # Add batch dimension if needed

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.qnet(obs_tensor)
            action = q_values.argmax(dim=1).item()
        # Choose the action with highest Q-value
        return action

    def learn(
        self, batch: PrioritizedReplayBufferSamples
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Perform a learning step using the provided experience batch.

        Args:
            batch (PrioritizedReplayBufferSamples): A batch of experience data.

        Returns:
            Tuple[Dict[str, float], np.ndarray, np.ndarray]:
                - A dictionary with the current loss value.
                - Indices of the replay buffer samples.
                - New priorities for the replay buffer based on TD error.
        """
        obs = batch.obs.to(self.device)
        next_obs = batch.next_obs.to(self.device)
        action = batch.actions.to(self.device, dtype=torch.long)
        reward = batch.rewards.to(self.device)
        done = batch.dones.to(self.device)
        weights = torch.tensor(batch.weights,
                               dtype=torch.float32,
                               device=self.device).reshape(-1, 1)

        # Compute current Q-values
        current_q_values = self.qnet(obs).gather(1, action)

        # Compute next Q-values
        with torch.no_grad():
            if self.args.double_dqn:
                # Double DQN: Get next action using Q-network, then evaluate it using the target network
                next_actions = self.qnet(next_obs).argmax(dim=1, keepdim=True)
                next_q_values = self.target_qnet(next_obs).gather(
                    1, next_actions)
            else:
                # Regular DQN: Just use the maximum Q-value from the target network
                next_q_values = self.target_qnet(next_obs).max(dim=1,
                                                               keepdim=True)[0]

        # Calculate the target Q-values (Bellman equation)
        target_q_values = (
            reward + (1 - done) *
            (self.args.gamma**self.args.n_steps) * next_q_values)

        # Calculate TD error (used for prioritized replay)
        td_error = torch.abs(current_q_values - target_q_values)

        # Compute loss with importance sampling weights
        elementwise_loss = F.mse_loss(current_q_values,
                                      target_q_values,
                                      reduction='none')
        loss = (elementwise_loss * weights).mean()

        # Perform backpropagation
        self.optimizer.zero_grad()
        if self.args.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(),
                                           self.args.max_grad_norm)
        loss.backward()
        self.optimizer.step()

        # Soft update of the target network
        if self.learner_update_step % self.args.target_update_frequency == 0:
            soft_target_update(self.qnet, self.target_qnet,
                               self.args.soft_update_tau)

        # Update the number of learning steps
        self.learner_update_step += 1

        # Calculate new priorities for replay buffer
        new_priorities = td_error.detach().cpu().numpy() + self.args.prior_eps

        # Return the loss and new priorities
        return {'loss': loss.item()}, batch.indices, new_priorities
