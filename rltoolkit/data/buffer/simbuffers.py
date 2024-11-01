import os
import random
import sys
from collections import deque
from typing import Any, Deque, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
from gymnasium import spaces

sys.path.append(os.getcwd())
from rltoolkit.cleanrl.rl_args import RLArguments
from rltoolkit.data.utils.segment_tree import MinSegmentTree, SumSegmentTree
from rltoolkit.data.utils.type_aliases import (PrioritizedReplayBufferSamples,
                                               ReplayBufferSamples,
                                               RolloutBufferSamples)
from stable_baselines3.common.preprocessing import (get_action_dim,
                                                    get_obs_shape)
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class SimpleReplayBuffer:
    """A simple FIFO experience replay buffer for off-policy RL or offline RL.

    Args:
        args (RLArguments): Hyperparameters for the buffer and algorithms.
        observation_space (spaces.Space): The observation space of the environment.
        action_space (spaces.Space): The action space of the environment.
        device (str, optional): Device to store the tensors ('cpu' or 'cuda'). Default is 'cpu'.
    """

    def __init__(
        self,
        args: RLArguments,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize the replay buffer."""
        self.args: RLArguments = args
        self.buffer_size = args.buffer_size
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)

        # Memory buffers for storing experiences
        self.observations = np.zeros((self.buffer_size, *self.obs_shape),
                                     dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size, *self.obs_shape),
                                          dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim),
                                dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)

        # For N-step Learning
        assert (isinstance(args.n_steps, int) and args.n_steps > 0
                ), 'n_steps should be an integer and greater than 0.'

        self.n_step_buffer: Deque = deque(maxlen=args.n_steps)

        self.curr_ptr = 0  # Pointer to the current position in the buffer
        self.curr_size = 0  # Tracks the current size of the buffer
        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """Add an experience sample to the replay buffer.

        Args:
            obs (np.ndarray): The current observation.
            next_obs (np.ndarray): The next observation.
            action (np.ndarray): The action taken.
            reward (np.ndarray): The reward received.
            done (np.ndarray): Whether the episode has ended.
        """
        # For N-step transition
        if self.args.n_steps > 1:
            transition = (obs, action, next_obs, reward, done)
            self.n_step_buffer.append(transition)

            # Make a n-step transition
            # Firstly, get curr obs and curr action
            obs, action = self.n_step_buffer[0][:2]
            # get the next_obs, reward, terminal in n_step_buffer deque
            next_obs, reward, done = self.get_n_step_info(
                self.n_step_buffer, self.args.gamma)

        self.observations[self.curr_ptr] = np.array(obs).copy()
        self.next_observations[self.curr_ptr] = np.array(next_obs).copy()
        self.actions[self.curr_ptr] = np.array(action).copy()
        self.rewards[self.curr_ptr] = np.array(reward).copy()
        self.dones[self.curr_ptr] = np.array(done).copy()

        # Update pointer and size
        self.curr_ptr = (self.curr_ptr + 1) % self.buffer_size
        self.curr_size = min(self.curr_size + 1, self.buffer_size)

    def get_n_step_info(self, n_step_buffer: Deque,
                        gamma: float) -> Tuple[np.ndarray, float, np.ndarray]:
        """Calculate the n-step reward, next observation, and done flag.

        Args:
            n_step_buffer (Deque): A deque storing the last n transitions.
            gamma (float): The discount factor.

        Returns:
            Tuple[np.ndarray, float, np.ndarray]: Next observation, n-step reward, and done flag.
        """
        next_obs, reward, done = n_step_buffer[-1][-3:]

        # info of the n-1 transition
        sub_n_step_buffer = list(n_step_buffer)[:-1]
        for transition in reversed(sub_n_step_buffer):
            next_o, r, d = transition[-3:]
            reward = r + gamma * reward * (1 - d)
            next_obs, done = (next_o, d) if d else (next_obs, done)

        return next_obs, reward, done

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """Convert a numpy array to a PyTorch tensor.

        Args:
            array (np.ndarray): The numpy array to convert.
            copy (bool, optional): Whether to copy the data. Defaults to True.

        Returns:
            torch.Tensor: The PyTorch tensor.
        """
        if copy:
            return torch.tensor(array, dtype=torch.float32).to(self.device)
        return torch.as_tensor(array, dtype=torch.float32).to(self.device)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences from the replay buffer.

        Args:
            batch_size (int): The number of samples to return.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing sampled observations, actions, rewards,
            next observations, dones, and indices.
        """
        batch_inds = np.random.randint(self.curr_size, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        """Retrieve the experiences corresponding to the given indices.

        Args:
            batch_inds (np.ndarray): Indices of the batch to sample.

        Returns:
            ReplayBufferSamples: A named tuple of the sampled experiences.
        """
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.next_observations[batch_inds],
            self.dones[batch_inds],
            self.rewards[batch_inds],
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def size(self) -> int:
        """Return the current size of the buffer."""
        return self.curr_size

    def __len__(self) -> int:
        return self.curr_size

    def save(self, pathname: str) -> None:
        """Save the replay buffer to a file.

        Args:
            pathname (str): File path for saving the buffer.
        """
        other = np.array([self.curr_size, self.curr_ptr], dtype=np.int32)
        np.savez(
            pathname,
            obs=self.observations,
            action=self.actions,
            reward=self.rewards,
            terminal=self.dones,
            next_obs=self.next_observations,
            other=other,
        )

    def load(self, pathname: str) -> None:
        """Load the replay buffer from a file.

        Args:
            pathname (str): File path to load the buffer from.
        """
        data = np.load(pathname)
        other = data['other']
        self.curr_size = min(int(other[0]), self.buffer_size)
        self.curr_ptr = min(int(other[1]), self.buffer_size - 1)

        self.observations[:self.curr_size] = data['obs'][:self.curr_size]
        self.actions[:self.curr_size] = data['action'][:self.curr_size]
        self.rewards[:self.curr_size] = data['reward'][:self.curr_size]
        self.dones[:self.curr_size] = data['terminal'][:self.curr_size]
        self.next_observations[:self.curr_size] = data['next_obs'][:self.
                                                                   curr_size]


class SimplePerReplayBuffer(SimpleReplayBuffer):
    """Replay buffer with prioritized sampling based on Temporal-Difference
    (TD) error.

    Args:
        args (RLArguments): Hyperparameters for the buffer and algorithms.
        observation_space (spaces.Space): The observation space of the environment.
        action_space (spaces.Space): The action space of the environment.
        device (Union[torch.device, str]): PyTorch device.
    """

    def __init__(
        self,
        args: RLArguments,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = 'cpu',
    ) -> None:
        """Initialize the prioritized replay buffer."""
        super().__init__(args, observation_space, action_space, device)

        # Parameters for prioritized replay
        self.alpha = args.alpha
        self.beta = args.beta
        self.max_prior = 1.0
        self.tree_ptr = 0

        # Capacity of the segment trees should be the next power of 2 greater than buffer size
        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2

        # Sum tree and min tree for efficient priority sampling
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """Add an experience to the prioritized replay buffer.

        Args:
            obs (np.ndarray): The current observation.
            next_obs (np.ndarray): The next observation.
            action (np.ndarray): The action taken.
            reward (np.ndarray): The reward received.
            done (np.ndarray): Whether the episode terminated after the action.
        """
        super().add(obs, next_obs, action, reward, done)
        # Update priorities in the segment trees
        self.sum_tree[self.tree_ptr] = self.max_prior**self.alpha
        self.min_tree[self.tree_ptr] = self.max_prior**self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.buffer_size

    def sample(self, batch_size: int) -> PrioritizedReplayBufferSamples:
        assert self.curr_size > batch_size
        batch_inds = self._sample_proportional(batch_size)
        weights = np.array([self._calculate_weight(i) for i in batch_inds])
        return self._get_samples(batch_inds, weights)

    def _get_samples(self, batch_inds: np.ndarray,
                     weights: torch.Tensor) -> PrioritizedReplayBufferSamples:
        """Retrieve the experiences corresponding to the given indices, along
        with their weights.

        Args:
            batch_inds (np.ndarray): The indices of the experiences to sample.
            weights (torch.Tensor): Importance sampling weights for the experiences.

        Returns:
            PrioritizedReplayBufferSamples: Sampled experiences and their weights.
        """
        # Get base samples using the parent method
        samples = super()._get_samples(batch_inds)
        return PrioritizedReplayBufferSamples(*samples, weights, batch_inds)

    def _sample_proportional(self, batch_size: int) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            left = segment * i
            right = segment * (i + 1)
            upperbound = random.uniform(left, right)
            idx = self.sum_tree.find_prefixsum_idx(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int):
        """Calculate the weight of the experience at idx."""
        # Get max weight
        min_prob = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (min_prob * len(self))**(-self.beta)

        # Calculate the sampling probabilities and importance weights
        sample_prob = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (sample_prob * len(self))**(-self.beta)
        weight = weight / max_weight
        return weight

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0, 'Priority must be positive.'
            assert 0 <= idx < self.curr_size, 'Index out of range.'

            # Clip priority and update the trees
            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha

            self.max_prior = max(self.max_prior, priority)


class SimpleRolloutBuffer:
    """Rollout buffer used in on-policy algorithms like A2C/PPO. This buffer
    stores `buffer_size` transitions collected using the current policy. The
    experience is discarded after the policy update.

    Args:
        args (RLArguments): Hyperparameters for the buffer and algorithms.
        observation_space (spaces.Space): Observation space.
        action_space (spaces.Space): Action space.
        device (Union[torch.device, str]): PyTorch device.
    """

    def __init__(
        self,
        args: RLArguments,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[str, torch.device] = 'cpu',
    ) -> None:
        """Initialize the rollout buffer."""
        self.args = args
        self.buffer_size = args.buffer_size
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.device = device
        self.reset()

    def reset(self) -> None:
        """Reset the rollout buffer by clearing all stored transitions."""
        self.observations = np.zeros((self.buffer_size, *self.obs_shape),
                                     dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim),
                                dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, 1), dtype=np.float32)

        self.curr_ptr = 0
        self.curr_size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: Union[float, np.ndarray],
        log_prob: Union[float, np.ndarray],
    ) -> None:
        """Add a new transition to the buffer.

        Args:
            obs (np.ndarray): Observation.
            action (np.ndarray): Action.
            reward (float): Reward.
            done (bool): Whether the episode is done.
            value (torch.Tensor): Estimated value of the current state.
            log_prob (torch.Tensor): Log probability of the action.
        """
        self.observations[self.curr_ptr] = np.array(obs)
        self.actions[self.curr_ptr] = np.array(action)
        self.rewards[self.curr_ptr] = np.array(reward)
        self.dones[self.curr_ptr] = np.array(done)
        self.values[self.curr_ptr] = np.array(value)
        self.log_probs[self.curr_ptr] = np.array(log_prob)

        self.curr_ptr = (self.curr_ptr + 1) % self.buffer_size
        self.curr_size = min(self.curr_size + 1, self.buffer_size)

    def compute_returns_and_advantage(self, last_values: Union[float,
                                                               np.ndarray],
                                      dones: np.ndarray) -> None:
        """Compute the lambda-return (TD(lambda) estimate) and GAE(lambda)
        advantage.

        Args:
            last_values (torch.Tensor): State value estimation for the last step (one for each env).
            dones (np.ndarray): Boolean array indicating if the last step was terminal.
        """
        last_gae_lam = 0

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            delta = (self.rewards[step] +
                     self.args.gamma * next_values * next_non_terminal -
                     self.values[step])

            last_gae_lam = (delta + self.args.gamma * self.args.gae_lambda *
                            next_non_terminal * last_gae_lam)
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

    def sample(
        self,
        batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        """Sample transitions from the buffer.

        Args:
            batch_size (Optional[int]): Batch size. If None, the entire buffer is used.

        Yields:
            RolloutBufferSamples: Sampled batch of transitions.
        """
        assert self.curr_size == self.buffer_size, 'Buffer not full'
        indices = np.random.permutation(self.buffer_size)

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += self.args.batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> RolloutBufferSamples:
        """Get a batch of samples based on provided indices.

        Args:
            batch_inds (np.ndarray): Indices for sampling.

        Returns:
            RolloutBufferSamples: Batch of samples as tensors.
        """
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds],
            self.log_probs[batch_inds],
            self.advantages[batch_inds],
            self.returns[batch_inds],
        )
        samples = RolloutBufferSamples(*tuple(map(self.to_torch, data)))
        return samples

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """Convert a numpy array to a PyTorch tensor.

        Args:
            array (np.ndarray): The numpy array to convert.
            copy (bool, optional): Whether to copy the data. Defaults to True.

        Returns:
            torch.Tensor: The PyTorch tensor.
        """
        if copy:
            return torch.tensor(array, dtype=torch.float32).to(self.device)
        return torch.as_tensor(array, dtype=torch.float32).to(self.device)

    def size(self) -> int:
        """Return the current size of the buffer."""
        return self.curr_size

    def data_generator(self,
                       num_mini_batch: int = None,
                       mini_batch_size: int = None
                       ) -> Generator[RolloutBufferSamples, Any, None]:
        if mini_batch_size is None:
            assert self.buffer_size >= num_mini_batch, (
                'PPO requires the number of buffer_size ({}) '
                'to be greater than or equal to the number of PPO mini batches ({}).'
                ''.format(
                    self.buffer_size,
                    num_mini_batch,
                ))
            mini_batch_size = self.buffer_size // num_mini_batch
        indices = np.arange(self.buffer_size)
        sampler = BatchSampler(SubsetRandomSampler(indices),
                               mini_batch_size,
                               drop_last=True)
        for batch_inds in sampler:
            data = (
                self.observations[batch_inds],
                self.actions[batch_inds],
                self.values[batch_inds],
                self.log_probs[batch_inds],
                self.advantages[batch_inds],
                self.returns[batch_inds],
            )
            samples = RolloutBufferSamples(*tuple(map(self.to_torch, data)))
            yield samples
