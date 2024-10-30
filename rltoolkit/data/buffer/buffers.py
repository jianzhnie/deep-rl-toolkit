import os
import sys
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
from gymnasium import spaces

sys.path.append(os.getcwd())
from rltoolkit.data.utils import SegmentTree, to_numpy
from rltoolkit.data.utils.type_aliases import (PrioritizedReplayBufferSamples,
                                               ReplayBufferSamples,
                                               RolloutBufferSamples)
from stable_baselines3.common.preprocessing import (get_action_dim,
                                                    get_obs_shape)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class BaseBuffer(ABC):
    """Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param num_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: Tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = 'auto',
        num_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(
            observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.curr_ptr = 0
        self.curr_size = 0
        self.device = get_device(device)
        self.num_envs = num_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """Swap and then flatten axes 0 (buffer_size) and 1 (num_envs) to
        convert shape from [n_steps, num_envs, ...] (when ... is the shape of
        the features)

        to [n_steps * num_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        return self.curr_size

    def __len__(self) -> int:
        """
        :return: The total capacity of the buffer
        """
        return self.size()

    def add(self, *args, **kwargs) -> None:
        """Add elements to the buffer."""
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """Add a new batch of transitions to the buffer."""
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """Reset the buffer."""
        self.curr_ptr = 0
        self.curr_size = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        batch_inds = np.random.randint(0, self.curr_size, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray,
                          env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class OffPolicyBuffer(BaseBuffer):
    """Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param num_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = 'auto',
        num_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            num_envs=num_envs,
        )

        # Adjust buffer size
        self.buffer_size = max(buffer_size // num_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                'ReplayBuffer does not support optimize_memory_usage = True '
                'and handle_timeout_termination = True simultaneously.')
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros(
            (self.buffer_size, self.num_envs, *self.obs_shape),
            dtype=observation_space.dtype,
        )

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros(
                (self.buffer_size, self.num_envs, *self.obs_shape),
                dtype=observation_space.dtype,
            )

        self.actions = np.zeros(
            (self.buffer_size, self.num_envs, self.action_dim),
            dtype=self._maybe_cast_dtype(action_space.dtype),
        )

        self.rewards = np.zeros((self.buffer_size, self.num_envs),
                                dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.num_envs),
                              dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.num_envs),
                                 dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (self.observations.nbytes +
                                         self.actions.nbytes +
                                         self.rewards.nbytes +
                                         self.dones.nbytes)

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    'This system does not have apparently enough memory to store the complete '
                    f'replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB'
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.num_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.num_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.num_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.curr_ptr] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.curr_ptr + 1) %
                              self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.curr_ptr] = np.array(next_obs)

        self.actions[self.curr_ptr] = np.array(action)
        self.rewards[self.curr_ptr] = np.array(reward)
        self.dones[self.curr_ptr] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.curr_ptr] = np.array(
                [info.get('TimeLimit.truncated', False) for info in infos])

        self.curr_ptr = (self.curr_ptr + 1) % self.buffer_size
        self.curr_size = min(self.curr_size + 1, self.buffer_size)

    def sample(self,
               batch_size: int,
               env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """Sample elements from the replay buffer. Custom sampling when using
        memory efficient variant, as we should not sample the element with
        index `self.curr_ptr` See https://github.com/DLR-RM/stable-
        baselines3/pull/28#issuecomment-637559274.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.curr_ptr` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.size() == self.buffer_size:
            batch_inds = np.random.randint(1,
                                           self.buffer_size,
                                           size=batch_size)
        else:
            batch_inds = np.random.randint(0, self.curr_size, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(
            self,
            batch_inds: np.ndarray,
            env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0,
                                        high=self.num_envs,
                                        size=(len(batch_inds), ))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size,
                                  env_indices, :],
                env,
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :],
                                env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] *
             (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """Cast `np.float64` action datatype to `np.float32`, keep the others
        dtype unchanged. See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype


class RolloutBuffer(BaseBuffer):
    """Rollout buffer used in on-policy algorithms like A2C/PPO. It corresponds
    to ``buffer_size`` transitions collected using the current policy. This
    experience will be discarded after the policy update. In order to use PPO
    objective, we also store the current value of each state and the log
    probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param num_envs: Number of parallel environments
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = 'auto',
        gae_lambda: float = 1,
        gamma: float = 0.99,
        num_envs: int = 1,
    ) -> None:
        super().__init__(buffer_size,
                         observation_space,
                         action_space,
                         device,
                         num_envs=num_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros(
            (self.buffer_size, self.num_envs, *self.obs_shape),
            dtype=np.float32)
        self.actions = np.zeros(
            (self.buffer_size, self.num_envs, self.action_dim),
            dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.num_envs),
                                dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.num_envs),
                                dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.num_envs),
                                       dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.num_envs),
                               dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.num_envs),
                                  dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.num_envs),
                                   dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: torch.Tensor,
                                      dones: np.ndarray) -> None:
        """Post-processing step: compute the lambda-return (TD(lambda)
        estimate) and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (self.rewards[step] +
                     self.gamma * next_values * next_non_terminal -
                     self.values[step])
            last_gae_lam = (delta + self.gamma * self.gae_lambda *
                            next_non_terminal * last_gae_lam)
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.num_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.num_envs, self.action_dim))

        self.observations[self.curr_ptr] = np.array(obs)
        self.actions[self.curr_ptr] = np.array(action)
        self.rewards[self.curr_ptr] = np.array(reward)
        self.episode_starts[self.curr_ptr] = np.array(episode_start)
        self.values[self.curr_ptr] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.curr_ptr] = log_prob.clone().cpu().numpy()

        self.curr_ptr = (self.curr_ptr + 1) % self.buffer_size
        self.curr_size = min(self.curr_size + 1, self.buffer_size)

    def get(
        self,
        batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert (
            len(self) == self.buffer_size
        ), f'Buffer not full, get {len(self)}, required {self.buffer_size}'
        indices = np.random.permutation(self.buffer_size * self.num_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                'observations',
                'actions',
                'values',
                'log_probs',
                'advantages',
                'returns',
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(
                    self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.num_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.num_envs:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


class PrioritizedReplayBuffer(BaseBuffer):
    """Replay buffer used in off-policy algorithms like SAC/TD3. This time with
    priorization!

    Mainly copy/paste from
        https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/buffers.py

    :param buffer_size: Max number of element in the buffer
    :param alpha: How much priorization is used (0: disabled, 1: full priorization)
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param num_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = 'auto',
        alpha: float = 1.0,
        beta: float = 1.0,
        weight_norm: bool = True,
        num_envs: int = 1,
    ) -> None:
        super().__init__(buffer_size, observation_space, action_space, device,
                         num_envs)
        assert alpha >= 0.0 and beta >= 0.0
        self._alpha, self._beta = alpha, beta

        per_capacity = 1
        while per_capacity < buffer_size:
            per_capacity *= 2

        self.weight = SegmentTree(per_capacity)
        self._eps = np.finfo(np.float32).eps.item()
        self._weight_norm = weight_norm
        self._max_prio = self._min_prio = 1.0

        self.observations = np.zeros(
            (self.buffer_size, self.num_envs) + self.obs_shape,
            dtype=observation_space.dtype,
        )
        self.next_observations = np.zeros(
            (self.buffer_size, self.num_envs) + self.obs_shape,
            dtype=observation_space.dtype,
        )
        self.actions = np.zeros(
            (self.buffer_size, self.num_envs, self.action_dim),
            dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.num_envs),
                                dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.num_envs),
                              dtype=np.float32)

    def init_weight(self, index: Union[int, np.ndarray]) -> None:
        self.weight[index] = self._max_prio**self._alpha

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self.curr_ptr] = np.array(obs).copy()
        self.next_observations[self.curr_ptr] = np.array(next_obs).copy()

        self.actions[self.curr_ptr] = np.array(action).copy()
        self.rewards[self.curr_ptr] = np.array(reward).copy()
        self.dones[self.curr_ptr] = np.array(done).copy()
        self.init_weight(self.curr_ptr)

        self.curr_ptr = (self.curr_ptr + 1) % self.buffer_size
        self.curr_size = min(self.curr_size + 1, self.buffer_size)

    def _get_samples(self,
                     batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None
                     ) -> PrioritizedReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0,
                                        high=self.num_envs,
                                        size=(len(batch_inds), ))

        next_obs = self._normalize_obs(
            self.next_observations[batch_inds, env_indices, :], env)
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :],
                                env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            self.dones[batch_inds, env_indices],
            self._normalize_reward(self.rewards[batch_inds, env_indices], env),
        )

        return data

    def sample(
            self,
            batch_size: int,
            env: Optional[VecNormalize] = None
    ) -> PrioritizedReplayBufferSamples:
        """Sample elements from the replay buffer using priorization.

        :param batch_size: Number of element to sample
        :param beta: To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        # Sample indices
        if batch_size > 0 and self.size() > 0:
            scalar = np.random.rand(batch_size) * self.weight.reduce()
            batch_inds = self.weight.get_prefix_sum_idx(scalar)

        weights = self.get_weight(batch_inds)
        weights = weights / np.max(weights) if self._weight_norm else weights
        th_data = self._get_samples(batch_inds, env)
        return PrioritizedReplayBufferSamples(*tuple(
            map(self.to_torch, th_data)),
                                              weights=weights,
                                              indices=batch_inds)

    def get_weight(self, index: Union[int,
                                      np.ndarray]) -> Union[float, np.ndarray]:
        """Get the importance sampling weight.

        The "weight" in the returned Batch is the weight on loss function to
        debias the sampling process (some transition tuples are sampled more
        often so their losses are weighted less).
        """
        # important sampling weight calculation
        # original formula: ((p_j/p_sum*N)**(-beta))/((p_min/p_sum*N)**(-beta))
        # simplified formula: (p_j/p_min)**(-beta)
        return (self.weight[index] / self._min_prio)**(-self._beta)

    def update_weights(self, batch_inds: np.ndarray, weights: np.ndarray):
        """Update weights of sampled transitions.

        sets weight of transition at index idxes[i] in buffer
        to weights[i].

        :param batch_inds: ([int]) np.ndarray of idxes of sampled transitions
        :param weights: ([float]) np.ndarray of updated weights corresponding to transitions at the sampled idxes
            denoted by variable `batch_inds`.
        """
        assert len(batch_inds) == len(weights)
        assert np.min(weights) > 0
        assert np.min(batch_inds) >= 0
        assert np.max(batch_inds) < self.size()
        weight = np.abs(to_numpy(weights)) + self._eps
        self.weight[batch_inds] = weight**self._alpha
        self._max_prio = max(self._max_prio, np.max(weights))
        self._min_prio = min(self._min_prio, weight.min())
