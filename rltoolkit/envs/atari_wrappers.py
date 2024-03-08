# Borrow a lot from openai baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

# Borrow a lot from rllib:
# https://github.com/ray-project/ray/blob/master/rllib/env/wrappers/atari_wrappers.py

from collections import deque
from typing import Optional, SupportsFloat, Union

import cv2
import gymnasium as gym
import numpy as np


def is_atari(env: Union[gym.Env, str]) -> bool:
    """Returns, whether a given env object or env descriptor (str) is an Atari
    env.

    Args:
        env: The gym.Env object or a string descriptor of the env (e.g. "ALE/Pong-v5").

    Returns:
        Whether `env` is an Atari environment.
    """
    # If a gym.Env, check proper spaces as well as occurrence of the "Atari<ALE" string
    # in the class name.
    if not isinstance(env, str):
        if (hasattr(env.observation_space, 'shape')
                and env.observation_space.shape is not None
                and len(env.observation_space.shape) <= 2):
            return False
        return 'AtariEnv<ALE' in str(env)
    # If string, check for "ALE/" prefix.
    else:
        return env.startswith('ALE/')


def _parse_reset_result(reset_result):
    contains_info = (isinstance(reset_result, tuple) and len(reset_result) == 2
                     and isinstance(reset_result[1], dict))
    if contains_info:
        return reset_result[0], reset_result[1], contains_info
    return reset_result, {}, contains_info


class ClipRewardEnv(gym.RewardWrapper):
    """clips the reward to {+1, 0, -1} by its sign.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.reward_range = (-1, 1)

    def reward(self, reward: SupportsFloat):
        """Bin reward to {+1, 0, -1} by its sign.

        Note: np.sign(0) == 0.
        """
        return np.sign(float(reward))


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over. It
    helps the value estimation.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True
        self._return_info = False

    def step(self, action: Union[int, np.array]):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            new_step_api = False
        else:
            obs, reward, term, trunc, info = step_result
            done = term or trunc
            new_step_api = True

        self.was_real_done = done
        # check current lives, make loss of life terminal, then update lives to
        # handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few
            # frames, so its important to keep lives > 0, so that we only reset
            # once the environment is actually done.
            done = True
            term = True
        self.lives = lives
        if new_step_api:
            return obs, reward, term, trunc, info
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Calls the Gym environment reset, only when lives are exhausted.

        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, info, self._return_info = _parse_reset_result(
                self.env.reset(**kwargs))
        else:
            # no-op step to advance from terminal/lost life state
            step_result = self.env.step(0)
            obs, info = step_result[0], step_result[-1]
        self.lives = self.env.unwrapped.ale.lives()
        if self._return_info:
            return obs, info
        else:
            return obs


class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing.
    Related discussion: https://github.com/openai/baselines/issues/240.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        _, _, return_info = _parse_reset_result(self.env.reset(**kwargs))
        step_result = self.env.step(1)
        if len(step_result) == 4:
            obs, _, done, info = step_result
            new_step_api = False
        else:
            obs, _, term, trunc, info = step_result
            done = term or trunc
            new_step_api = True
        if done:
            self.env.reset(**kwargs)

        step_result = self.env.step(2)
        if new_step_api:
            obs, _, done, info = step_result
        else:
            obs, _, term, trunc, info = step_result
            done = term or trunc
        if done:
            self.env.reset(**kwargs)
        return (obs, info) if return_info else obs


class FrameStack(gym.Wrapper):
    """Stack n_frames last frames.

    :param gym.Env env: the environment to wrap.
    :param int n_frames: the number of frames to stack.
    """

    def __init__(self, env, n_frames):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shape = (n_frames, ) + env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=shape,
            dtype=env.observation_space.dtype,
        )

    def reset(self, *, seed=None, options=None):
        obs, info, return_info = _parse_reset_result(
            self.env.reset(seed=seed, options=options))
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return (self._get_ob(), info) if return_info else self._get_ob()

    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            new_step_api = False
        else:
            obs, reward, term, trunc, info = step_result
            new_step_api = True
        self.frames.append(obs)
        if new_step_api:
            return self._get_ob(), reward, term, trunc, info
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        # the original wrapper use `LazyFrames` but since we use np buffer,
        # it has no effect
        assert len(self.frames) == self.n_frames
        return np.stack(self.frames, axis=0)


class FrameStackTrajectoryView(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        """No stacking.

        Trajectory View API takes care of this.
        """
        super().__init__(env)
        obs_shape = env.observation_space.shape
        assert obs_shape[2] == 1
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0], obs_shape[1]),
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation: np.array) -> np.array:
        return np.squeeze(observation, axis=-1)


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame (frameskipping) using most recent raw
    observations (for max pooling across time steps)

    :param gym.Env env: the environment to wrap.
    :param int skip: number of `skip`-th frame.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Step the environment with the given action.

        Repeat action, sum reward, and max over last observations.
        """
        obs_list, total_reward = [], 0.0
        new_step_api = False
        for _ in range(self._skip):
            step_result = self.env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, term, trunc, info = step_result
                done = term or trunc
                new_step_api = True
            obs_list.append(obs)
            total_reward += reward
            if done:
                break
        # Note that the observation on the terminated|truncated=True frame
        # doesn't matter
        max_frame = np.max(obs_list[-2:], axis=0)
        if new_step_api:
            return max_frame, total_reward, term, trunc, info

        return max_frame, total_reward, done, info


class MonitorEnv(gym.Wrapper):

    def __init__(self, env: gym.Env) -> None:
        """Record episodes stats prior to EpisodicLifeEnv, etc."""
        super().__init__(env)
        self._current_reward = None
        self._num_steps = None
        self._total_steps = None
        self._episode_rewards = []
        self._episode_lengths = []
        self._num_episodes = 0
        self._num_returned = 0

    def reset(self, **kwargs):
        obs, info, return_info = self.env.reset(**kwargs)

        if self._total_steps is None:
            self._total_steps = sum(self._episode_lengths)

        if self._current_reward is not None:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._num_episodes += 1

        self._current_reward = 0
        self._num_steps = 0

        return (obs, info) if return_info else obs

    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            new_step_api = False
        else:
            obs, reward, term, trunc, info = step_result
            new_step_api = True

        self._current_reward += reward
        self._num_steps += 1
        self._total_steps += 1
        if new_step_api:
            return obs, reward, term, trunc, info
        return obs, reward, done, info

    def get_episode_rewards(self):
        return self._episode_rewards

    def get_episode_lengths(self):
        return self._episode_lengths

    def get_total_steps(self):
        return self._total_steps

    def next_episode_results(self):
        for i in range(self._num_returned, len(self._episode_rewards)):
            yield (self._episode_rewards[i], self._episode_lengths[i])
        self._num_returned = len(self._episode_rewards)


class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset. No-op
    is assumed to be action 0.

    :param gym.Env env: the environment to wrap.
    :param int noop_max: the maximum value of no-ops to run.
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        reset_info = self.env.reset(**kwargs)
        _, info, return_info = _parse_reset_result(reset_info)

        if hasattr(self.unwrapped.np_random, 'integers'):
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0, f'noops={noops}'
        for _ in range(noops):
            step_result = self.env.step(self.noop_action)
            if len(step_result) == 4:
                obs, _, done, info = step_result
            else:
                obs, _, term, trunc, info = step_result
                done = term or trunc
            if done:
                obs, info, _ = _parse_reset_result(self.env.reset())
        if return_info:
            return obs, info
        return obs

    def step(self, action):
        return self.env.step(action)


class NormalizedImageEnv(gym.ObservationWrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(
            -1.0,
            1.0,
            shape=self.observation_space.shape,
            dtype=np.float32,
        )

    # Divide by scale and center around 0.0, such that observations are in the range
    # of -1.0 and 1.0.
    def observation(self, observation):
        return (observation.astype(np.float32) / 128.0) - 1.0


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize observations to 0~1.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        low = np.min(env.observation_space.low)
        high = np.max(env.observation_space.high)
        self.bias = low
        self.scale = high - low
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32)

    def observation(self, observation):
        return (observation - self.bias) / self.scale


class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 84x84 as done in the Nature paper and later work.

    :param env: Environment to wrap
    :param width: New frame width
    :param height: New frame height
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        assert isinstance(
            env.observation_space,
            gym.spaces.Box), f'Expected Box space, got {env.observation_space}'

        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=(self.height, self.width, 1),
            dtype=env.observation_space.dtype,
        )

    def observation(self, frame: np.ndarray):
        """returns the current observation from a frame.

        :param frame: environment frame
        :return: the observation
        """
        assert (
            cv2 is not None
        ), 'OpenCV is not installed, you can do `pip install opencv-python`'
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class StickyActionEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """Sticky action.

    Paper: https://arxiv.org/abs/1709.06009
    Official implementation: https://github.com/mgbellemare/Arcade-Learning-Environment

    :param env: Environment to wrap
    :param action_repeat_probability: Probability of repeating the last action
    """

    def __init__(self, env: gym.Env, action_repeat_probability: float) -> None:
        super().__init__(env)
        self.action_repeat_probability = action_repeat_probability
        assert env.unwrapped.get_action_meanings(
        )[0] == 'NOOP'  # type: ignore[attr-defined]

    def reset(self, **kwargs):
        self._sticky_action = 0  # NOOP
        return self.env.reset(**kwargs)

    def step(self, action: int):
        if self.np_random.random() >= self.action_repeat_probability:
            self._sticky_action = action
        return self.env.step(self._sticky_action)


class AtariWrapper(gym.Wrapper):
    """Atari 2600 preprocessings.

    Specifically:

    * Noop reset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation
    * Clip reward to {-1, 0, 1}
    * Sticky actions: disabled by default

    See https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
    for a visual explanation.

    .. warning::
        Use this wrapper only with Atari v4 without frame skip: ``env_id = "*NoFrameskip-v4"``.

    :param env: Environment to wrap
    :param noop_max: Max number of no-ops
    :param frame_skip: Frequency at which the agent experiences the game.
        This correspond to repeating the action ``frame_skip`` times.
    :param screen_size: Resize Atari frame
    :param terminal_on_life_loss: If True, then step() returns done=True whenever a life is lost.
    :param clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
    :param action_repeat_probability: Probability of repeating the last action
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        frame_stack: Optional[int] = None,
        screen_size: int = 84,
        episode_life: bool = True,
        warp_frame: bool = True,
        normalize_obs: bool = True,
        clip_rewards: bool = True,
    ) -> None:
        # Time limit.
        env = gym.wrappers.TimeLimit(env, max_episode_steps=108000)
        # Grayscale + resize
        if warp_frame:
            env = WarpFrame(env, width=screen_size, height=screen_size)
        # Normalize the image.
        if normalize_obs:
            env = NormalizedImageEnv(env)
        # Frameskip: Take max over these n frames.
        if frame_skip > 1:
            assert env.spec is not None
            env = MaxAndSkipEnv(env, skip=frame_skip)
        # Send n noop actions into env after reset to increase variance in the
        # "start states" of the trajectories. These dummy steps are NOT included in the
        # sampled data used for learning.
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)
        # Each life is one episode.
        if episode_life:
            env = EpisodicLifeEnv(env)
        # Some envs only start playing after pressing fire. Unblock those.
        if 'FIRE' in env.unwrapped.get_action_meanings(
        ):  # type: ignore[attr-defined]
            env = FireResetEnv(env)
        # Framestack.
        if frame_stack:
            env = FrameStack(env, n_frames=frame_stack)
        if clip_rewards:
            env = ClipRewardEnv(env)
        super().__init__(env)


def wrap_deepmind(
    env_id: str,
    noop_max: int = 30,
    screen_size: int = 84,
    frame_skip: int = 4,
    frame_stack: Optional[int] = None,
    episode_life: bool = True,
    warp_frame: bool = True,
    normalize_obs: bool = True,
    clip_rewards: bool = True,
):
    """Configure environment for DeepMind-style Atari. The observation is
    channel-first: (c, h, w) instead of (h, w, c).

    :param str env_id: the atari environment id.
    :param bool episode_life: wrap the episode life wrapper.
    :param bool clip_rewards: wrap the reward clipping wrapper.
    :param int frame_stack: wrap the frame stacking wrapper.
    :param bool scale: wrap the scaling observation wrapper.
    :param bool warp_frame: wrap the grayscale + resize observation wrapper.
    :return: the wrapped atari environment.
    """
    assert 'NoFrameskip' in env_id
    env = gym.make(env_id)
    env = MonitorEnv(env)
    # Time limit.
    env = gym.wrappers.TimeLimit(env, max_episode_steps=108000)
    # Grayscale + resize
    if warp_frame:
        env = WarpFrame(env, width=screen_size, height=screen_size)
    # Normalize the image.
    if normalize_obs:
        env = NormalizedImageEnv(env)
    # Frameskip: Take max over these n frames.
    if frame_skip > 1:
        assert env.spec is not None
        env = MaxAndSkipEnv(env, skip=frame_skip)
    # Send n noop actions into env after reset to increase variance in the
    # "start states" of the trajectories. These dummy steps are NOT included in the
    # sampled data used for learning.
    if noop_max > 0:
        env = NoopResetEnv(env, noop_max=noop_max)
    # Each life is one episode.
    if episode_life:
        env = EpisodicLifeEnv(env)
    # Some envs only start playing after pressing fire. Unblock those.
    if 'FIRE' in env.unwrapped.get_action_meanings(
    ):  # type: ignore[attr-defined]
        env = FireResetEnv(env)
    # Framestack.
    if frame_stack:
        env = FrameStack(env, n_frames=frame_stack)
    if clip_rewards:
        env = ClipRewardEnv(env)
    return env


def wrap_atari_for_new_api_stack(
    env: gym.Env,
    sceeen_size: int = 64,
    frame_skip: int = 4,
    frame_stack: Optional[int] = None,
) -> gym.Env:
    """Wraps `env` for new-API-stack-friendly RLlib Atari experiments.

    Note that we assume reward clipping is done outside the wrapper.

    Args:
        env: The env object to wrap.
        dim: Dimension to resize observations to (dim x dim).
        frameskip: Whether to skip n frames and max over them.
        framestack: Whether to stack the last n (grayscaled) frames. Note that this
            step happens after(!) a possible frameskip step, meaning that if
            frameskip=4 and framestack=2, we would perform the following over this
            trajectory:
            actual env timesteps: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 -> ...
            frameskip:            ( max ) ( max ) ( max   ) ( max     )
            framestack:           ( stack       ) (stack              )

    Returns:
        The wrapped gym.Env.
    """
    # Time limit.
    env = gym.wrappers.TimeLimit(env, max_episode_steps=108000)
    # Grayscale + resize
    env = WarpFrame(env, width=sceeen_size, height=sceeen_size)
    # Normalize the image.
    env = NormalizedImageEnv(env)
    # Frameskip: Take max over these n frames.
    if frame_skip > 1:
        assert env.spec is not None
        env = MaxAndSkipEnv(env, skip=frame_skip)
    # Send n noop actions into env after reset to increase variance in the
    # "start states" of the trajectories. These dummy steps are NOT included in the
    # sampled data used for learning.
    env = NoopResetEnv(env, noop_max=30)
    # Each life is one episode.
    env = EpisodicLifeEnv(env)
    # Some envs only start playing after pressing fire. Unblock those.
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    # Framestack.
    if frame_stack:
        env = FrameStack(env, n_frames=frame_stack)
    return env
