import sys
import warnings

sys.path.append('../../')
from rltoolkit.envs import ShmemVectorEnv
from rltoolkit.envs.atari_wrappers import wrap_deepmind

try:
    import envpool
except ImportError:
    envpool = None


def make_atari_env(env_id, seed, train_num, test_num, **kwargs):
    """Wrapper function for Atari env.

    If EnvPool is installed, it will automatically switch to EnvPool's Atari env.

    :return: a tuple of (single env, training envs, test envs).
    """
    if envpool is not None:
        if kwargs.get('scale', 0):
            warnings.warn(
                'EnvPool does not include ScaledFloatFrame wrapper, '
                "please set `x = x / 255.0` inside CNN network's forward function.",
                stacklevel=2,
            )
        # parameters convertion
        train_envs = env = envpool.make_gymnasium(
            env_id.replace('NoFrameskip-v4', '-v5'),
            num_envs=train_num,
            seed=seed,
            episodic_life=True,
            reward_clip=True,
            stack_num=kwargs.get('frame_stack', 4),
        )
        test_envs = envpool.make_gymnasium(
            env_id.replace('NoFrameskip-v4', '-v5'),
            num_envs=test_num,
            seed=seed,
            episodic_life=False,
            reward_clip=False,
            stack_num=kwargs.get('frame_stack', 4),
        )
    else:
        warnings.warn(
            'Recommend using envpool (pip install envpool) '
            'to run Atari games more efficiently.',
            stacklevel=2,
        )
        env = wrap_deepmind(env_id, **kwargs)
        train_envs = ShmemVectorEnv([
            lambda: wrap_deepmind(
                env_id, episode_life=True, clip_rewards=True, **kwargs)
            for _ in range(train_num)
        ])
        test_envs = ShmemVectorEnv([
            lambda: wrap_deepmind(
                env_id, episode_life=False, clip_rewards=False, **kwargs)
            for _ in range(test_num)
        ])
        env.seed(seed)
        train_envs.seed(seed)
        test_envs.seed(seed)
    return env, train_envs, test_envs
