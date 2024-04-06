"""Env package."""

from rltoolkit.envs.gym_wrappers import (ContinuousToDiscrete,
                                         MultiDiscreteToDiscrete,
                                         TruncatedAsTerminated)
from rltoolkit.envs.vec_env import (BaseVectorEnv, DummyVectorEnv,
                                    RayVectorEnv, ShmemVectorEnv,
                                    SubprocVectorEnv)

try:
    from rltoolkit.envs.pettingzoo_env import PettingZooEnv
except ImportError:
    pass

__all__ = [
    'BaseVectorEnv',
    'DummyVectorEnv',
    'SubprocVectorEnv',
    'ShmemVectorEnv',
    'RayVectorEnv',
    'VectorEnvWrapper',
    'VectorEnvNormObs',
    'PettingZooEnv',
    'ContinuousToDiscrete',
    'MultiDiscreteToDiscrete',
    'TruncatedAsTerminated',
]
