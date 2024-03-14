from .base_policy import BasePolicy
from .modelfree.c51 import C51Policy
from .modelfree.dqn import DQNPolicy
from .modelfree.pg import PGPolicy
from .modelfree.rainbow import RainbowPolicy
from .multiagent.mapolicy import MultiAgentPolicyManager
from .random_policy import RandomPolicy

__all__ = [
    'BasePolicy',
    'DQNPolicy',
    'RandomPolicy',
    'MultiAgentPolicyManager',
    'PGPolicy',  # Policy Gradient]
    'C51Policy',  # Categorical 51
    'RainbowPolicy',  # Rainbow
]
