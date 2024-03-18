from .base_policy import BasePolicy
from .modelfree.a2c import A2CPolicy
from .modelfree.c51 import C51Policy
from .modelfree.ddpg import DDPGPolicy
from .modelfree.discrete_sac import DiscreteSACPolicy
from .modelfree.dqn import DQNPolicy
from .modelfree.pg import PGPolicy
from .modelfree.ppo import PPOPolicy
from .modelfree.rainbow import RainbowPolicy
from .modelfree.sac import SACPolicy
from .modelfree.td3 import TD3Policy
from .modelfree.trpo import TRPOPolicy
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
    'A2CPolicy',
    'DDPGPolicy',
    'DiscreteSACPolicy',
    'PPOPolicy',
    'SACPolicy',
    'TD3Policy',
    'TRPOPolicy',
]
