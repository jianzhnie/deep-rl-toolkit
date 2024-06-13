import argparse
import os
import random
import sys
from typing import Any, Dict

import numpy as np
import torch
import tyro
import yaml

sys.path.append(os.getcwd())
import gymnasium as gym
from rltoolkit.cleanrl.agent import C51Agent, DDPGAgent, DQNAgent
from rltoolkit.cleanrl.offpolicy_runner import OffPolicyRunner
from rltoolkit.cleanrl.rl_args import (C51Arguments, DDPGArguments,
                                       DQNArguments, PPOArguments,
                                       SACArguments)
from rltoolkit.data import SimpleReplayBuffer
from rltoolkit.utils.logger.logging import get_logger

logger = get_logger(__name__)


def make_env(env_id: str = 'CartPole-v0', seed: int = 42):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def update_dataclass_from_dict(instance: Any, config: Dict[str, Any]):
    """Update the attributes of a dataclass instance with values from a config
    dictionary.

    Args:
        instance (Any): The dataclass instance to update.
        config (Dict[str, Any]): The configuration dictionary.
    """
    for key, value in config.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def main() -> None:
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description='CleanRL Runner')
    parser.add_argument(
        '--algo_name',
        type=str,
        choices=[
            'dqn',
            'ddqn',
            'dueling_dqn',
            'dueling_ddqn',
            'rainbow',
            'c51',
            'pg',
            'ac',
            'a2c',
            'ppo',
            'ddpg',
            'sac',
            'td3',
        ],
        default='dqn',
        help="Name of the algorithm. Defaults to 'dqn'",
    )
    parser.add_argument(
        '--env_id',
        type=str,
        default='CartPole-v0',
        help="The environment name. Defaults to 'CartPole-v0'",
    )
    curr_path = os.getcwd()
    # Load YAML configuration
    config_file = os.path.join(curr_path, 'examples/cleanrl/config.yaml')
    config = load_yaml_config(config_file)
    # Parse arguments
    run_args = parser.parse_args()
    if run_args.algo_name in [
            'dqn',
            'ddqn',
            'noisy_dqn',
            'dueling_dqn',
            'dueling_ddqn',
            'rainbow',
    ]:
        # Update parser with DQN configuration
        algo_args: DQNArguments = tyro.cli(DQNArguments)
        Agent: DQNAgent = DQNAgent
    elif run_args.algo_name == 'c51':
        algo_args: C51Arguments = tyro.cli(C51Arguments)
        Agent: C51Agent = C51Agent
    elif run_args.algo_name == 'ddpg':
        algo_args: DDPGArguments = tyro.cli(DDPGArguments)
        Agent: DDPGAgent = DDPGAgent
    elif run_args.algo_name == 'ppo':
        algo_args: PPOArguments = tyro.cli(PPOArguments)
    elif run_args.algo_name == 'sac':
        algo_args: SACArguments = tyro.cli(SACArguments)

    # Extract Algo-specific settings
    if run_args.algo_name in config:
        algo_config = config.get(run_args.algo_name)
        env_config = algo_config.get(run_args.env_id)
    else:
        env_config = {}
        logger.warning(
            'No configuration found for {}, {} in {}. Using default settings.'.
            format(run_args.algo_name, run_args.env_id, config_file))
    # Update parser with YAML configuration
    args = update_dataclass_from_dict(algo_args, env_config)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    train_env: gym.Env = make_env(args.env_id)
    test_env: gym.Env = make_env(args.env_id)
    state_shape = train_env.observation_space.shape or train_env.observation_space.n
    action_shape = train_env.action_space.shape or train_env.action_space.n
    args.action_bound = (train_env.action_space.high[0] if isinstance(
        train_env.action_space, gym.spaces.Box) else None)
    device = torch.device(
        'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

    print('---------------------------------------')
    print('Environment:', args.env_id)
    print('Algorithm:', args.algo_name)
    print('State Shape:', state_shape)
    print('Action Shape:', action_shape)
    print('Action Bound:', args.action_bound)
    print('---------------------------------------')

    # agent
    agent = Agent(
        args=args,
        env=train_env,
        state_shape=state_shape,
        action_shape=action_shape,
        device=device,
    )
    buffer = SimpleReplayBuffer(
        buffer_size=args.buffer_size,
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        n_steps=args.n_steps,
        gamma=args.gamma,
        device=device,
    )
    runner = OffPolicyRunner(
        args,
        train_env=train_env,
        test_env=test_env,
        agent=agent,
        buffer=buffer,
    )
    runner.run()


if __name__ == '__main__':
    main()
