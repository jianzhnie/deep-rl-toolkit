import random
import sys

import numpy as np
import torch
import tyro

sys.path.append('../../')
import gymnasium as gym
from rltoolkit.data import SimpleReplayBuffer

from cleanrl.dqn import DQNAgent
from cleanrl.rl_args import RLArguments
from cleanrl.runner import Runner


def make_env(env_id, seed):

    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


if __name__ == '__main__':
    args: RLArguments = tyro.cli(RLArguments)
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Note: You can easily define other networks.
    train_env: gym.Env = gym.make(args.env_id)
    test_env: gym.Env = gym.make(args.env_id)
    state_shape = train_env.observation_space.shape or train_env.observation_space.n
    action_shape = train_env.action_space.shape or train_env.action_space.n
    device = torch.device(
        'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    # should be N_FRAMES x H x W
    print('Observations shape:', state_shape)
    print('Actions shape:', action_shape)

    # agent
    agent = DQNAgent(args, train_env, state_shape, action_shape, device)
    buffer = SimpleReplayBuffer(
        buffer_size=args.buffer_size,
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        device=device,
    )
    runner = Runner(
        args,
        train_env=train_env,
        test_env=test_env,
        agent=agent,
        buffer=buffer,
    )
    runner.run()
