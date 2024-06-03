import random
import sys

import numpy as np
import torch

sys.path.append('../../')
import gymnasium as gym
from rltoolkit.data import OffPolicyBuffer
from transformers import HfArgumentParser

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
    parser = HfArgumentParser((RLArguments, ))
    (args,
     _) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    args: RLArguments
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i) for i in range(args.num_envs)])
    test_envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i) for i in range(args.num_envs)])
    assert isinstance(
        train_envs.single_action_space,
        gym.spaces.Discrete), 'only discrete action space is supported'

    # Note: You can easily define other networks.
    env: gym.Env = gym.make(args.env_id, render_mode='human')
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    device = torch.device(
        'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    # should be N_FRAMES x H x W
    print('Observations shape:', state_shape)
    print('Actions shape:', action_shape)

    # agent
    agent = DQNAgent(args, train_envs, state_shape, action_shape, device)
    buffer = OffPolicyBuffer(
        args.buffer_size,
        train_envs.single_observation_space,
        train_envs.single_action_space,
        device=device,
        num_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    runner = Runner(
        args,
        train_envs=train_envs,
        test_envs=test_envs,
        agent=agent,
        buffer=buffer,
    )
    runner.run()
