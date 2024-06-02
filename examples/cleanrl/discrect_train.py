import random
import sys

import numpy as np
import torch

sys.path.append('../')
import gym
from rltoolkit.data import OffPolicyBuffer
from rltoolkit.envs import SubprocVectorEnv
from transformers import HfArgumentParser

from cleanrl.dqn import DQNAgent
from cleanrl.rl_args import RLArguments
from cleanrl.runner import Runner

if __name__ == '__main__':
    parser = HfArgumentParser(RLArguments)
    args: RLArguments = parser.parse_args_into_dataclasses()
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    train_envs = SubprocVectorEnv(
        [lambda: gym.make(args.env_id) for _ in range(args.num_envs)])
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(args.env_id) for _ in range(args.num_envs)])
    env = gym.make(args.env_id)
    device = torch.device(
        'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print('Observations shape:', args.state_shape)
    print('Actions shape:', args.action_shape)

    # agent
    agent = DQNAgent(args, train_envs, device)
    buffer = OffPolicyBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device=device,
        num_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    runner = Runner(args)
    runner.run()
