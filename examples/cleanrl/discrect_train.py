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

    # You can also try SubprocVectorEnv, which will use parallelization
    train_envs = SubprocVectorEnv(
        [lambda: gym.make(args.env_id) for _ in range(args.train_num)])
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(args.env_id) for _ in range(args.test_num)])

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
