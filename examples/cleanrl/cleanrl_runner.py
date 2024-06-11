import os
import random
import sys

import numpy as np
import torch
import tyro

sys.path.append('../../')
sys.path.append(os.getcwd())
import gymnasium as gym
from rltoolkit.cleanrl.agent import DQNAgent
from rltoolkit.cleanrl.offpolicy_runner import OffPolicyRunner
from rltoolkit.cleanrl.rl_args import DQNArguments
from rltoolkit.data import SimpleReplayBuffer


def make_env(env_id, seed):

    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


if __name__ == '__main__':
    args: DQNArguments = tyro.cli(DQNArguments)
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    train_env: gym.Env = gym.make(args.env_id)
    test_env: gym.Env = gym.make(args.env_id)
    state_shape = train_env.observation_space.shape or train_env.observation_space.n
    action_shape = train_env.action_space.shape or train_env.action_space.n

    device = torch.device(
        'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print('Observations shape:', state_shape)
    print('Actions shape:', action_shape)

    # agent
    agent = DQNAgent(
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
