import random
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

sys.path.append('../')
from rltoolkit.agents import DQNAgent
from rltoolkit.agents.configs import BaseConfig as DQNConfig


def make_env(env_id, seed, idx, capture_video):

    def thunk() -> gym.wrappers.RecordEpisodeStatistics:
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode='rgb_array')
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):

    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(
                np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':
    config = DQNConfig()
    # TRY NOT TO MODIFY: seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    device = torch.device(
        'cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')

    run_name = f'{config.env_id}_{int(time.time())}'

    # env setup
    envs = gym.vector.SyncVectorEnv([
        make_env(config.env_id, config.seed + i, i, False)
        for i in range(config.num_envs)
    ])
    eval_envs = gym.vector.SyncVectorEnv([
        make_env(config.env_id, 0, 0, config.capture_video) for i in range(10)
    ])
    assert isinstance(
        envs.single_action_space,
        gym.spaces.Discrete), 'only discrete action space is supported'

    agent = DQNAgent(
        config=config,
        envs=envs,
        device=device,
    )
    agent.train()
