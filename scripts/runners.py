import random
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('../')
from rltoolkit.agents import DQNAgent
from rltoolkit.agents.configs import BaseConfig as DQNConfig
from rltoolkit.buffers import OffPolicyBuffer
from rltoolkit.utils import LinearDecayScheduler


def make_env(env_id, seed, idx, capture_video, run_name):

    def thunk():
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
        'cuda' if torch.cuda.is_available() and config.cuda else 'cpu')

    run_name = f'{config.env_name}_{int(time.time())}'

    # env setup
    envs = gym.vector.SyncVectorEnv([
        make_env(config.env_name, config.seed + i, i, config.capture_video,
                 run_name) for i in range(config.num_envs)
    ])
    assert isinstance(
        envs.single_action_space,
        gym.spaces.Discrete), 'only discrete action space is supported'

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=config.learning_rate)

    rb = OffPolicyBuffer(
        config.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    eps_greedy_scheduler = LinearDecayScheduler(
        config.eps_greedy_start,
        config.eps_greedy_end,
        max_steps=config.max_train_steps)

    agent = DQNAgent(
        config=config,
        envs=envs,
        buffer=rb,
        actor_model=q_network,
        actor_optimizer=optimizer,
        eps_greedy_scheduler=eps_greedy_scheduler,
        device=device,
    )

    agent.train()
