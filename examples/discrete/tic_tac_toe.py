"""This is a minimal example of using rltoolkit with MARL to train agents.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/rltoolkit
"""

import os
import sys

import gymnasium
import numpy as np
import torch
from pettingzoo.classic import tictactoe_v3

sys.path.append(os.getcwd())
from rltoolkit.data import Collector, VectorReplayBuffer
from rltoolkit.envs import DummyVectorEnv
from rltoolkit.envs.pettingzoo_env import PettingZooEnv
from rltoolkit.policy import DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from rltoolkit.trainer import offpolicy_trainer
from rltoolkit.utils.net.common import Net


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    return PettingZooEnv(tictactoe_v3.env())


if __name__ == '__main__':
    # ======== Step 1: Environment setup =========
    train_envs = DummyVectorEnv([_get_env for _ in range(10)])
    test_envs = DummyVectorEnv([_get_env for _ in range(10)])

    # seed
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    env = _get_env()
    observation_space = (env.observation_space['observation'] if isinstance(
        env.observation_space, gymnasium.spaces.Dict) else
                         env.observation_space)
    # model
    net = Net(
        state_shape=observation_space.shape or observation_space.n,
        action_shape=env.action_space.shape or env.action_space.n,
        hidden_sizes=[128, 128, 128, 128],
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)

    # ======== Step 2: Agent setup =========
    agent_learn = DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=0.9,
        estimation_step=3,
        action_space=env.action_space,
        target_update_freq=320,
    )
    agent_opponent = RandomPolicy()

    agents = [agent_opponent, agent_learn]
    policy = MultiAgentPolicyManager(agents, env)

    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(20000, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=64 * 10)  # batch size * training_num

    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model_save_path = os.path.join('log', 'ttt', 'dqn', 'policy.pth')
        os.makedirs(os.path.join('log', 'ttt', 'dqn'), exist_ok=True)
        torch.save(policy.policies[agents[1]].state_dict(), model_save_path)

    def stop_fn(mean_rewards):
        return mean_rewards >= 0.6

    def train_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.1)

    def test_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.05)

    def reward_metric(rews):
        return rews[:, 1]

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50,
        step_per_epoch=1000,
        step_per_collect=50,
        episode_per_test=10,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    # return result, policy.policies[agents[1]]
    print(f'\n==========Result==========\n{result}')
    print(
        '\n(the trained policy can be accessed via policy.policies[agents[1]])'
    )
