import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../../')
from rltoolkit.buffers.collector import Collector
from rltoolkit.policy.modelfree.dqn import DQNPolicy
from rltoolkit.trainer.offpolicy import OffpolicyTrainer
from tianshou.data import VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='CartPole-v1')
    parser.add_argument('--algo-name', type=str, default='dqn')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--scale-obs', type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.0)
    parser.add_argument('--eps-train-final', type=float, default=0.05)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=500)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step-per-epoch', type=int, default=10000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--train-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.0)
    parser.add_argument('--device',
                        type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument(
        '--logger',
        type=str,
        default='tensorboard',
        choices=['tensorboard', 'wandb'],
    )
    parser.add_argument('--wandb-project', type=str, default='CartPole')
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='watch the play of pre-trained policy only',
    )
    parser.add_argument('--save-buffer-name', type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    # You can also try SubprocVectorEnv, which will use parallelization
    train_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.train_num)])
    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)])

    # Note: You can easily define other networks.
    # See https://tianshou.readthedocs.io/en/master/01_tutorials/00_dqn.html#build-the-network
    env = gym.make(args.task, render_mode='human')
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    args.total_steps = args.epoch * args.step_per_epoch
    print('Observations shape:', state_shape)
    print('Actions shape:', action_shape)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # define model
    net = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=[128, 128, 128],
        device=args.device,
    ).to(args.device)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    # define policy
    policy: DQNPolicy = DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=args.gamma,
        action_space=env.action_space,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(
            torch.load(args.resume_path, map_location=args.device))
        print('Loaded agent from: ', args.resume_path)

    # replay buffer
    buffer = VectorReplayBuffer(args.buffer_size, args.train_num)
    # data collector
    train_collector = Collector(
        policy,
        train_envs,
        buffer,
        exploration_noise=True,
    )
    test_collector = Collector(
        policy,
        test_envs,
        exploration_noise=True,
    )  # because DQN uses epsilon-greedy method

    log_name = os.path.join(args.task, args.algo_name)
    log_path = os.path.join(args.logdir, log_name)

    logger = TensorboardLogger(
        SummaryWriter(log_path))  # TensorBoard is supported!

    # For other loggers, see https://tianshou.readthedocs.io/en/master/tutorials/logger.html

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, f'checkpoint_{epoch}.pth')
        torch.save({'model': policy.state_dict()}, ckpt_path)
        return ckpt_path

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= args.total_steps:
            eps = args.eps_train - env_step / args.total_steps * (
                args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        if env_step % 100 == 0:
            logger.write('train/env_step', env_step, {'train/eps': eps})

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec:
            if not env.spec.reward_threshold:
                return False
            else:
                return mean_rewards >= env.spec.reward_threshold
        return False

    # watch agent's performance
    def watch():
        print('Setup test envs ...')
        policy.eval()
        policy.set_eps(args.eps_test)
        print('Testing agent ...')
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num,
                                        render=args.render)
        rew = result['rews'].mean()
        print(f"Mean reward (over {result['n/ep']} episodes): {rew}")

    if args.watch:
        watch()
        exit(0)

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.train_num)
    # trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
    ).run()
    print(result)


if __name__ == '__main__':
    main()
