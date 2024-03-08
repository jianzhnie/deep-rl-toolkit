import argparse
import datetime
import os
import pprint
import sys
import warnings

import numpy as np
import torch
from atari_network import DQN
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../../')
from rltoolkit.buffers import Collector
from rltoolkit.envs.atari_wrappers import wrap_deepmind
from rltoolkit.policy import DQNPolicy
from rltoolkit.trainer import offpolicy_trainer
from rltoolkit.utils import TensorboardLogger, WandbLogger
from tianshou.data import VectorReplayBuffer
from tianshou.env import ShmemVectorEnv

try:
    import envpool
except ImportError:
    envpool = None


def make_atari_env(env_id, seed, train_num, test_num, **kwargs):
    """Wrapper function for Atari env.

    If EnvPool is installed, it will automatically switch to EnvPool's Atari env.

    :return: a tuple of (single env, training envs, test envs).
    """
    if envpool is not None:
        if kwargs.get('scale', 0):
            warnings.warn(
                'EnvPool does not include ScaledFloatFrame wrapper, '
                "please set `x = x / 255.0` inside CNN network's forward function."
            )
        # parameters convertion
        train_envs = env = envpool.make_gymnasium(
            env_id.replace('NoFrameskip-v4', '-v5'),
            num_envs=train_num,
            seed=seed,
            episodic_life=True,
            reward_clip=True,
            stack_num=kwargs.get('frame_stack', 4),
        )
        test_envs = envpool.make_gymnasium(
            env_id.replace('NoFrameskip-v4', '-v5'),
            num_envs=test_num,
            seed=seed,
            episodic_life=False,
            reward_clip=False,
            stack_num=kwargs.get('frame_stack', 4),
        )
    else:
        warnings.warn('Recommend using envpool (pip install envpool) '
                      'to run Atari games more efficiently.')
        env = wrap_deepmind(env_id, **kwargs)
        train_envs = ShmemVectorEnv([
            lambda: wrap_deepmind(
                env_id, episode_life=True, clip_rewards=True, **kwargs)
            for _ in range(train_num)
        ])
        test_envs = ShmemVectorEnv([
            lambda: wrap_deepmind(
                env_id, episode_life=False, clip_rewards=False, **kwargs)
            for _ in range(test_num)
        ])
        env.seed(seed)
        train_envs.seed(seed)
        test_envs.seed(seed)
    return env, train_envs, test_envs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--algo-name', type=str, default='dqn')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--scale-obs', type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.0)
    parser.add_argument('--eps-train-final', type=float, default=0.05)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=500)
    parser.add_argument('--target-update-tau', type=float, default=1.0)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--train-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.0)
    parser.add_argument('--device',
                        type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames-stack', type=int, default=4)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--resume-id', type=str, default=None)
    parser.add_argument(
        '--logger',
        type=str,
        default='tensorboard',
        choices=['tensorboard', 'wandb'],
    )
    parser.add_argument('--wandb-project', type=str, default='atari.benchmark')
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='watch the play of pre-trained policy only',
    )
    parser.add_argument('--save-buffer-name', type=str, default=None)
    return parser.parse_args()


def test_dqn(args=get_args()):
    env, train_envs, test_envs = make_atari_env(
        args.env_id,
        args.seed,
        args.train_num,
        args.test_num,
        scale=args.scale_obs,
        frame_stack=args.frames_stack,
    )
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print('Observations shape:', args.state_shape)
    print('Actions shape:', args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # define model
    net = DQN(*args.state_shape, args.action_shape,
              args.device).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # define policy
    policy = DQNPolicy(net,
                       optim,
                       args.gamma,
                       args.n_step,
                       target_update_freq=args.target_update_freq)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(
            torch.load(args.resume_path, map_location=args.device))
        print('Loaded agent from: ', args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=args.frames_stack,
    )
    # collector
    train_collector = Collector(policy,
                                train_envs,
                                buffer,
                                exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # log
    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    log_name = os.path.join(args.env_id, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    if args.logger == 'wandb':
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, '__'),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(log_path)
    writer.add_text('args', str(args))
    if args.logger == 'tensorboard':
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        elif 'Pong' in args.env_id:
            return mean_rewards >= 20
        else:
            return False

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * (args.eps_train -
                                                     args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        if env_step % 1000 == 0:
            logger.write('train/env_step', env_step, {'train/eps': eps})

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, f'checkpoint_{epoch}.pth')
        torch.save({'model': policy.state_dict()}, ckpt_path)
        return ckpt_path

    # watch agent's performance
    def watch():
        print('Setup test envs ...')
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f'Generate buffer with size {args.buffer_size}')
            buffer = VectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack,
            )
            collector = Collector(policy,
                                  test_envs,
                                  buffer,
                                  exploration_noise=True)
            result = collector.collect(n_step=args.buffer_size)
            print(f'Save buffer into {args.save_buffer_name}')
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
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
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
        resume_from_log=args.resume_id is not None,
        save_checkpoint_fn=save_checkpoint_fn,
    )

    pprint.pprint(result)
    watch()


if __name__ == '__main__':
    test_dqn(get_args())
