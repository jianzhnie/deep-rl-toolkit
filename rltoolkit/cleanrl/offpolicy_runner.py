import os
import time

import gymnasium as gym
import numpy as np
from rltoolkit.cleanrl.agent import BaseAgent
from rltoolkit.cleanrl.rl_args import RLArguments
from rltoolkit.cleanrl.utils.utils import calculate_mean
from rltoolkit.data import SimpleReplayBuffer as ReplayBuffer
from rltoolkit.utils import (ProgressBar, TensorboardLogger, WandbLogger,
                             get_outdir, get_text_logger)
from torch.utils.tensorboard import SummaryWriter


class OffPolicyRunner:

    def __init__(
        self,
        args: RLArguments,
        train_env: gym.Env,
        test_env: gym.Env,
        agent: BaseAgent,
        buffer: ReplayBuffer,
    ) -> None:
        self.args = args
        self.train_env = train_env
        self.test_env = test_env
        self.agent = agent
        self.buffer = buffer

        # Training
        self.episode_cnt = 0
        self.global_step = 0
        self.start_time = time.time()
        self.eps_greedy = 0.0

        # Logs and Visualizations
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_name = os.path.join(args.project, args.env_id, args.algo_name,
                                timestamp).replace(os.path.sep, '_')
        work_dir = os.path.join(args.work_dir, args.project, args.env_id,
                                args.algo_name)
        tensorboard_log_dir = get_outdir(work_dir, 'tensorboard_log')
        text_log_dir = get_outdir(work_dir, 'text_log')
        text_log_file = os.path.join(text_log_dir, log_name + '.log')
        self.text_logger = get_text_logger(log_file=text_log_file,
                                           log_level='INFO')

        if args.logger == 'wandb':
            self.vis_logger = WandbLogger(
                dir=work_dir,
                train_interval=args.train_log_interval,
                test_interval=args.test_log_interval,
                update_interval=args.train_log_interval,
                save_interval=args.save_interval,
                project=args.project,
                name=log_name,
                config=args,
            )
        self.writer = SummaryWriter(tensorboard_log_dir)
        self.writer.add_text('args', str(args))
        if args.logger == 'tensorboard':
            self.vis_logger = TensorboardLogger(self.writer)
        else:  # wandb
            self.vis_logger.load(self.writer)

        # ProgressBar
        self.progress_bar = ProgressBar(args.max_timesteps)
        # Video Save
        self.video_save_dir = get_outdir(work_dir, 'video_dir')
        self.model_save_dir = get_outdir(work_dir, 'model_dir')

    # train an episode
    def run_train_episode(self) -> dict[str, float]:
        episode_result_info = []
        obs, _ = self.train_env.reset()
        done = False
        while not done:
            action = self.agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = self.train_env.step(
                action)
            self.global_step += 1
            done = terminated or truncated
            if info and 'episode' in info:
                info_item = {k: v.item() for k, v in info['episode'].items()}
                episode_reward = info_item['r']
                episode_step = info_item['l']
            self.buffer.add(obs, next_obs, action, reward, done)
            if self.buffer.size() > self.args.warmup_learn_steps:
                if self.global_step % self.args.train_frequency == 0:
                    for _ in range(self.args.gradient_steps):
                        batchs = self.buffer.sample(self.args.batch_size)
                        learn_result = self.agent.learn(batchs)
                        episode_result_info.append(learn_result)

            obs = next_obs
            if done:
                break
        episode_info = calculate_mean(episode_result_info)
        train_info = {
            'episode_reward': episode_reward,
            'episode_step': episode_step,
        }
        train_info.update(episode_info)
        return train_info

    def run_evaluate_episodes(self,
                              n_eval_episodes: int = 5) -> dict[str, float]:
        eval_rewards = []
        eval_steps = []
        for _ in range(n_eval_episodes):
            obs, info = self.test_env.reset()
            done = False
            episode_reward = 0.0
            episode_step = 0
            while not done:
                action = self.agent.predict(obs)
                next_obs, reward, terminated, truncated, info = self.test_env.step(
                    action)
                obs = next_obs
                done = terminated or truncated
                if info and 'episode' in info:
                    info_item = {
                        k: v.item()
                        for k, v in info['episode'].items()
                    }
                    episode_reward = info_item['r']
                    episode_step = info_item['l']
                if done:
                    self.test_env.reset()
            eval_rewards.append(episode_reward)
            eval_steps.append(episode_step)

        reward_mean = np.mean(eval_rewards)
        reward_std = np.std(eval_rewards)
        length_mean = np.mean(eval_steps)
        length_std = np.std(eval_steps)
        test_info = {
            'reward_mean': reward_mean,
            'reward_std': reward_std,
            'length_mean': length_mean,
            'length_std': length_std,
        }
        return test_info

    def run(self) -> None:
        """Train the agent."""
        self.text_logger.info('Start Training')
        progress_bar = ProgressBar(self.args.max_timesteps)
        while self.global_step < self.args.max_timesteps:
            # Training logic
            train_info = self.run_train_episode()
            episode_step = train_info['episode_step']
            progress_bar.update(episode_step)
            self.episode_cnt += 1

            train_info['num_episode'] = self.episode_cnt
            train_info['rpm_size'] = self.buffer.size()
            train_info['eps_greedy'] = (self.agent.eps_greedy if hasattr(
                self.agent, 'eps_greedy') else 0.0)
            train_info['learning_rate'] = self.agent.learning_rate
            train_info['learner_update_step'] = self.agent.learner_update_step
            train_info[
                'target_model_update_step'] = self.agent.target_model_update_step

            # Log training information
            train_fps = int(self.global_step / (time.time() - self.start_time))
            train_info['fps'] = train_fps

            # Log training information
            if self.episode_cnt % self.args.train_log_interval == 0:
                log_message = (
                    '[Train], global_step: {}, episodes: {}, train_fps: {}, '
                    'episode_reward: {:.2f}, episode_step: {:.2f}').format(
                        self.global_step,
                        self.episode_cnt,
                        train_fps,
                        train_info['episode_reward'],
                        train_info['episode_step'],
                    )
                self.text_logger.info(log_message)
                self.log_train_infos(train_info, self.global_step)

            # Log testing information
            if self.episode_cnt % self.args.test_log_interval == 0:
                test_info = self.run_evaluate_episodes(
                    n_eval_episodes=self.args.eval_episodes)
                test_info['num_episode'] = self.episode_cnt
                self.text_logger.info(
                    '[Eval], global_step: {}, episode: {}, eval_rewards: {:.2f}'
                    .format(self.global_step, self.episode_cnt,
                            test_info['reward_mean']))
                self.log_test_infos(test_info, self.global_step)

        # Save model
        if self.args.save_model:
            self.agent.save_model(self.model_save_dir)

    def log_train_infos(self, infos: dict, steps: int) -> None:
        """Log training information.

        Args:
            infos (dict): Information to be visualized.
            steps (int): Current training step.
        """
        self.vis_logger.log_train_data(infos, steps)

    def log_test_infos(self, infos: dict, steps: int) -> None:
        """Log testing information.

        Args:
            infos (dict): Information to be visualized.
            steps (int): Current training step.
        """
        self.vis_logger.log_test_data(infos, steps)
