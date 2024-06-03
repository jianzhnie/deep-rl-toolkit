import os
import sys
import time

import gymnasium as gym
import numpy as np

sys.path.append((os.path.join(os.path.dirname(__file__), '..')))
from rltoolkit.data import BaseBuffer
from rltoolkit.utils import (ProgressBar, TensorboardLogger, WandbLogger,
                             get_outdir, get_text_logger)
from torch.utils.tensorboard import SummaryWriter

from cleanrl.base_agent import BaseAgent
from cleanrl.rl_args import RLArguments


class Runner:

    def __init__(
        self,
        args: RLArguments,
        train_env: gym.Env,
        test_env: gym.Env,
        agent: BaseAgent,
        buffer: BaseBuffer,
    ) -> None:
        self.args = args
        self.train_env = train_env
        self.test_env = test_env
        self.agent = agent
        self.buffer = buffer

        # Training
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
    def run_train_episode(self):
        episode_reward = 0
        episode_step = 0
        obs, _ = self.train_env.reset()
        done = False
        while not done:
            episode_step += 1
            action = self.agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = self.train_env.step(
                action)
            done = np.logical_or(terminated, truncated)
            self.buffer.add(obs, next_obs, action, reward, done, info)
            if self.buffer.size() > self.args.warmup_learn_steps:
                if self.global_step % self.args.train_frequency == 0:
                    graddient_step_losses = []
                    for _ in range(self.args.gradient_steps):
                        batchs = self.buffer.sample(self.args.batch_size)
                        loss = self.agent.learn(batchs)
                        graddient_step_losses.append(loss)

            episode_reward += reward
            obs = next_obs

        train_info = {
            'episode_reward': episode_reward,
            'episode_step': episode_step,
            'loss': np.mean(graddient_step_losses),
        }
        return train_info

    def run_evaluate_episodes(
        self,
        n_eval_episodes: int = 5,
    ) -> dict[str, float]:
        eval_rewards = []
        eval_steps = []
        for _ in range(n_eval_episodes):
            self.test_env.seed(np.random.randint(100))
            obs = self.test_env.reset()
            done = False
            episode_reward = 0.0
            episode_step = 0
            while not done:
                action = self.agent.predict(obs)
                next_obs, reward, terminated, truncated, info = self.test_env.step(
                    action)
                obs = next_obs
                episode_reward += reward
                episode_step += 1
                if done:
                    self.test_env.close()
            eval_rewards.append(episode_reward)
            eval_steps.append(episode_step)
        mean_reward = np.mean(eval_rewards)
        mean_steps = np.mean(eval_steps)
        test_info = {
            'reward_mean': mean_reward,
            'length_mean': mean_steps,
        }
        return test_info

    def run(self) -> None:
        """Train the agent."""
        self.text_logger.info('Start Training')
        progress_bar = ProgressBar(self.args.max_timesteps)
        episode_cnt = 0
        while self.buffer.size() < self.args.warmup_learn_steps:
            train_info = self.run_train_episode()
            progress_bar.update(train_info['episode_step'])

        while self.global_step < self.args.max_timesteps:
            # Training logic
            train_info = self.run_train_episode()
            episode_step = train_info['episode_step']
            self.progress_bar.update(episode_step)
            self.global_step += episode_step
            episode_cnt += 1

            train_info['learning_rate'] = self.args.learning_rate
            train_info['eps_greedy'] = self.eps_greedy
            train_info['episode_cnt'] = episode_cnt

            # Log training information
            train_fps = int(self.global_step / (time.time() - self.start_time))
            train_info['fps'] = train_fps
            log_message = ('[Train], global_step: {}, train_fps: {}, '
                           'loss: {:.2f}'.format(self.global_step, train_fps,
                                                 train_info['loss']))
            self.text_logger.info(log_message)
            self.log_train_infos(train_info, self.global_step)

            # perform evaluation
            if episode_cnt % self.args.test_log_interval == 0:
                test_info = self.run_evaluate_episodes(n_eval_episodes=5)
                self.text_logger.info(
                    '[Eval], episode: {}, eval_rewards: {:.2f}'.format(
                        episode_cnt, test_info.get('reward_mean', 0.0)))
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