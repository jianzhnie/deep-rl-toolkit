import os
import sys
import time

import gymnasium as gym
import numpy as np

sys.path.append((os.path.join(os.path.dirname(__file__), '..')))
from rltoolkit.data import BaseBuffer
from rltoolkit.utils import (ProgressBar, TensorboardLogger, WandbLogger,
                             get_outdir, get_text_logger, soft_target_update)
from torch.utils.tensorboard import SummaryWriter

from cleanrl.base_agent import BaseAgent
from cleanrl.rl_args import RLArguments


class Runner:

    def __init__(
        self,
        args: RLArguments,
        train_envs: gym.Env,
        test_envs: gym.Env,
        agent: BaseAgent,
        buffer: BaseBuffer,
    ) -> None:
        self.args = args
        self.train_envs = train_envs
        self.test_envs = test_envs
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
            wandb_log_dir = get_outdir(work_dir, 'wandb_log')
            self.vis_logger = WandbLogger(
                dir=wandb_log_dir,
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
        episode_loss = []
        obs, _ = self.train_envs.reset()
        done = False
        while not done:
            episode_step += 1
            action = self.agent.sample(obs)
            next_obs, reward, terminated, truncated, info = self.train_envs.step(
                action)
            done = np.logical_or(terminated, truncated)
            self.buffer.add(obs, next_obs, action, reward, done, info)
            # train model
            if self.buffer.size() > self.args.wormup_learn_steps:
                samples = self.buffer.sample(self.args.batch_size)
                loss = self.agent.learn(samples)
                episode_loss.append(loss)
            episode_reward += reward
            obs = next_obs
        return episode_reward, episode_step, np.mean(episode_loss)

    def run_evaluate_episodes(
        self,
        n_eval_episodes: int = 5,
    ):
        eval_rewards = []
        eval_steps = []
        for _ in range(n_eval_episodes):
            self.test_envs.seed(np.random.randint(100))
            obs = self.test_envs.reset()
            done = False
            episode_reward = 0.0
            episode_step = 0
            while not done:
                action = self.agent.predict(obs)
                next_obs, reward, done, _ = self.test_envs.step(action)
                obs = next_obs
                episode_reward += reward
                episode_step += 1
                if done:
                    self.test_envs.close()
            eval_rewards.append(episode_reward)
            eval_steps.append(episode_step)
        mean_reward = np.mean(eval_rewards)
        mean_steps = np.mean(eval_steps)
        return mean_reward, mean_steps

    def run(self) -> None:
        """Train the agent."""
        obs, _ = self.train_envs.reset(seed=self.args.seed)
        self.text_logger.info('Start Training')
        while self.global_step < self.args.max_timesteps:
            self.eps_greedy = self.agent.eps_greedy_scheduler.step()
            actions = self.agent.get_action(obs, eps_greedy=self.eps_greedy)
            next_obs, rewards, terminals, truncations, infos = self.train_envs.step(
                actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if 'final_info' in infos:
                for info in infos['final_info']:
                    if info and 'episode' in info:
                        self.text_logger.info(
                            f"global_step={self.global_step}, episodic_return={info['episode']['r']}"
                        )
                        self.writer.add_scalar(
                            'charts/episodic_return',
                            info['episode']['r'],
                            self.global_step,
                        )
                        self.writer.add_scalar(
                            'charts/episodic_length',
                            info['episode']['l'],
                            self.global_step,
                        )

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos['final_observation'][idx]

            dones = np.logical_or(terminals, truncations)
            self.buffer.add(obs, next_obs, actions, rewards, dones, infos)

            # Crucial step easy to overlook
            obs = next_obs

            # Training logic
            if self.global_step >= self.args.warmup_learn_steps:
                if self.global_step % self.args.train_frequency == 0:
                    # Learner: Update model parameters
                    graddient_step_losses = []
                    step_info = dict()
                    for _ in range(self.args.gradient_steps):
                        batchs = self.buffer.sample(self.args.batch_size)
                        loss = self.agent.learn(batchs)
                        graddient_step_losses.append(loss)

                    step_info['loss'] = np.mean(graddient_step_losses)
                    step_info['learning_rate'] = self.args.learning_rate
                    step_info['eps_greedy'] = self.eps_greedy
                    step_info['num_episode'] = 1
                    step_info['reward_mean'] = 0
                    step_info['length_mean'] = 0

                    # Log training information
                    train_fps = int(self.global_step /
                                    (time.time() - self.start_time))
                    step_info['fps'] = train_fps
                    log_message = ('[Train], global_step: {}, train_fps: {}, '
                                   'loss: {:.2f}'.format(
                                       self.global_step, train_fps,
                                       step_info['loss']))
                    self.text_logger.info(log_message)
                    self.log_train_infos(step_info, self.global_step)

                # Update target network
                if self.global_step % self.args.target_update_frequency == 0:
                    self.text_logger.info('Update Target Model')
                    soft_target_update(
                        src_model=self.agent.q_network,
                        tgt_model=self.agent.q_target,
                        tau=self.args.soft_update_tau,
                    )
            self.global_step += 1
            self.progress_bar.update(1)

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
