import time
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch
from rltoolkit.cleanrl.agent import BaseAgent
from rltoolkit.cleanrl.base_runner import BaseRunner
from rltoolkit.cleanrl.rl_args import RLArguments
from rltoolkit.cleanrl.utils.utils import calculate_mean
from rltoolkit.data import SimpleReplayBuffer
from rltoolkit.utils import ProgressBar


class OffPolicyRunner(BaseRunner):

    def __init__(
        self,
        args: RLArguments,
        train_env: gym.Env,
        test_env: gym.Env,
        agent: BaseAgent,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(args, train_env, test_env, agent)

        self.buffer = SimpleReplayBuffer(
            args=args,
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            device=device,
        )
        # Training
        self.episode_cnt = 0
        self.global_step = 0
        self.start_time = time.time()
        self.eps_greedy = 0.0
        self.device = device if device is not None else torch.device('cpu')

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

        return {
            'reward_mean': np.mean(eval_rewards),
            'reward_std': np.std(eval_rewards),
            'length_mean': np.mean(eval_steps),
            'length_std': np.std(eval_steps),
        }

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
            train_info['learning_rate'] = (self.agent.learning_rate if hasattr(
                self.agent, 'learning_rate') else 0.0)
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
                log_message = (
                    '[Eval], global_step: {}, episode: {}, eval_rewards: {:.2f}'
                    .format(self.global_step, self.episode_cnt,
                            test_info['reward_mean']))
                self.text_logger.info(log_message)
                self.log_test_infos(test_info, self.global_step)

        # Save model
        if self.args.save_model:
            self.agent.save_model(self.model_save_dir)
