import time
from typing import Dict, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from rltoolkit.cleanrl.agent import BaseAgent
from rltoolkit.cleanrl.base_runner import BaseRunner
from rltoolkit.cleanrl.rl_args import RLArguments
from rltoolkit.cleanrl.utils.utils import calculate_mean
from rltoolkit.data import SimpleRolloutBuffer
from rltoolkit.utils import ProgressBar


class OnPolicyRunner(BaseRunner):
    """Runner for on-policy training and evaluation."""

    def __init__(
        self,
        args: RLArguments,
        train_env: gym.Env,
        test_env: gym.Env,
        agent: BaseAgent,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize the OnPolicyRunner.

        Args:
            args (RLArguments): Configuration arguments for RL.
            train_env (gym.Env): Training environment.
            test_env (gym.Env): Testing environment.
            agent (BaseAgent): Agent to be trained and evaluated.
            device (Optional[Union[str, torch.device]]): Device to use.
        """
        super().__init__(args, train_env, test_env, agent)

        self.buffer = SimpleRolloutBuffer(
            args=args,
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            device=device,
        )
        # Training
        self.episode_cnt = 0
        self.global_step = 0
        self.start_time = time.time()
        self.device = device if device is not None else torch.device('cpu')

    def run_evaluate_episodes(self,
                              n_eval_episodes: int = 5) -> Dict[str, float]:
        """Run evaluation episodes and collect statistics.

        Args:
            n_eval_episodes (int): Number of evaluation episodes.

        Returns:
            Dict[str, float]: A dictionary with evaluation statistics.
        """
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
                episode_reward += reward
                episode_step += 1
                done = terminated or truncated

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
        total_steps = self.args.num_episode * self.args.rollout_length
        progress_bar = ProgressBar(total_steps)
        self.start_time = time.time()
        obs, _ = self.train_env.reset()
        for self.episode_cnt in range(1, self.args.num_episode + 1):
            # Collect rollout data
            self.buffer.reset()
            for step in range(self.args.rollout_length):
                progress_bar.update(1)
                self.global_step += 1

                value, action, log_prob, entropy = self.agent.get_action(obs)
                next_obs, reward, terminations, truncations, infos = (
                    self.train_env.step(action))
                obs = next_obs
                done = np.logical_or(terminations, truncations)
                self.buffer.add(obs, action, reward, done, value, log_prob)

            # Bootstrap value if not done
            last_value = self.agent.get_value(obs)
            self.buffer.compute_returns_and_advantage(last_value, done)

            # Learn from the collected rollout data
            episode_learn_info = []
            for batch_data in self.buffer.sample(self.args.batch_size):
                learn_info = self.agent.learn(batch_data)
                episode_learn_info.append(learn_info)

            train_info = calculate_mean(episode_learn_info)
            train_info['num_episode'] = self.episode_cnt
            train_info['num_steps'] = self.global_step

            # Calculate training FPS
            train_fps = int(self.global_step / (time.time() - self.start_time))
            train_info['fps'] = train_fps

            # Log training information
            if self.episode_cnt % self.args.train_log_interval == 0:
                log_message = (
                    '[Train], global_step: {}, episode: {}, train_fps: {}'.
                    format(self.global_step, self.episode_cnt, train_fps))
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
