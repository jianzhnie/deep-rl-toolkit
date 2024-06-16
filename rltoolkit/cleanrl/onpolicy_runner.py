import time

import gymnasium as gym
import numpy as np
from rltoolkit.cleanrl.agent import BaseAgent
from rltoolkit.cleanrl.base_runner import BaseRunner
from rltoolkit.cleanrl.rl_args import RLArguments
from rltoolkit.cleanrl.utils.utils import calculate_mean
from rltoolkit.data import SimpleRolloutBuffer
from rltoolkit.utils import ProgressBar


class OnPolicyRunner(BaseRunner):

    def __init__(
        self,
        args: RLArguments,
        train_env: gym.Env,
        test_env: gym.Env,
        agent: BaseAgent,
        buffer: SimpleRolloutBuffer,
    ) -> None:
        super().__init__(args, train_env, test_env, agent, buffer)
        self.args = args
        self.train_env = train_env
        self.test_env = test_env
        self.buffer: SimpleRolloutBuffer = buffer

    def run_train_episode(self, obs: np.array) -> dict[str, float]:
        # Training logic
        episode_result_info = []
        for step in range(self.args.rollout_length):
            self.global_step += 1
            (
                value,
                action,
                log_prob,
                entropy,
            ) = self.agent.get_action(obs)
            next_obs, reward, terminations, truncations, infos = self.train_env.step(
                action)
            obs = next_obs
            done = np.logical_or(terminations, truncations)
            self.buffer.add(obs, action, reward, done, value, log_prob)

        # Bootstrap value if not done
        last_value = self.agent.get_value(obs)
        self.buffer.compute_returns_and_advantage(last_value, done)
        for batch_data in self.buffer.sample(self.args.batch_size):
            learn_info = self.agent.learn(batch_data)
            episode_result_info.append(learn_info)

        episode_info = calculate_mean(episode_result_info)
        return episode_info

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
                episode_reward += reward
                episode_step += 1
                done = terminated or truncated
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
        self.global_step = 0
        obs, _ = self.train_env.reset()
        total_steps = self.args.num_episode * self.args.rollout_length
        progress_bar = ProgressBar(total_steps)
        for self.episode_cnt in range(1, self.args.num_episode + 1):
            self.global_step += self.args.rollout_length
            progress_bar.update(self.args.rollout_length)
            train_info = self.run_train_episode(obs)
            train_info['num_episode'] = self.episode_cnt

            # Calculate training FPS
            train_fps = int(self.global_step / (time.time() - self.start_time))
            train_info['fps'] = train_fps

            # Log training information
            if self.episode_cnt % self.args.train_log_interval == 0:
                log_message = (
                    '[Train], global_step: {}, episode: {}, train_fps: {}, '
                ).format(self.global_step, self.episode_cnt, train_fps)
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
