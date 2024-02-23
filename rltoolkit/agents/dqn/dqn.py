import time
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rltoolkit.agents.base_agent import BaseAgent
from rltoolkit.agents.configs import BaseConfig
from rltoolkit.buffers import BaseBuffer
from rltoolkit.utils import soft_target_update
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class DQNAgent(BaseAgent):
    """Deep Q-Network algorithm.

    “Human-Level Control Through Deep Reinforcement Learning”. Mnih V. et al..
    2015.

    Args:
        config (Namespace): Configuration object for the agent.
        envs (Union[None]): Environment object.
        buffer (BaseBuffer): Replay buffer for experience replay.
        actor_model (nn.Module): Actor model representing the policy network.
        critic_model (Optional[nn.Module], optional): Critic model representing the value network. Defaults to None.
        actor_optimizer (Optimizer, optional): Optimizer for actor model. Defaults to None.
        critic_optimizer (Optional[Optimizer], optional): Optimizer for critic model. Defaults to None.
        actor_lr_scheduler (Optional[_LRScheduler], optional): Learning rate scheduler for actor model. Defaults to None.
        critic_lr_scheduler (Optional[_LRScheduler], optional): Learning rate scheduler for critic model. Defaults to None.
        eps_greedy_scheduler (Optional[_LRScheduler], optional): Epsilon-greedy scheduler. Defaults to None.
        device (Optional[Union[str, torch.device]], optional): Device to run the agent on. Defaults to None.
    """

    def __init__(
        self,
        config: BaseConfig,
        envs: Union[None],
        eval_envs: Union[None],
        buffer: BaseBuffer,
        actor_model: nn.Module,
        critic_model: Optional[nn.Module] = None,
        actor_optimizer: Optimizer = None,
        critic_optimizer: Optional[Optimizer] = None,
        actor_lr_scheduler: Optional[LRScheduler] = None,
        critic_lr_scheduler: Optional[LRScheduler] = None,
        eps_greedy_scheduler: Optional[LRScheduler] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(
            config,
            envs=envs,
            eval_envs=eval_envs,
            buffer=buffer,
            actor_model=actor_model,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            actor_lr_scheduler=actor_lr_scheduler,
            eps_greedy_scheduler=eps_greedy_scheduler,
            device=device,
        )
        self.start_time = time.time()

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from the actor network.

        Sample an action when given an observation, based on the current
        epsilon value, either a greedy action or a random action will be
        returned.

        Args:
            obs: Current observation

        Returns:
            actions (np.array): Action
        """
        # Choose a random action with probability epsilon
        if np.random.rand() <= self.eps_greedy:
            actions = np.array([
                self.envs.single_action_space.sample()
                for _ in range(self.envs.num_envs)
            ])
        else:
            actions = self.predict(obs)
        return actions

    def predict(self, obs: torch.Tensor) -> Union[int, List[int]]:
        """Predict an action when given an observation, a greedy action will be
        returned.

        Args:
            obs (torch.Tensor): Current observation

        Returns:
            actions (Union[int, List[int]]): Action
        """
        if obs.ndim == 1:
            # If obs is 1-dimensional, we need to expand it to have batch_size = 1
            obs = obs.unsqueeze(0)

        obs = torch.Tensor(obs).to(self.device)
        q_values = self.critic_model(obs)
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        return actions

    def learn(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        terminal: torch.Tensor,
    ) -> dict[str, Union[float, int]]:
        """DQN learner.

        Args:
            obs (torch.Tensor): Current observation.
            action (torch.Tensor): Action taken.
            reward (torch.Tensor): Reward received.
            next_obs (torch.Tensor): Next observation.
            terminal (torch.Tensor): Whether the episode terminated.

        Returns:
            dict[str, Union[float, int]]: Information about the learning process.
        """
        action = action.to(self.device, dtype=torch.long)
        # Prediction Q(s)
        pred_value = self.actor_model(obs).gather(1, action)
        # Target for Q regression
        with torch.no_grad():
            next_q_value = self.actor_target(next_obs).max(1, keepdim=True)[0]
        # TD target
        target = reward + (1 - terminal) * self.config.gamma * next_q_value
        # TD loss
        loss = F.mse_loss(pred_value, target)
        # Set the gradients to zero
        self.actor_optimizer.zero_grad()
        loss.backward()
        # Backward propagation to update parameters
        self.actor_optimizer.step()

        info = {'loss': loss.item(), 'q_values': pred_value.mean().item()}
        return info

    def train(self) -> None:
        """Train the agent."""
        obs, _ = self.envs.reset(seed=self.config.seed)
        self.text_logger.info('Start Training')
        for global_step in range(self.config.max_train_steps):
            self.global_step = global_step
            self.eps_greedy = self.eps_greedy_scheduler.step()
            actions = self.get_action(obs)

            next_obs, rewards, terminations, truncations, infos = self.envs.step(
                actions)
            # Record rewards for plotting purposes
            if 'final_info' in infos:
                for info in infos['final_info']:
                    if info and 'episode' in info:
                        env_info = dict(
                            episodic_return=info['episode']['r'],
                            episodic_length=info['episode']['l'],
                        )
                        self.log_train_infos(env_info, global_step)

            # Save data to replay buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos['final_observation'][idx]
            self.buffer.add(obs, real_next_obs, actions, rewards, terminations,
                            infos)

            # Crucial step easy to overlook
            obs = next_obs
            # Training logic
            if global_step > self.config.warmup_learn_steps:
                if global_step % self.config.train_frequency == 0:
                    self.progress_bar.update(self.config.train_frequency)
                    data = self.buffer.sample(self.config.batch_size)
                    # Learner: Update model parameters
                    step_info = self.learn(
                        data.observations,
                        data.actions,
                        data.rewards,
                        data.next_observations,
                        data.dones,
                    )
                    step_info['learning_rate'] = self.learning_rate
                    step_info['eps_greedy'] = self.eps_greedy
                    # Log training information
                    train_fps = int(global_step /
                                    (time.time() - self.start_time))
                    log_message = (
                        '[Train],  global_step: {}, train_fps: {}, loss: {:.2f}'
                        .format(global_step, train_fps, step_info['loss']))
                    self.text_logger.info(log_message)
                    self.log_train_infos(step_info, global_step)

                # Update target network
                if global_step % self.config.target_update_frequency == 0:
                    self.text_logger.info('Update Target Model')
                    soft_target_update(
                        src_model=self.actor_target,
                        tgt_model=self.actor_model,
                        tau=self.config.soft_update_tau,
                    )
                # Evaluate model
                if global_step % self.config.eval_frequency:
                    eval_infos = self.evaluate(
                        eval_envs=self.eval_envs,
                        eval_episodes=self.config.eval_episodes,
                        save_video_dir=self.video_save_dir,
                    )
                    self.log_test_infos(eval_infos, global_step)

        # Save model
        if self.config.save_model:
            self.save_model(self.model_save_dir, global_step)

    def evaluate(
        self,
        eval_envs: Union[None],
        eval_episodes: int = 100,
        save_video_dir: str = None,
    ) -> List[float]:
        obs, _ = eval_envs.reset()
        episodic_returns = []
        episodic_steps = []
        for _ in range(eval_episodes):
            actions = self.get_action(obs)
            next_obs, _, _, _, infos = eval_envs.step(actions)
            if 'final_info' in infos:
                for info in infos['final_info']:
                    if 'episode' not in info:
                        continue
                    env_info = dict(
                        episodic_return=info['episode']['r'],
                        episodic_length=info['episode']['l'],
                    )
                    self.log_test_infos(env_info, self.global_step)
                    episodic_returns.append(info['episode']['r'])
                    episodic_steps.append(info['episode']['l'])
            obs = next_obs

        mean_return = np.mean(episodic_returns)
        mean_step = np.mean(episodic_steps)
        return dict(mean_episodic_return=mean_return,
                    mean_episodic_step=mean_step)
