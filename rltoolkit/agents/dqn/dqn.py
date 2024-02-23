import time
from argparse import Namespace
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rltoolkit.agents.base_agent import BaseAgent as AgentBase
from rltoolkit.buffers import BaseBuffer
from rltoolkit.utils import soft_target_update
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class AgentDQN(AgentBase):
    """Deep Q-Network algorithm.

    “Human-Level Control Through Deep Reinforcement Learning”. Mnih V. et al..
    2015.
    """

    def __init__(
        self,
        config: Namespace,
        envs: Union[None],
        buffer: BaseBuffer,
        actor_model: nn.Module,
        critic_model: Optional[nn.Module] = None,
        actor_optimizer: Optimizer = None,
        critic_optimizer: Optional[Optimizer] = None,
        actor_lr_scheduler: Optional[LRScheduler] = None,
        critic_lr_scheduler: Optional[LRScheduler] = None,
        eps_greedy_scheduler: Optional[Union[str, List[float]]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(
            config,
            envs,
            buffer,
            actor_model,
            actor_optimizer,
            actor_lr_scheduler,
            eps_greedy_scheduler,
            device,
        )
        self.start_time = time.time()

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from the actor network.

        Sample an action when given an observation, base on the current
        epsilon value, either a greedy action or a random action will be
        returned.

        Args:
            obs: current observation

        Returns:
            act (int): action
        """
        # Choose a random action with probability epsilon
        if np.random.rand() <= self.eps_greedy:
            action = np.random.choice(self.action_space.n, self.num_envs)
        else:
            action = self.predict(obs)
        return action

    def predict(self, obs: torch.Tensor) -> Union[int, List[int]]:
        """Predict an action when given an observation, a greedy action will be
        returned.

        Args:
            obs (np.float32): shape of (batch_size, obs_dim) , current observation

        Returns:
            act(int): action
        """
        if obs.ndim == 1:
            # if obs is 1 dimensional, we need to expand it to have batch_size = 1
            obs = np.expand_dims(obs, axis=0)
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        action = self.critic_model(obs).argmax().item()
        return action

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
            obs (torch.Tensor): _description_
            action (torch.Tensor): _description_
            reward (torch.Tensor): _description_
            next_obs (torch.Tensor): _description_
            terminal (torch.Tensor): _description_

        Returns:
            dict[str, Any]: _description_
        """
        action = action.to(self.device, dtype=torch.long)
        # Prediction Q(s)
        pred_value = self.actor_model(obs).gather(1, action)
        # Target for Q regression
        with torch.no_grad():
            next_q_value = self.actor_target(next_obs).max(1, keepdim=True)[0]
        # TD target
        target = reward + (1 - terminal) * self.gamma * next_q_value
        # TD loss
        loss = F.mse_loss(pred_value, target)
        # Set the gradients to zero
        self.actor_optimizer.zero_grad()
        loss.backward()
        # Backward propagation to update parameters
        self.actor_optimizer.step()

        info = {
            'qloss': loss.item(),
            'q_values': pred_value.mean().item(),
            'learning_rate': self.learning_rate,
            'eps_greedy': self.eps_greedy,
        }
        return info

    def train(self) -> None:
        obs, _ = self.envs.reset(seed=self.config.seed)
        for global_step in range(self.max_steps):
            self.eps_greedy = self.eps_greedy_scheduler()
            actions = self.get_action(obs)

            next_obs, rewards, terminations, truncations, infos = self.envs.step(
                actions)
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if 'final_info' in infos:
                for info in infos['final_info']:
                    if info and 'episode' in info:
                        env_info = dict(
                            episodic_return=info['episode']['r'],
                            episodic_length=info['episode']['l'],
                        )
                        self.log_train_infos(env_info, global_step)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos['final_observation'][idx]
            self.buffer.add(obs, real_next_obs, actions, rewards, terminations,
                            infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            # ALGO LOGIC: training.
            if global_step > self.warmup_learn_steps:
                if global_step % self.train_frequency == 0:
                    data = self.buffer.sample(self.batch_size)
                    train_infos = self.learn(
                        data.observations,
                        data.actions,
                        data.rewards,
                        data.next_observations,
                        data.dones,
                    )
                    train_fps = int(global_step /
                                    (time.time() - self.start_time))
                    train_infos['train_fps'] = train_fps
                    self.log_train_infos(train_infos, global_step)

                if global_step % self.target_update_frequency == 0:
                    soft_target_update(
                        src_model=self.actor_target,
                        tgt_model=self.actor_model,
                        tau=self.soft_update_tau,
                    )

            if self.config.save_model:
                self.save_model(self.model_save_dir)
