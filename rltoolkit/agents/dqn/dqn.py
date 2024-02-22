from argparse import Namespace
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rltoolkit.agents.base_agent import BaseAgent as AgentBase
from rltoolkit.buffers import BaseBuffer
from rltoolkit.utils import hard_target_update, soft_target_update
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
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(
            config,
            envs,
            buffer,
            actor_model,
            actor_optimizer,
            actor_lr_scheduler,
            device,
        )
        if self.is_soft_tgt_update:
            self.tgt_update_fn = soft_target_update
        else:
            self.tgt_update_fn = hard_target_update

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

        if self.global_update_step % self.update_target_step == 0:
            self.tgt_update_fn(self.actor_model, self.actor_target)

        action = action.to(self.device, dtype=torch.long)
        # Prediction Q(s)
        pred_value = self.actor_model(obs).gather(1, action)
        # Target for Q regression
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
        self.global_update_step += 1

        info = {
            'qloss': loss.item(),
            'learning_rate': self.learning_rate,
            'predictQ': pred_value.mean().item(),
        }
        return info

    # train an episode
    def run_train_episode(self, train_steps: int):
        episode_reward = 0
        episode_step = 0
        episode_loss = []
        obs = self.envs.reset()
        done = False
        while not done:
            episode_step += 1
            action = self.get_action(obs)
            next_obs, reward, done, _ = self.envs.step(action)
            self.buffer.add(obs, action, reward, next_obs, done)
            # train model
            if self.buffer.size() > self.warmup_buffer_size:
                # s,a,r,s',done
                samples = self.buffer.sample_batch()

                batch_obs = samples['obs']
                batch_action = samples['action']
                batch_reward = samples['reward']
                batch_next_obs = samples['next_obs']
                batch_terminal = samples['terminal']

                train_loss = self.learn(
                    batch_obs,
                    batch_action,
                    batch_reward,
                    batch_next_obs,
                    batch_terminal,
                )
                episode_loss.append(train_loss)
            episode_reward += reward
            obs = next_obs
        return episode_reward, episode_step, np.mean(episode_loss)
