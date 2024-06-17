import copy
from typing import Dict, List, Union

import gym
import numpy as np
import torch
import torch.nn.functional as F
from rltoolkit.cleanrl.agent.base import BaseAgent
from rltoolkit.cleanrl.rl_args import DDPGArguments
from rltoolkit.cleanrl.utils.ac_net import OUNoise, PolicyNet, ValueNet
from rltoolkit.data.utils.type_aliases import ReplayBufferSamples
from rltoolkit.utils import soft_target_update


class DDPGAgent(BaseAgent):
    """DDPG Agent for interacting with the environment.

    Attributes:
        args (DDPGArguments): Arguments for DDPG agent.
        env (gym.Env): Environment to interact with.
        state_shape (Union[int, List[int]]): Shape of state space.
        action_shape (Union[int, List[int]]): Shape of action space.
        action_bound (float): Bound for action space.
        device (torch.device): Device (CPU/GPU).
    """

    def __init__(
        self,
        args: DDPGArguments,
        env: gym.Env,
        state_shape: Union[int, List[int]],
        action_shape: Union[int, List[int]],
        device: str = 'cpu',
    ) -> None:
        super().__init__(args)
        self.args = args
        self.env = env
        self.device = device
        self.action_bound = args.action_bound
        self.obs_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))
        self.action_high = self.env.action_space.high
        self.action_low = self.env.action_space.low
        self.learner_update_step = 0
        self.target_model_update_step = 0

        # Initialize Policy Network
        self.policy_net = PolicyNet(
            self.obs_dim,
            self.args.hidden_dim,
            self.action_dim,
            self.action_high,
            self.action_low,
        ).to(self.device)
        # Initialize Value Network
        self.critic_net = ValueNet(self.obs_dim, self.args.hidden_dim,
                                   self.action_dim).to(self.device)

        # Target Networks
        self.target_actor = copy.deepcopy(self.policy_net)
        self.target_critic = copy.deepcopy(self.critic_net)

        self.noiser = OUNoise(
            mu=0.0,
            sigma=self.args.ou_noise_sigma,
            theta=self.args.ou_noise_theta,
        )

        # loss function
        self.loss_fn = F.smooth_l1_loss if self.args.use_smooth_l1_loss else F.mse_loss
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                                lr=self.args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(),
                                                 lr=self.args.critic_lr)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Select action based on policy network.

        Args:
            obs (np.ndarray): Observation from environment.

        Returns:
            np.ndarray: Action selected by policy.
        """
        action = self.predict(obs)
        return action

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict action based on policy network (for evaluation).

        Args:
            obs (np.ndarray): Observation from environment.

        Returns:
            np.ndarray: Action selected by policy.
        """
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            action = self.policy_net(obs)
            action += torch.normal(
                0, self.policy_net.action_scale * self.args.exploration_noise)
            action = action.cpu().numpy().clip(self.action_low,
                                               self.action_high)
        return action.flatten()

    def learn(self, batch: ReplayBufferSamples) -> Dict[str, float]:
        """Perform a learning step.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of experience.

        Returns:
            float: Loss value.
        """
        obs = batch.obs
        next_obs = batch.next_obs
        action = batch.actions
        reward = batch.rewards
        done = batch.dones

        # Soft update target networks
        if self.learner_update_step % self.args.target_update_frequency == 0:
            soft_target_update(self.policy_net, self.target_actor,
                               self.args.soft_update_tau)
            soft_target_update(self.critic_net, self.target_critic,
                               self.args.soft_update_tau)
            self.target_model_update_step += 1

        self.learner_update_step += 1

        # Current Q values
        curr_q_values = self.critic_net(obs, action)
        with torch.no_grad():
            next_actions = self.target_actor(next_obs)
            next_q_values = self.target_critic(next_obs, next_actions)

        # Temporal difference target
        q_targets = reward + self.args.gamma * next_q_values * (1 - done)
        # Mean squared error loss
        value_loss = self.loss_fn(curr_q_values, q_targets)

        # Update value network
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        policy_loss = torch.tensor(0.0).to(self.device)
        if self.learner_update_step % self.args.policy_frequency == 0:
            # Calculate policy loss
            pred_action = self.policy_net(obs)
            policy_loss = -torch.mean(self.critic_net(obs, pred_action))

            # Update policy network
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

        result = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'loss': policy_loss.item() + value_loss.item(),
        }
        return result
