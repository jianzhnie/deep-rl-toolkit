import copy
from typing import Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from rltoolkit.cleanrl.agent.base import BaseAgent
from rltoolkit.cleanrl.rl_args import SACArguments
from rltoolkit.cleanrl.utils.ac_net import ActorNet, CriticNet
from rltoolkit.data.utils.type_aliases import ReplayBufferSamples
from rltoolkit.utils import soft_target_update
from torch.distributions import Categorical


class SACAgent(BaseAgent):
    """Agent interacting with environment using Soft Actor-Critic (SAC)
    algorithm.

    The "Critic" estimates the value function. This could be the action-value
    (the Q value) or state-value (the V value). The "Actor" updates the policy
    distribution in the direction suggested by the Critic (such as with policy
    gradients).
    """

    def __init__(
        self,
        args: SACArguments,
        env: gym.Env,
        state_shape: Union[int, List[int]],
        action_shape: Union[int, List[int]],
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize the SAC agent.

        Args:
            args (SACArguments): Hyperparameters for SAC.
            env (gym.Env): Environment to interact with.
            state_shape (Union[int, List[int]]): Shape of the state space.
            action_shape (Union[int, List[int]]): Shape of the action space.
            device (str, optional): Device to run the computations on. Defaults to 'cpu'.
        """
        super().__init__(args)
        self.args = args
        self.env = env
        self.device = device
        self.obs_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))
        self.learner_update_step = 0
        self.target_model_update_step = 0

        # Policy Network
        self.actor = ActorNet(self.obs_dim, self.args.hidden_dim,
                              self.action_dim).to(device)
        # Value Networks
        self.critic1 = CriticNet(self.obs_dim, self.args.hidden_dim,
                                 self.action_dim).to(device)
        self.critic2 = CriticNet(self.obs_dim, self.args.hidden_dim,
                                 self.action_dim).to(device)

        # Target Value Networks
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.args.actor_lr,
                                                eps=self.args.epsilon)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(),
                                                  lr=self.args.critic_lr,
                                                  eps=self.args.epsilon)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(),
                                                  lr=self.args.critic_lr,
                                                  eps=self.args.epsilon)

        # Temperature parameter (alpha) for entropy term
        if self.args.autotune:
            self.target_entropy = -self.args.target_entropy * torch.log(
                1 / torch.tensor(self.action_dim))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=self.args.alpha_lr,
                                                    eps=self.args.epsilon)
        else:
            self.alpha = self.args.alpha

    def get_action(self, obs: np.ndarray) -> int:
        """Select an action from the input state with exploration.

        Args:
            obs (np.ndarray): Observation from the environment.

        Returns:
            int: Selected action.
        """
        if self.learner_update_step < self.args.warmup_learn_steps:
            action = self.env.action_space.sample()
        else:
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(
                self.device)
            logits = self.actor(obs_tensor)
            action_dist = Categorical(logits=logits)
            action = action_dist.sample()
        return action.item()

    def predict(self, obs: np.ndarray) -> int:
        """Predict an action from the input state without exploration (greedy).

        Args:
            obs (np.ndarray): Observation from the environment.

        Returns:
            int: Predicted action.
        """
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(obs_tensor)
            action_dist = Categorical(logits=logits)
            action = action_dist.probs.argmax(dim=1, keepdim=True)
        return action.item()

    def calc_q_target(
        self,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the target Q-value for the next state.

        Args:
            next_obs (torch.Tensor): Next state observations.
            reward (torch.Tensor): Reward received.
            done (torch.Tensor): Done flag indicating episode termination.

        Returns:
            torch.Tensor: Target Q-value.
        """
        next_logits = self.actor(next_obs)
        next_action_dist = Categorical(logits=next_logits)
        next_log_pi = F.log_softmax(next_logits, dim=1)
        next_action_prob = next_action_dist.probs

        q1_next_target = self.target_critic1(next_obs)
        q2_next_target = self.target_critic2(next_obs)
        min_q_target = (
            next_action_prob * torch.min(q1_next_target, q2_next_target) -
            self.alpha * next_log_pi)
        min_q_target = torch.sum(min_q_target, dim=1, keepdim=True)
        next_q_value = reward + self.args.gamma * (1 - done) * min_q_target
        return next_q_value

    def learn(self, batch: ReplayBufferSamples) -> Dict[str, float]:
        """Update the model by TD actor-critic.

        Args:
            batch (ReplayBufferSamples): Batch of samples from the replay buffer.

        Returns:
            Dict[str, float]: Dictionary of loss values.
        """
        obs = batch.obs
        next_obs = batch.next_obs
        action = batch.actions.long()
        reward = batch.rewards
        done = batch.dones

        # Update Critic Networks
        with torch.no_grad():
            next_q_target = self.calc_q_target(next_obs, reward, done)

        q1_action_value = self.critic1(obs).gather(1, action)
        critic1_loss = F.mse_loss(q1_action_value, next_q_target)
        q2_action_value = self.critic2(obs).gather(1, action)
        critic2_loss = F.mse_loss(q2_action_value, next_q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update Actor Network
        logits = self.actor(obs)
        action_dist = Categorical(logits=logits)
        log_pi = F.log_softmax(logits, dim=1)
        action_prob = action_dist.probs

        with torch.no_grad():
            q1_value = self.critic1(obs)
            q2_value = self.critic2(obs)
            min_qvalue = torch.min(q1_value, q2_value)

        actor_loss = torch.mean(action_prob *
                                (self.alpha * log_pi - min_qvalue))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Alpha
        if self.args.autotune:
            alpha_loss = torch.mean(action_prob.detach() *
                                    (-self.log_alpha.exp() *
                                     (log_pi + self.target_entropy).detach()))
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # Soft Update of Target Networks
        if self.learner_update_step % self.args.target_update_frequency == 0:
            soft_target_update(self.critic1,
                               self.target_critic1,
                               tau=self.args.soft_update_tau)
            soft_target_update(self.critic2,
                               self.target_critic2,
                               tau=self.args.soft_update_tau)
            self.target_model_update_step += 1

        self.learner_update_step += 1

        result = {
            'alpha': self.alpha,
            'alpha_loss': alpha_loss.item(),
            'actor_loss': actor_loss.item(),
            'q1_value': q1_action_value.mean().item(),
            'q2_value': q2_action_value.mean().item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
        }
        return result
