import copy
from typing import Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from rltoolkit.cleanrl.agent.base import BaseAgent
from rltoolkit.cleanrl.rl_args import PERArguments
from rltoolkit.cleanrl.utils.qlearing_net import DuelingNet, QNet
from rltoolkit.data.utils.type_aliases import PrioritizedReplayBufferSamples
from rltoolkit.utils import LinearDecayScheduler, soft_target_update


class PERAgent(BaseAgent):
    """Deep Q-Network algorithm.

    "Human-Level Control Through Deep Reinforcement Learning" - Mnih V. et al., 2015.

    Args:
        args (DQNArguments): Configuration object for the agent.
        env (gym.Env): Environment object.
        state_shape (Optional[Union[int, List[int]]]): Shape of the state.
        action_shape (Optional[Union[int, List[int]]]): Shape of the action.
        device (Optional[Union[str, torch.device]]): Device to use for computation.
    """

    def __init__(
        self,
        args: PERArguments,
        env: gym.Env,
        state_shape: Union[int, List[int]] = None,
        action_shape: Union[int, List[int]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(args)
        assert (isinstance(args.n_steps, int) and args.n_steps > 0
                ), 'N-step should be an integer and greater than 0.'
        self.args = args
        self.env = env
        self.device = device
        self.learner_update_step = 0
        self.target_model_update_step = 0
        self.eps_greedy = args.eps_greedy_start
        self.learning_rate = args.learning_rate
        self.obs_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))

        # Initialize networks
        if args.dueling_dqn:
            self.qnet = DuelingNet(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=self.args.hidden_dim,
            ).to(device)
        else:
            self.qnet = QNet(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=self.args.hidden_dim,
            ).to(device)
        self.target_qnet = copy.deepcopy(self.qnet)
        self.target_qnet.eval()
        # Initialize optimizer and schedulers
        self.optimizer = torch.optim.Adam(params=self.qnet.parameters(),
                                          lr=args.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer,
            start_factor=args.learning_rate,
            end_factor=args.min_learning_rate,
            total_iters=args.max_timesteps,
        )
        self.eps_greedy_scheduler = LinearDecayScheduler(
            args.eps_greedy_start,
            args.eps_greedy_end,
            max_steps=args.max_timesteps * 0.8,
        )

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from the actor network.

        Args:
            obs (np.ndarray): Current observation.

        Returns:
            np.ndarray: Selected action.
        """
        # epsilon greedy policy
        if np.random.rand() <= self.eps_greedy:
            action = self.env.action_space.sample()
        else:
            action = self.predict(obs)

        self.eps_greedy = max(self.eps_greedy_scheduler.step(),
                              self.args.eps_greedy_end)
        return action

    def predict(self, obs: np.ndarray) -> int:
        """Predict an action given an observation.

        Args:
            obs (np.ndarray): Current observation.

        Returns:
            int: Selected action.
        """
        if obs.ndim == 1:
            # Expand to have batch_size = 1
            obs = np.expand_dims(obs, axis=0)

        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        q_values = self.qnet(obs)
        action = torch.argmax(q_values, dim=1).item()
        return action

    def learn(self, batch: PrioritizedReplayBufferSamples) -> Dict[str, float]:
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
        indices = batch.indices
        weights = torch.tensor(batch.weights,
                               dtype=torch.float,
                               device=self.device).reshape(-1, 1)

        action = action.to(self.device, dtype=torch.long)

        # Compute current Q values
        current_q_values = self.qnet(obs).gather(1, action)
        # Compute target Q values
        if self.args.double_dqn:
            with torch.no_grad():
                greedy_action = self.qnet(next_obs).max(dim=1, keepdim=True)[1]
                next_q_values = self.target_qnet(next_obs).gather(
                    1, greedy_action)
        else:
            with torch.no_grad():
                next_q_values = self.target_qnet(next_obs).max(dim=1,
                                                               keepdim=True)[0]

        target_q_values = (
            reward +
            (1 - done) * self.args.gamma**self.args.n_steps * next_q_values)

        td_error = torch.abs(current_q_values - target_q_values)
        # Compute loss
        elementwise_loss = F.mse_loss(current_q_values,
                                      target_q_values,
                                      reduction='none')
        loss = torch.mean(elementwise_loss * weights)
        # Optimize the model
        self.optimizer.zero_grad()

        if self.args.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(),
                                           self.args.max_grad_norm)
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        if self.learner_update_step % self.args.target_update_frequency == 0:
            soft_target_update(self.qnet, self.target_qnet,
                               self.args.soft_update_tau)
            self.target_model_update_step += 1
        self.learner_update_step += 1

        new_priorities = td_error.detach().cpu().numpy() + self.args.prior_eps
        learn_result = {'loss': loss.item()}
        return (learn_result, indices, new_priorities)
