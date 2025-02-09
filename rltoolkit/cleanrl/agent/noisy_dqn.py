import copy
from typing import Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from rltoolkit.cleanrl.agent.base import BaseAgent
from rltoolkit.cleanrl.rl_args import DQNArguments
from rltoolkit.cleanrl.utils.qlearing_net import NoisyNet
from rltoolkit.data.utils.type_aliases import ReplayBufferSamples
from rltoolkit.utils import LinearDecayScheduler, soft_target_update


class NoisyDQNAgent(BaseAgent):
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
        args: DQNArguments,
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
        self.qnet = NoisyNet(
            self.obs_dim,
            self.action_dim,
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
        """Get action from the Q-network.

        Args:
            obs (np.ndarray): The current observation.

        Returns:
            np.ndarray: Selected action.
        """
        return self.predict(obs)

    def predict(self, obs: np.ndarray) -> int:
        """Predict an action given the observation using the Q-network.

        Args:
            obs (np.ndarray): The observation.

        Returns:
            int: The selected action.
        """
        # Ensure observation is in the correct shape
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.qnet(obs_tensor).argmax().item()

        return action

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

        action = action.to(self.device, dtype=torch.long)

        # Compute current Q values
        current_q_values = self.qnet(obs).gather(1, action)
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_qnet(next_obs).max(dim=1,
                                                           keepdim=True)[0]

        target_q_values = (
            reward +
            (1 - done) * self.args.gamma**self.args.n_steps * next_q_values)

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values, reduction='mean')

        # Optimize the model
        self.optimizer.zero_grad()

        if self.args.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(),
                                           self.args.max_grad_norm)
        loss.backward()
        self.optimizer.step()

        # NoisyNet: reset noise
        self.qnet.reset_noise()
        self.target_qnet.reset_noise()

        # Soft update target network
        if self.learner_update_step % self.args.target_update_frequency == 0:
            soft_target_update(self.qnet, self.target_qnet,
                               self.args.soft_update_tau)
            self.target_model_update_step += 1
        self.learner_update_step += 1

        learn_result = {
            'loss': loss.item(),
        }
        return learn_result
