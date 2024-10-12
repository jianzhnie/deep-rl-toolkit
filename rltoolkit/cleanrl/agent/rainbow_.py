import copy
from typing import Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from rltoolkit.cleanrl.agent.base import BaseAgent
from rltoolkit.cleanrl.rl_args import RainbowArguments
from rltoolkit.cleanrl.utils.qlearing_net import RainbowNet
from rltoolkit.data.utils.type_aliases import PrioritizedReplayBufferSamples
from rltoolkit.utils import LinearDecayScheduler, soft_target_update


class RainbowAgent(BaseAgent):
    """Implementation of the Rainbow DQN algorithm.

    Args:
        args (RainbowArguments): Configuration object for the agent.
        env (gym.Env): Environment object.
        state_shape (Union[int, List[int]]): Shape of the state space.
        action_shape (Union[int, List[int]]): Shape of the action space.
        device (Optional[Union[str, torch.device]]): Device to use for computation ('cpu' or 'cuda').
    """

    def __init__(
        self,
        args: RainbowArguments,
        env: gym.Env,
        state_shape: Union[int, List[int]],
        action_shape: Union[int, List[int]],
        device: Optional[Union[str, torch.device]] = 'cpu',
    ) -> None:
        super().__init__(args)

        # Validate arguments
        assert args.n_steps > 0, 'N-step should be an integer and greater than 0.'
        assert args.learning_rate > 0, 'Learning rate must be greater than zero.'
        assert args.learn_steps >= 1, 'Learn step must be greater than or equal to one.'
        assert (
            args.v_max >= args.v_min
        ), 'Max support value must be greater than or equal to min support.'

        # Store environment and device details
        self.args = args
        self.env = env
        self.device = torch.device(device)

        self.obs_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))

        # Hyperparameters
        self.learner_update_step = 0
        self.target_model_update_step = 0
        self.eps_greedy = args.eps_greedy_start
        self.learning_rate = args.learning_rate

        # Categorical DQN parameters
        self.v_min = args.v_min
        self.v_max = args.v_max
        self.num_atoms = args.num_atoms
        self.noisy_std = args.noisy_std
        self.prior_eps = args.prior_eps

        # Categorical support
        self.support = torch.nn.Parameter(torch.linspace(
            self.v_min, self.v_max, self.num_atoms),
                                          requires_grad=False)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        # Initialize networks
        self.qnet: RainbowNet = RainbowNet(
            obs_dim=self.obs_dim,
            hidden_dim=self.args.hidden_dim,
            action_dim=self.action_dim,
            num_atoms=self.num_atoms,
            support=self.support,
            noisy_std=self.noisy_std,
        ).to(self.device)

        self.target_qnet = copy.deepcopy(self.qnet)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.target_qnet.eval()

        # Initialize optimizer and schedulers
        self.optimizer = torch.optim.Adam(self.qnet.parameters(),
                                          lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer,
            start_factor=self.learning_rate,
            end_factor=self.args.min_learning_rate,
            total_iters=self.args.max_timesteps,
        )
        self.eps_greedy_scheduler = LinearDecayScheduler(
            start=self.args.eps_greedy_start,
            end=self.args.eps_greedy_end,
            max_steps=int(self.args.max_timesteps * 0.8),
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

    def learn(self, batch: PrioritizedReplayBufferSamples) -> Dict[str, float]:
        """Perform a learning step, update the Q-network.

        Args:
            batch (PrioritizedReplayBufferSamples): Batch of experience samples.

        Returns:
            Dict[str, float]: The loss value and updated priorities.
        """
        indices = batch.indices
        weights = torch.tensor(batch.weights,
                               dtype=torch.float32,
                               device=self.device).view(-1, 1)

        # Compute loss
        elementwise_loss = self._compute_dqn_loss(batch)
        loss = torch.mean(elementwise_loss * weights)

        # Optimize the model
        self.optimizer.zero_grad()
        if self.args.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(),
                                           self.args.max_grad_norm)
        loss.backward()
        self.optimizer.step()

        # Reset noise for next step
        self.qnet.reset_noise()
        self.target_qnet.reset_noise()

        # Soft update target network
        if self.learner_update_step % self.args.target_update_frequency == 0:
            soft_target_update(self.qnet, self.target_qnet,
                               self.args.soft_update_tau)
            self.target_model_update_step += 1

        self.learner_update_step += 1

        # Update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps

        return {'loss': loss.item()}, indices, new_priorities

    def _compute_dqn_loss(
            self, batch: PrioritizedReplayBufferSamples) -> torch.Tensor:
        """Compute the distributional DQN loss.

        Args:
            batch (PrioritizedReplayBufferSamples): Batch of experience samples.

        Returns:
            torch.Tensor: Element-wise loss for each experience in the batch.
        """
        obs = batch.obs.to(self.device)
        next_obs = batch.next_obs.to(self.device)
        action = batch.actions.to(self.device, dtype=torch.long).squeeze()
        reward = batch.rewards.to(self.device)
        done = batch.dones.to(self.device)

        # Compute target distribution
        with torch.no_grad():
            next_action = self.qnet(next_obs).argmax(dim=1)
            target_dist = self.target_qnet(next_obs, return_qval=False)
            target_dist = target_dist[range(self.args.batch_size), next_action]

            gamma = (self.args.gamma**self.args.n_steps
                     if self.args.n_steps > 1 else self.args.gamma)
            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(self.v_min, self.v_max)

            # Compute projection of distribution
            b = (t_z - self.v_min) / self.delta_z
            l, u = b.floor().long(), b.ceil().long()
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1

            offset = (torch.linspace(
                0, (self.args.batch_size - 1) * self.num_atoms,
                self.args.batch_size).long().unsqueeze(1).expand(
                    self.args.batch_size, self.num_atoms).to(self.device))

            proj_dist = torch.zeros_like(target_dist)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1),
                                          (target_dist *
                                           (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1),
                                          (target_dist *
                                           (b - l.float())).view(-1))

        # Compute current distribution
        curr_q_dist = self.qnet(obs, return_qval=False)
        log_p = torch.log(curr_q_dist + 1e-8)
        log_p = log_p[range(self.args.batch_size), action]

        # Loss is the negative log-likelihood of the projected distribution
        elementwise_loss = -(proj_dist * log_p).sum(dim=1)

        return elementwise_loss
