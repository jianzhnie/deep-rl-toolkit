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
    """Rainbow algorithm.

    Args:
        args (DQNArguments): Configuration object for the agent.
        env (gym.Env): Environment object.
        state_shape (Optional[Union[int, List[int]]]): Shape of the state.
        action_shape (Optional[Union[int, List[int]]]): Shape of the action.
        device (Optional[Union[str, torch.device]]): Device to use for computation.
    """

    def __init__(
        self,
        args: RainbowArguments,
        env: gym.Env,
        state_shape: Union[int, List[int]] = None,
        action_shape: Union[int, List[int]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(args)
        assert (isinstance(args.n_steps, int) and args.n_steps > 0
                ), 'N-step should be an integer and greater than 0.'

        assert isinstance(args.learning_rate,
                          float), 'Learning rate must be a float.'
        assert args.learning_rate > 0, 'Learning rate must be greater than zero.'
        assert isinstance(args.learn_steps,
                          int), 'Learn step rate must be an integer.'
        assert args.learn_steps >= 1, 'Learn step must be greater than or equal to one.'
        assert isinstance(args.gamma, (float, int)), 'Gamma must be a float.'
        assert isinstance(args.soft_update_tau, float), 'Tau must be a float.'
        assert args.soft_update_tau > 0, 'Tau must be greater than zero.'
        assert isinstance(
            args.prior_eps,
            float), 'Minimum priority for sampling must be a float.'
        assert (args.prior_eps >
                0), 'Minimum priority for sampling must be greater than zero.'
        assert isinstance(args.num_atoms,
                          int), 'Number of atoms must be an integer.'
        assert (args.num_atoms >=
                1), 'Number of atoms must be greater than or equal to one.'
        assert isinstance(
            args.v_min,
            (float, int)), 'Minimum value of support must be a float.'
        assert isinstance(
            args.v_max,
            (float, int)), 'Maximum value of support must be a float.'
        assert (
            args.v_max >= args.v_min
        ), 'Maximum value of support must be greater than or equal to minimum value.'
        assert isinstance(args.n_steps, int), 'Step number must be an integer.'
        assert args.n_steps >= 1, 'Step number must be greater than or equal to one.'

        self.args = args
        self.env = env
        self.device = device
        self.obs_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))

        # Hparams
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

        self.support = torch.nn.Parameter(
            torch.linspace(self.v_min, self.v_max, self.num_atoms),
            requires_grad=False,
        )
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        # Initialize networks
        self.qnet: RainbowNet = RainbowNet(
            obs_dim=self.obs_dim,
            hidden_dim=self.args.hidden_dim,
            action_dim=self.action_dim,
            num_atoms=self.args.num_atoms,
            support=self.support,
            noisy_std=self.noisy_std,
        ).to(self.device)
        self.target_qnet = copy.deepcopy(self.qnet)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
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
        action = self.predict(obs)
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
        with torch.no_grad():
            action = self.qnet(obs).argmax().item()
        return action

    def learn(self, batch: PrioritizedReplayBufferSamples) -> Dict[str, float]:
        """Perform a learning step.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of experience.

        Returns:
            float: Loss value.
        """
        indices = batch.indices
        weights = torch.tensor(batch.weights,
                               dtype=torch.float,
                               device=self.device).reshape(-1, 1)
        # N-step Learning loss
        elementwise_loss = self._compute_dqn_loss(batch)
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # Optimize the model
        self.optimizer.zero_grad()

        if self.args.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(),
                                           self.args.max_grad_norm)
        loss.backward()
        self.optimizer.step()
        self.qnet.reset_noise()
        self.target_qnet.reset_noise()

        # Soft update target network
        if self.learner_update_step % self.args.target_update_frequency == 0:
            soft_target_update(self.qnet, self.target_qnet,
                               self.args.soft_update_tau)
            self.target_model_update_step += 1
        self.learner_update_step += 1

        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        learn_result = {'loss': loss.item()}
        return (learn_result, indices, new_priorities)

    def _compute_dqn_loss(
            self, batch: PrioritizedReplayBufferSamples) -> Dict[str, float]:
        obs = batch.obs
        next_obs = batch.next_obs
        action = batch.actions
        reward = batch.rewards
        done = batch.dones

        action = action.to(self.device, dtype=torch.long)

        with torch.no_grad():
            # Double DQN
            # Predict next actions from next_states
            next_action = self.qnet.forward(next_obs).argmax(dim=1)
            # Predict the target q distribution for the same next states
            target_dist = self.target_qnet.forward(next_obs, return_qval=False)
            # Index the target q_dist to select the distributions corresponding to next_actions
            target_dist = target_dist[range(self.args.batch_size), next_action]

            if self.args.n_steps > 1:
                gamma = self.args.gamma**self.args.n_steps
            else:
                gamma = self.args.gamma
            # Determine the target z values

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)

            # Finds closest support element index value
            b = (t_z - self.v_min) / self.delta_z

            # Find the neighbouring indices of b
            l = b.floor().long()
            u = b.ceil().long()

            # Shape of projected q distribution is (batch_size, num_atoms) as we have argmaxed over actions
            # Fix disappearing probability mass
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1
            offset = (torch.linspace(
                0, (self.args.batch_size - 1) * self.num_atoms,
                self.args.batch_size).long().unsqueeze(1).expand(
                    self.args.batch_size, self.num_atoms).to(self.device))

            proj_dist = torch.zeros(target_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1),
                                          (target_dist *
                                           (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1),
                                          (target_dist *
                                           (b - l.float())).view(-1))

        # Calculate the current state
        curr_q_dist = self.qnet.forward(obs, return_qval=False)
        log_p = torch.log(curr_q_dist + 1e-8)
        log_p = log_p[range(self.args.batch_size), action.squeeze()]

        # loss
        elementwise_loss = -(proj_dist * log_p).sum(dim=1)
        return elementwise_loss
