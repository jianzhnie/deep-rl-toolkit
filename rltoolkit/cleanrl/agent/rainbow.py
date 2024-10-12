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
        self.prior_eps = args.prior_eps

        # Categorical DQN parameters
        self.v_min = args.v_min
        self.v_max = args.v_max
        self.num_atoms = args.num_atoms
        self.noisy_std = args.noisy_std
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

        # Soft update target network
        if self.learner_update_step % self.args.target_update_frequency == 0:
            soft_target_update(self.qnet, self.target_qnet,
                               self.args.soft_update_tau)
            self.target_model_update_step += 1
        self.learner_update_step += 1

        # N-step Learning loss
        elementwise_loss = self._compute_dqn_loss(batch)
        print('elementwise_loss', elementwise_loss.shape)
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # Optimize the model
        self.optimizer.zero_grad()

        if self.args.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(),
                                           self.args.max_grad_norm)
        loss.backward()
        self.optimizer.step()
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
            next_action = self.qnet(next_obs).max(dim=1, keepdim=True)[1]
            next_dist = self.target_qnet.get_prob_dist(next_obs)
            next_dist = next_dist[range(self.args.batch_size), next_action, :]
            print(next_action.shape, next_dist.shape)
            if self.args.n_steps > 1:
                gamma = self.args.gamma**self.args.n_steps
            else:
                gamma = self.args.gamma
            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (torch.linspace(
                0, (self.args.batch_size - 1) * self.num_atoms,
                self.args.batch_size).long().unsqueeze(1).expand(
                    self.args.batch_size, self.num_atoms).to(self.device))

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1),
                                          (next_dist *
                                           (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1),
                                          (next_dist *
                                           (b - l.float())).view(-1))

        dist = self.qnet.get_prob_dist(obs)
        dist = dist[range(self.args.batch_size), action, :]
        log_p = torch.log(dist + 1e-8)
        print(
            'next_dist',
            next_dist.shape,
            'prob_dist',
            proj_dist.shape,
            'dist',
            dist.shape,
            'log_p',
            log_p.shape,
        )
        elementwise_loss = -(proj_dist * log_p).sum(dim=1)
        return elementwise_loss
