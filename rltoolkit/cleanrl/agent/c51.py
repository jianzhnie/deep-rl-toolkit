import copy
from typing import Dict, List, Union

import gymnasium as gym
import numpy as np
import torch
from rltoolkit.cleanrl.agent.base import BaseAgent
from rltoolkit.cleanrl.rl_args import C51Arguments
from rltoolkit.cleanrl.utils.discrete_action import C51Network
from rltoolkit.utils import LinearDecayScheduler, soft_target_update


class C51Agent(BaseAgent):
    """C51 Agent for reinforcement learning.

    Args:
        args (C51Arguments): Configuration arguments for the C51 agent.
        env (gym.Env): The environment in which the agent will operate.
        state_shape (Union[int, List[int]]): The shape of the state space.
        action_shape (Union[int, List[int]]): The shape of the action space.
        device (str, optional): The device to use for computation ('cpu' or 'cuda'). Defaults to 'cpu'.
    """

    def __init__(
        self,
        args: C51Arguments,
        env: gym.Env,
        state_shape: Union[int, List[int]],
        action_shape: Union[int, List[int]],
        device: str = 'cpu',
    ) -> None:
        super().__init__(args)
        self.args = args
        self.env = env
        self.device = device
        self.global_update_step = 0
        self.target_model_update_step = 0
        self.eps_greedy = args.eps_greedy_start
        self.learning_rate = args.learning_rate
        self.obs_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))

        # Initialize the main and target networks
        self.qnet = C51Network(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
        ).to(device)

        self.target_qnet = copy.deepcopy(self.qnet)
        self.optimizer = torch.optim.Adam(
            self.qnet.parameters(),
            lr=args.learning_rate,
            eps=0.01 / args.batch_size,
        )

        # Initialize epsilon-greedy scheduler
        self.eps_greedy_scheduler = LinearDecayScheduler(
            args.eps_greedy_start,
            args.eps_greedy_end,
            max_steps=args.max_timesteps * 0.8,
        )

    def get_action(self, obs: np.ndarray) -> int:
        """Sample an action based on the current observation and epsilon value.

        Args:
            obs (np.ndarray): Current observation.

        Returns:
            int: Selected action.
        """
        if np.random.rand() <= self.eps_greedy:
            act = self.env.action_space.sample()
        else:
            act = self.predict(obs)

        self.eps_greedy = max(self.eps_greedy_scheduler.step(),
                              self.args.eps_greedy_end)
        return act

    def predict(self, obs: np.ndarray) -> int:
        """Predict an action based on the current observation using the
        Q-network.

        Args:
            obs (np.ndarray): Current observation.

        Returns:
            int: Predicted action.
        """
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)

        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        action, _ = self.qnet(obs)
        return action.item()

    def learn(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update the model based on a batch of experience.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of experience.

        Returns:
            float: Loss value.
        """
        obs = batch['obs']
        next_obs = batch['next_obs']
        action = batch['action']
        reward = batch['reward']
        done = batch['done']

        if self.global_update_step % self.args.target_update_frequency == 0:
            soft_target_update(self.qnet,
                               self.target_qnet,
                               tau=self.args.soft_update_tau)
            self.target_model_update_step += 1

        self.global_update_step += 1
        action = action.to(dtype=torch.long)

        with torch.no_grad():
            _, next_pmfs = self.target_qnet(next_obs)
            next_atoms = reward + self.args.gamma * self.target_qnet.atoms * (
                1 - done)

            delta_z = self.target_qnet.atoms[1] - self.target_qnet.atoms[0]
            tz = next_atoms.clamp(self.args.v_min, self.args.v_max)
            b = (tz - self.args.v_min) / delta_z
            l = b.floor().clamp(0, self.args.num_atoms - 1)
            u = b.ceil().clamp(0, self.args.num_atoms - 1)

            d_m_l = (u + (l == u).float() - b) * next_pmfs
            d_m_u = (b - l) * next_pmfs
            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        _, old_pmfs = self.qnet(obs, action)
        loss = torch.mean(
            -(target_pmfs *
              old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        result = {'loss': loss.item()}
        return result
