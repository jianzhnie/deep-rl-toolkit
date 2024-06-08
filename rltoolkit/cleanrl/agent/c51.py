import copy
from typing import Dict, List, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from rltoolkit.cleanrl.agent.base import BaseAgent
from rltoolkit.cleanrl.rl_args import C51Argments
from rltoolkit.utils import LinearDecayScheduler, soft_target_update


class QNetwork(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_atoms: int = 101,
        v_min: float = -100,
        v_max: float = 100,
    ) -> None:
        super().__init__()
        self.args.num_atoms = num_atoms
        self.register_buffer('atoms',
                             torch.linspace(v_min, v_max, steps=num_atoms))

        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * num_atoms),
        )

    def forward(self, x: torch.Tensor, action: torch.Tensor = None):
        # x : batch_size * obs_dim
        # logits: batch_size * (action_dim * num_atoms)
        logits = self.network(x)
        # logits: batch_size * action_dim * num_atoms
        logits = logits.view(len(x), self.action_dim, self.args.num_atoms)
        # probability mass function for each action
        # pmfs: batch_size * action_dim * num_atoms
        pmfs = torch.softmax(logits, dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
            action = action.to(dtype=torch.long)
        return action, pmfs[torch.arange(len(x)), action]


class C51Agent(BaseAgent):
    """Agent.

    Args:
        action_dim (int): action space dimension
        total_step (int): total epsilon decay steps
        learning_rate (float): initial learning rate
        target_model_update_step (int): target network update frequency
    """

    def __init__(
        self,
        args: C51Argments,
        env: gym.Env,
        state_shape: Union[int, List[int]] = None,
        action_shape: Union[int, List[int]] = None,
        device='cpu',
    ):
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

        # Main network
        self.qnet = QNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
        ).to(device)
        # Target network
        self.target_qnet = copy.deepcopy(self.qnet)
        # Create an optimizer
        self.optimizer = torch.optim.Adam(self.qnet.parameters(),
                                          lr=args.learning_rate)

        self.eps_greedy_scheduler = LinearDecayScheduler(
            args.eps_greedy_start,
            args.eps_greedy_end,
            max_steps=args.max_timesteps * 0.8,
        )

    def get_action(self, obs) -> int:
        """Sample an action when given an observation, base on the current
        epsilon value, either a greedy action or a random action will be
        returned.

        Args:
            obs: current observation

        Returns:
            act (int): action
        """
        # Choose a random action with probability epsilon
        if np.random.rand() <= self.eps_greedy:
            act = np.random.randint(self.action_dim)
        else:
            # Choose the action with highest Q-value at the current state
            act = self.predict(obs)

        self.eps_greedy = max(self.eps_greedy_scheduler.step(),
                              self.args.eps_greedy_end)

        return act

    def predict(self, obs) -> int:
        """Predict an action when given an observation, a greedy action will be
        returned.

        Args:
            obs (np.float32): shape of (batch_size, obs_dim) , current observation

        Returns:
            act(int): action
        """
        obs = np.expand_dims(obs, axis=0)
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        action, _ = self.qnet(obs)
        action = action.item()
        return action

    def learn(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update model with an episode data.

        Returns:
            loss (float)
        """
        obs = batch['obs']
        next_obs = batch['next_obs']
        action = batch['action']
        reward = batch['reward']
        done = batch['done']

        if self.global_update_step % self.target_model_update_step == 0:
            soft_target_update(self.qnet,
                               self.target_qnet,
                               tau=self.args.soft_update_tau)

        action = action.to(dtype=torch.long)
        with torch.no_grad():
            _, next_pmfs = self.target_qnet(next_obs)
            next_atoms = reward + self.args.gamma * self.target_qnet.num_atoms * (
                1 - done)

            # projection
            delta_z = self.target_qnet.atoms[1] - self.target_qnet.atoms[0]
            tz = next_atoms.clamp(self.args.v_min, self.args.v_max)

            b = (tz - self.args.v_min) / delta_z
            l = b.floor().clamp(0, self.args.num_atoms - 1)
            u = b.ceil().clamp(0, self.args.num_atoms - 1)

            # (l == u).float() handles the case where bj is exactly an integer
            # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
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
        # 反向传播更新参数
        self.optimizer.step()
        self.global_update_step += 1
        return loss.item()
