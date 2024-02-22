from argparse import Namespace
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from rltoolkit.agents.base_agent import BaseAgent as AgentBase
from rltoolkit.buffers import BaseBuffer
from rltoolkit.buffers import OffPolicyBuffer as ReplayBuffer
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

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        """update network."""

        with torch.no_grad():
            states, actions, rewards, undones = buffer.add_item
            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self.get_cumulative_rewards(rewards=rewards,
                                                    undones=undones).reshape(
                                                        (-1, )),
            )
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            obj_actors += q_value.mean().item()
            self.optimizer_update(self.critic_optimizer, obj_critic)
            self.soft_update(self.critic_target, self.critic_model,
                             self.soft_update_tau)
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic_raw(
            self, buffer: ReplayBuffer,
            batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss of the network and predict Q values with.

        **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(
                batch_size)  # next_ss: next states
            next_qs = (self.critic_target(next_ss).max(
                dim=1, keepdim=True)[0].squeeze(1))  # next q_values
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.critic_model(states).gather(1,
                                                    actions.long()).squeeze(1)
        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, q_values

    def get_obj_critic_per(
            self, buffer: ReplayBuffer,
            batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss of the network and predict Q values with.

        **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            (
                states,
                actions,
                rewards,
                undones,
                next_ss,
                is_weights,
                is_indices,
            ) = buffer.sample_for_per(batch_size)
            # is_weights, is_indices: important sampling `weights, indices` by Prioritized Experience Replay (PER)

            next_qs = (self.cri_target(next_ss).max(dim=1,
                                                    keepdim=True)[0].squeeze(1)
                       )  # q values in next step
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.cri(states).gather(1, actions.long()).squeeze(1)
        td_errors = self.criterion(
            q_values, q_labels)  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_errors * is_weights).mean()

        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())
        return obj_critic, q_values

    def get_cumulative_rewards(self, rewards: torch.Tensor,
                               undones: torch.Tensor) -> torch.Tensor:
        returns = torch.empty_like(rewards)

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        last_state = self.last_state
        next_value = (self.act_target(last_state).argmax(dim=1).detach()
                      )  # actor is Q Network in DQN style
        for t in range(horizon_len - 1, -1, -1):
            returns[t] = next_value = rewards[t] + masks[t] * next_value
        return returns
