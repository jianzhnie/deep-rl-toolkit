import os
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from rltoolkit.buffers import BaseBuffer
from torch.nn.utils import clip_grad_norm_


class BaseAgent(ABC):
    """The basic agent of deep-rl-toolkit. It is an abstract class for all DRL
    agents.

    net_dims: the middle layer dimension of MLP (MultiLayer Perceptron)
    state_dim: the dimension of state (the number of state vector)
    action_dim: the dimension of action (or the number of discrete action)
    gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    args: the arguments for agent training. `args = Config()`
    """

    def __init__(
        self,
        config: Namespace,
        envs: Union[None],
        actor_model: nn.Module,
        critic_model: Optional[nn.Module] = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[Union[int, str, torch.device]] = None,
    ) -> None:
        self.device = device

        self.render = config.render

        # environment parameters
        self.num_envs = envs.num_envs if envs else 1
        self.obs_sapce = envs.observation_space if envs else None
        self.act_space = envs.action_space if envs else None

        # training parameters
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.repeat_times = config.repeat_times
        self.reward_scale = config.reward_scale
        self.if_off_policy = config.if_off_policy
        self.clip_grad_norm = config.clip_grad_norm
        self.soft_update_tau = config.soft_update_tau
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.learning_rate = config.learning_rate

        # Policy and Value Network
        self.actor_model = actor_model.to(self.device)
        self.actor_target = self.actor_model.copy().to(self.device)
        self.critic_model = (self.critic_model.to(self.device)
                             if critic_model else actor_model)
        self.critic_target = self.critic_model.copy().to(self.device)

        # Optimizer and Scheduler
        self.actor_optimizer = torch.optim.AdamW(self.actor_model.parameters(),
                                                 self.learning_rate)
        self.critic_optimizer = (torch.optim.AdamW(
            self.critic_model.parameters(), self.learning_rate)
                                 if critic_model else self.actor_optimizer)
        from types import MethodType  # built-in package of Python3

        self.actor_optimizer.parameters = MethodType(get_optim_param,
                                                     self.actor_optimizer)
        self.critic_optimizer.parameters = MethodType(get_optim_param,
                                                      self.critic_optimizer)

        self.if_use_per = getattr(
            config, 'if_use_per',
            None)  # use PER (Prioritized Experience Replay)
        if self.if_use_per:
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw

        # Save and Load
        self.save_attr_names = {
            'actor_model',
            'actor_target',
            'act_optimizer',
            'critic_model',
            'critic_target',
            'critic_optimizer',
        }

    def update_net(self, buffer: Union[BaseBuffer,
                                       tuple]) -> Tuple[float, ...]:
        obj_critic = 0.0  # criterion(q_value, q_label).mean().item()
        obj_actor = 0.0  # q_value.mean().item()
        assert isinstance(buffer, BaseBuffer) or isinstance(buffer, tuple)
        assert isinstance(self.batch_size, int)
        assert isinstance(self.repeat_times, int)
        assert isinstance(self.reward_scale, float)
        return obj_critic, obj_actor

    def get_obj_critic_raw(
        self,
        buffer: BaseBuffer,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(
                batch_size)  # next_ss: next states
            next_as = self.actor_target(next_ss)  # next actions
            next_qs = self.critic_target(next_ss, next_as)  # next q values
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.critic_model(states, actions)
        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, states

    def get_obj_critic_per(
        self,
        buffer: BaseBuffer,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.torch.Tensor]:
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

            next_as = self.actor_target(next_ss)
            next_qs = self.critic_target(next_ss, next_as)
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.critic_model(states, actions)
        td_errors = self.criterion(q_values, q_labels)
        obj_critic = (td_errors * is_weights).mean()

        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())
        return obj_critic, states

    def get_cumulative_rewards(self, rewards: torch.Tensor,
                               undones: torch.Tensor) -> torch.Tensor:
        returns = torch.empty_like(rewards)

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        last_state = self.last_state
        next_action = self.actor_target(last_state)
        next_value = self.critic_target(last_state, next_action).detach()
        for t in range(horizon_len - 1, -1, -1):
            returns[t] = next_value = rewards[t] + masks[t] * next_value
        return returns

    def optimizer_update(self, optimizer: torch.optim,
                         objective: torch.Tensor) -> None:
        """minimize the optimization objective via update the network
        parameters.

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]['params'],
                        max_norm=self.clip_grad_norm)
        optimizer.step()

    def optimizer_update_amp(
            self, optimizer: torch.optim,
            objective: torch.Tensor):  # automatic mixed precision
        """minimize the optimization objective via update the network
        parameters.

        amp: Automatic Mixed Precision

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        amp_scale = torch.cuda.amp.GradScaler()  # write in __init__()

        optimizer.zero_grad()
        amp_scale.scale(objective).backward()  # loss.backward()
        amp_scale.unscale_(optimizer)  # amp

        # from torch.nn.utils import clip_grad_norm_
        clip_grad_norm_(parameters=optimizer.param_groups[0]['params'],
                        max_norm=self.clip_grad_norm)
        amp_scale.step(optimizer)  # optimizer.step()
        amp_scale.update()  # optimizer.step()

    @staticmethod
    def soft_update(target_net: nn.Module, current_net: nn.Module, tau: float):
        """soft update target network via current network.

        target_net: update target network via current network to make training more stable.
        current_net: current network update via an optimizer
        tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, save_dir: str, if_save: bool) -> None:
        """save or load training files for Agent.

        cwd: Current Working Directory. ElegantRL save training files in CWD.
        if_save: True: save files. False: load files.
        """
        assert self.save_attr_names.issuperset(
            {'actor_model', 'actor_target', 'act_optimizer'})

        for attr_name in self.save_attr_names:
            file_path = f'{save_dir}/{attr_name}.pth'
            if if_save:
                torch.save(getattr(self, attr_name), file_path)
            elif os.path.isfile(file_path):
                setattr(self, attr_name,
                        torch.load(file_path, map_location=self.device))

    @abstractmethod
    def train(self, steps):
        raise NotImplementedError

    @abstractmethod
    def test(self, env_fn, steps):
        raise NotImplementedError


def get_optim_param(optimizer: torch.optim) -> list:  # backup
    params_list = []
    for params_dict in optimizer.state_dict()['state'].values():
        params_list.extend([
            t for t in params_dict.values()
            if isinstance(t, torch.torch.Tensor)
        ])
    return params_list
