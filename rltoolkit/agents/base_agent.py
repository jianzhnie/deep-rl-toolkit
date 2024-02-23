import os
import time
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from rltoolkit.buffers import BaseBuffer
from rltoolkit.utils import (ProgressBar, TensorboardLogger, WandbLogger,
                             get_outdir)
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter


class BaseAgent(ABC):
    """The basic agent of deep-rl-toolkit.

    It is an abstract class for all DRL agents.
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
        eps_greedy_scheduler: Optional[LRScheduler] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.config = config
        self.device = device
        self.render = config.render

        # environment parameters
        self.envs = envs
        self.num_envs = envs.num_envs
        self.obs_sapce = envs.observation_space
        self.action_space = envs.action_space

        # training parameters
        self.gamma = config.gamma
        self.repeat_times = config.repeat_times
        self.reward_scale = config.reward_scale
        self.if_off_policy = config.if_off_policy
        self.clip_grad_norm = config.clip_grad_norm
        self.state_value_tau = config.state_value_tau
        self.soft_update_tau = config.soft_update_tau
        self.learning_rate = config.learning_rate

        # Epsilon-Greedy Scheduler
        self.eps_greedy = config.eps_greedy_start
        self.eps_greedy_end = config.eps_greedy_end
        self.eps_greedy_start = config.eps_greedy_start
        self.eps_greedy_scheduler = eps_greedy_scheduler

        # ReplayBuffer
        self.buffer = buffer
        self.batch_size = config.batch_size
        self.max_buffer_size = config.max_buffer_size

        # Training Parameters
        self.max_steps = config.max_steps
        self.train_frequency = config.train_frequency
        self.warmup_learn_steps = config.warmup_learn_steps
        self.target_update_frequency = config.target_update_frequency

        # Policy and Value Network
        self.actor_model = actor_model.to(self.device)
        self.actor_target = self.actor_model.copy().to(self.device)
        self.critic_model = (self.critic_model.to(self.device)
                             if critic_model else actor_model)
        self.critic_target = self.critic_model.copy().to(self.device)

        # Optimizer and Scheduler
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.actor_scheduler = actor_lr_scheduler
        self.critic_scheduler = critic_lr_scheduler

        # PER (Prioritized Experience Replay)
        self.if_use_per = getattr(config, 'if_use_per', None)
        # use PER (Prioritized Experience Replay)
        if self.if_use_per:
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw

        # Logs and Plots
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        # text_log
        log_name = os.path.join(config.project, config.env_name,
                                config.algo_name,
                                timestamp).replace(os.path.sep, '_')
        log_path = os.path.join(config.work_dir, config.project,
                                config.env_name, config.algo_name)
        tensorboard_log_path = get_outdir(log_path, 'tensorboard_log_dir')
        wandb_log_path = get_outdir(log_path, 'wandb_log_dir')
        config.video_folder = get_outdir(log_path, 'video')

        if config.logger == 'wandb':
            wandb_log_path = get_outdir(log_path, 'wandb_log_dir')
            self.logger = WandbLogger(
                dir=wandb_log_path,
                train_interval=config.train_log_interval,
                test_interval=config.test_log_interval,
                update_interval=config.train_log_interval,
                save_interval=config.save_interval,
                project=config.project,
                name=log_name,
                config=config,
            )
            self.use_wandb = True
        self.writer = SummaryWriter(tensorboard_log_path)
        self.writer.add_text('config', str(config))
        if config.logger == 'tensorboard':
            self.logger = TensorboardLogger(self.writer)
            self.use_wandb = False
        else:  # wandb
            self.logger.load(self.writer)

        # ProgressBar
        self.progress_bar = ProgressBar(config.max_step)

        # Model Save and Load
        self.model_save_dir = get_outdir(log_path, 'model_dir')
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

    def optimizer_update_amp(self, optimizer: torch.optim,
                             objective: torch.Tensor) -> None:
        # automatic mixed precision
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

    def update_avg_std_for_normalization(self, states: torch.Tensor,
                                         returns: torch.Tensor) -> None:
        tau = self.state_value_tau
        if tau == 0:
            return

        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.actor_model.state_avg[:] = (self.actor_model.state_avg *
                                         (1 - tau) + state_avg * tau)
        self.actor_model.state_std[:] = (self.critic_model.state_std *
                                         (1 - tau) + state_std * tau + 1e-4)
        self.actor_model.state_avg[:] = self.actor_model.state_avg
        self.critic_model.state_std[:] = self.actor_model.state_std

        returns_avg = returns.mean(dim=0)
        returns_std = returns.std(dim=0)
        self.critic_model.value_avg[:] = (self.critic_model.value_avg *
                                          (1 - tau) + returns_avg * tau)
        self.critic_model.value_std[:] = (self.critic_model.value_std *
                                          (1 - tau) + returns_std * tau + 1e-4)

    def save_model(self, save_dir: str) -> None:
        """save or load training files for Agent.

        save_dir: Current Working Directory.
        if_save: True: save files. False: load files.
        """
        assert self.save_attr_names.issuperset(
            {'actor_model', 'actor_target', 'actor_optimizer'})

        for attr_name in self.save_attr_names:
            file_path = f'{save_dir}/{attr_name}.pth'
            torch.save(getattr(self, attr_name), file_path)

    def load_model(self, save_dir: str) -> None:
        assert self.save_attr_names.issuperset(
            {'actor_model', 'actor_target', 'actor_optimizer'})

        for attr_name in self.save_attr_names:
            file_path = f'{save_dir}/{attr_name}.pth'
            assert os.path.isfile(file_path)
            model_state = torch.load(file_path, map_location=self.device)
            setattr(self, attr_name, model_state)

    def log_train_infos(self, infos: dict, steps: int) -> None:
        """
        info: (dict) information to be visualized
        n_steps: current step
        """
        self.logger.log_train_data(infos, steps)

    def log_test_infos(self, infos: dict, steps: int) -> None:
        """
        info: (dict) information to be visualized
        steps: current step
        """
        self.logger.log_test_data(infos, steps)

    @abstractmethod
    def train(self, steps):
        raise NotImplementedError

    @abstractmethod
    def test(self, env_fn, steps):
        raise NotImplementedError
