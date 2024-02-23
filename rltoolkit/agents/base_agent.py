import copy
import os
import time
from typing import Optional, Union

import torch
import torch.nn as nn
from rltoolkit.buffers import BaseBuffer
from rltoolkit.utils import (LinearDecayScheduler, ProgressBar,
                             TensorboardLogger, WandbLogger, get_outdir)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter

from .configs import BaseConfig


class BaseAgent:
    """The basic agent of deep-rl-toolkit.

    It is an abstract class for all DRL agents.
    """

    def __init__(
        self,
        config: BaseConfig,
        envs: Union[None],
        buffer: BaseBuffer,
        actor_model: nn.Module,
        critic_model: Optional[nn.Module] = None,
        actor_optimizer: Optimizer = None,
        critic_optimizer: Optional[Optimizer] = None,
        actor_lr_scheduler: Optional[LRScheduler] = None,
        critic_lr_scheduler: Optional[LRScheduler] = None,
        eps_greedy_scheduler: Optional[LinearDecayScheduler] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.config = config
        self.device = device

        # environment parameters
        self.envs = envs
        self.num_envs = envs.num_envs
        self.obs_sapce = envs.observation_space
        self.action_space = envs.action_space

        # training parameters
        self.gamma = config.gamma
        self.soft_update_tau = config.soft_update_tau
        self.learning_rate = config.learning_rate
        self.repeat_update_times = config.repeat_update_times

        # Epsilon-Greedy Scheduler
        self.eps_greedy = config.eps_greedy_start
        self.eps_greedy_end = config.eps_greedy_end
        self.eps_greedy_start = config.eps_greedy_start
        self.eps_greedy_scheduler = eps_greedy_scheduler

        # ReplayBuffer
        self.buffer = buffer
        self.batch_size = config.batch_size
        self.buffer_size = config.buffer_size

        # Training Parameters
        self.max_train_steps = config.max_train_steps
        self.train_frequency = config.train_frequency
        self.warmup_learn_steps = config.warmup_learn_steps
        self.target_update_frequency = config.target_update_frequency

        # Policy and Value Network
        self.actor_model = actor_model.to(self.device)
        self.actor_target = copy.deepcopy(self.actor_model).to(self.device)
        self.critic_model = (critic_model.to(self.device)
                             if critic_model else actor_model)
        self.critic_target = copy.deepcopy(self.critic_model).to(self.device)

        # Optimizer and Scheduler
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.actor_scheduler = actor_lr_scheduler
        self.critic_scheduler = critic_lr_scheduler

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
        self.progress_bar = ProgressBar(config.max_train_steps)

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

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
