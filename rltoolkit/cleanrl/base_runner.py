import os
import time
from abc import ABC

import gymnasium as gym
from rltoolkit.cleanrl.agent import BaseAgent
from rltoolkit.cleanrl.rl_args import RLArguments
from rltoolkit.data import SimpleReplayBuffer as ReplayBuffer
from rltoolkit.utils import (TensorboardLogger, WandbLogger, get_outdir,
                             get_text_logger)
from torch.utils.tensorboard import SummaryWriter


class BaseRunner(ABC):

    def __init__(
        self,
        args: RLArguments,
        train_env: gym.Env,
        test_env: gym.Env,
        agent: BaseAgent,
        buffer: ReplayBuffer,
    ) -> None:
        self.args = args
        self.train_env = train_env
        self.test_env = test_env
        self.agent = agent
        self.buffer = buffer

        # Logs and Visualizations
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_name = os.path.join(args.project, args.env_id, args.algo_name,
                                timestamp).replace(os.path.sep, '_')
        work_dir = os.path.join(args.work_dir, args.project, args.env_id,
                                args.algo_name)
        tensorboard_log_dir = get_outdir(work_dir, 'tensorboard_log')
        text_log_dir = get_outdir(work_dir, 'text_log')
        text_log_file = os.path.join(text_log_dir, log_name + '.log')
        self.text_logger = get_text_logger(log_file=text_log_file,
                                           log_level='INFO')

        if args.logger == 'wandb':
            self.vis_logger = WandbLogger(
                dir=work_dir,
                train_interval=args.train_log_interval,
                test_interval=args.test_log_interval,
                update_interval=args.train_log_interval,
                save_interval=args.save_interval,
                project=args.project,
                name=log_name,
                config=args,
            )
        self.writer = SummaryWriter(tensorboard_log_dir)
        self.writer.add_text('args', str(args))
        if args.logger == 'tensorboard':
            self.vis_logger = TensorboardLogger(self.writer)
        else:  # wandb
            self.vis_logger.load(self.writer)

        # Video Save
        self.video_save_dir = get_outdir(work_dir, 'video_dir')
        self.model_save_dir = get_outdir(work_dir, 'model_dir')

    def run_train_episode(self) -> dict[str, float]:
        raise NotImplementedError

    def run_test_episode(self) -> dict[str, float]:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def log_train_infos(self, infos: dict, steps: int) -> None:
        """Log training information.

        Args:
            infos (dict): Information to be visualized.
            steps (int): Current training step.
        """
        self.vis_logger.log_train_data(infos, steps)

    def log_test_infos(self, infos: dict, steps: int) -> None:
        """Log testing information.

        Args:
            infos (dict): Information to be visualized.
            steps (int): Current training step.
        """
        self.vis_logger.log_test_data(infos, steps)
