import os
import time

from rltoolkit.utils import (ProgressBar, TensorboardLogger, WandbLogger,
                             get_outdir, get_root_logger)
from torch.utils.tensorboard import SummaryWriter

from .configs import BaseConfig


class BaseAgent:
    """The basic agent of deep-rl-toolkit. It is an abstract class for all DRL
    agents.

    Args:
        config (BaseConfig): Configuration object for the agent.
        envs (Union[None]): Environment object.
        buffer (BaseBuffer): Replay buffer for experience replay.
        actor_model (nn.Module): Actor model representing the policy network.
        critic_model (Optional[nn.Module], optional): Critic model representing the value network. Defaults to None.
        actor_optimizer (Optional[Optimizer], optional): Optimizer for actor model. Defaults to None.
        critic_optimizer (Optional[Optimizer], optional): Optimizer for critic model. Defaults to None.
        actor_lr_scheduler (Optional[_LRScheduler], optional): Learning rate scheduler for actor model. Defaults to None.
        critic_lr_scheduler (Optional[_LRScheduler], optional): Learning rate scheduler for critic model. Defaults to None.
        eps_greedy_scheduler (Optional[LinearDecayScheduler], optional): Epsilon-greedy scheduler. Defaults to None.
        device (Optional[Union[str, torch.device]], optional): Device to run the agent on. Defaults to None.
    """

    def __init__(self, config: BaseConfig) -> None:
        # Logs and Visualizations
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_name = os.path.join(config.project, config.env_name,
                                config.algo_name,
                                timestamp).replace(os.path.sep, '_')
        work_dir = os.path.join(config.work_dir, config.project,
                                config.env_name, config.algo_name)
        tensorboard_log_dir = get_outdir(work_dir, 'tensorboard_log')
        text_log_dir = get_outdir(work_dir, 'text_log')
        text_log_file = os.path.join(text_log_dir, log_name + '.log')
        self.text_logger = get_root_logger(log_file=text_log_file,
                                           log_level='INFO')

        if config.logger == 'wandb':
            wandb_log_dir = get_outdir(work_dir, 'wandb_log')
            self.vis_logger = WandbLogger(
                dir=wandb_log_dir,
                train_interval=config.train_log_interval,
                test_interval=config.test_log_interval,
                update_interval=config.train_log_interval,
                save_interval=config.save_interval,
                project=config.project,
                name=log_name,
                config=config,
            )
        self.writer = SummaryWriter(tensorboard_log_dir)
        self.writer.add_text('config', str(config))
        if config.logger == 'tensorboard':
            self.vis_logger = TensorboardLogger(self.writer)
        else:  # wandb
            self.vis_logger.load(self.writer)

        # ProgressBar
        self.progress_bar = ProgressBar(config.max_timesteps)
        # Video Save
        self.video_save_dir = get_outdir(work_dir, 'video_dir')
        self.model_save_dir = get_outdir(work_dir, 'model_dir')

    def save_model(self, save_dir: str) -> None:
        """Save the model.

        Args:
            save_dir (str): Directory to save the model.
            steps (int): Current training step.
        """
        raise NotImplementedError

    def load_model(self, save_dir: str) -> None:
        """Load the model.

        Args:
            save_dir (str): Directory to load the model from.
            steps (int): Current training step.
        """
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

    def train(self):
        """Train the agent."""
        raise NotImplementedError

    def evaluate(self):
        """Test the agent."""
        raise NotImplementedError
