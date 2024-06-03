from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RLArguments:
    # common settings
    project: Optional[str] = field(
        default='rltoolkit',
        metadata={'help': "Name of the project. Defaults to 'Gym'"},
    )
    algo_name: Optional[str] = field(
        default='dqn',
        metadata={'help': "Name of the algorithm. Defaults to 'dqn'"},
    )
    use_cuda: Optional[bool] = field(
        default=True,
        metadata={'help': 'Whether to use CUDA. Defaults to True'},
    )
    optimize_memory_usage: Optional[bool] = field(
        default=False,
        metadata={
            'help': 'Whether to optimize memory usage. Defaults to False'
        },
    )
    # Environment settings
    seed: Optional[int] = field(
        default=42,
        metadata={
            'help': 'Seed for environment randomization. Defaults to 123'
        },
    )
    env_id: Optional[str] = field(
        default='CartPole-v1',
        metadata={'help': "The environment name. Defaults to 'CartPole-v0'"},
    )
    num_envs: Optional[int] = field(
        default=10,
        metadata={
            'help':
            'Number of parallel environments to run for collecting experiences. Defaults to 10'
        },
    )
    capture_video: Optional[bool] = field(
        default=None,
        metadata={
            'help':
            'Flag indicating whether to capture videos of the environment during training. Defaults to True'
        },
    )
    # Buffer settings
    buffer_size: Optional[int] = field(
        default=1000000,
        metadata={
            'help': 'Maximum size of the replay buffer. Defaults to 10000'
        },
    )
    batch_size: Optional[int] = field(
        default=32,
        metadata={
            'help':
            'Size of the mini-batches sampled from the replay buffer during training. Defaults to 32'
        },
    )
    # Epsilon-Greedy Scheduler settings
    eps_greedy_start: Optional[float] = field(
        default=1.0,
        metadata={
            'help':
            'Initial value of epsilon for epsilon-greedy exploration. Defaults to 1.0'
        },
    )
    eps_greedy_end: Optional[float] = field(
        default=0.05,
        metadata={
            'help':
            'Final value of epsilon for epsilon-greedy exploration. Defaults to 0.001'
        },
    )
    eps_greedy_scheduler: Optional[str] = field(
        default='linear',
        metadata={
            'help':
            "Type of scheduler used for epsilon-greedy exploration. Defaults to 'Linear'"
        },
    )
    # Training parameters
    max_timesteps: Optional[int] = field(
        default=1e6,
        metadata={
            'help': 'Maximum number of training steps. Defaults to 12000'
        },
    )
    gamma: Optional[float] = field(
        default=0.99,
        metadata={
            'help': 'Discount factor for future rewards. Defaults to 0.99'
        },
    )
    learning_rate: Optional[float] = field(
        default=1e-3,
        metadata={
            'help': 'Learning rate used by the optimizer. Defaults to 0.001'
        },
    )
    min_learning_rate: Optional[float] = field(
        default=1e-5,
        metadata={
            'help':
            'Minimum learning rate used by the optimizer. Defaults to 1e-5'
        },
    )
    lr_scheduler_method: Optional[str] = field(
        default='linear',
        metadata={
            'help':
            "Method used for learning rate scheduling. Defaults to 'linear'"
        },
    )
    max_grad_norm: Optional[float] = field(
        default=1.0,
        metadata={'help': 'Maximum gradient norm. Defaults to 1.0'},
    )
    warmup_learn_steps: Optional[int] = field(
        default=1000,
        metadata={
            'help':
            'Number of steps before starting to update the model. Defaults to 1000'
        },
    )
    train_frequency: Optional[int] = field(
        default=100,
        metadata={'help': 'Frequency of training updates. Defaults to 200'},
    )
    gradient_steps: Optional[int] = field(
        default=1,
        metadata={
            'help':
            'Number of times to update the learner network. Defaults to 5'
        },
    )
    soft_update_tau: Optional[float] = field(
        default=0.05,
        metadata={
            'help':
            'Interpolation parameter for soft target updates. Defaults to 0.95'
        },
    )
    target_update_frequency: Optional[int] = field(
        default=500,
        metadata={
            'help': 'Frequency of updating the target network. Defaults to 500'
        },
    )

    # Evaluation settings
    eval_frequency: Optional[int] = field(
        default=1000,
        metadata={'help': 'Frequency of evaluation. Defaults to 1000'},
    )
    eval_episodes: Optional[int] = field(
        default=10,
        metadata={'help': 'Number of episodes to evaluate. Defaults to 10'},
    )
    # Log and Model Save
    work_dir: Optional[str] = field(
        default='work_dirs',
        metadata={
            'help':
            "Directory for storing work-related files. Defaults to 'work_dirs'"
        },
    )
    save_model: Optional[bool] = field(
        default=None,
        metadata={
            'help':
            'Flag indicating whether to save the trained model. Defaults to False'
        },
    )
    save_model_frequency: Optional[int] = field(
        default=1000,
        metadata={'help': 'Frequency of saving the model. Defaults to 10000'},
    )
    train_log_interval: Optional[int] = field(
        default=100,
        metadata={'help': 'Logging interval during training. Defaults to 1'},
    )
    test_log_interval: Optional[int] = field(
        default=200,
        metadata={'help': 'Logging interval during evaluation. Defaults to 5'},
    )
    save_interval: Optional[int] = field(
        default=1000,
        metadata={'help': 'Frequency of saving the model. Defaults to 1'},
    )
    logger: Optional[str] = field(
        default='wandb',
        metadata={
            'help': "Logger to use for recording logs. Defaults to 'wandb'"
        },
    )


class DQNConfig:
    """Configuration class for the DQN algorithm.

    DQNConfig contains parameters used to instantiate a DQN algorithm.
    These parameters define the algorithm's behavior, network architecture, and training settings.

    Attributes:
        project (str, optional): Name of the project. Defaults to "Gym".
        torch_deterministic (bool, optional): Whether to set torch's random seed for deterministic behavior. Defaults to True.
        seed (int, optional): Seed for environment randomization. Defaults to 123.
        env_id (str, optional): The environment name. Defaults to "CartPole-v0".
        num_envs (int, optional): Number of parallel environments to run for collecting experiences. Defaults to 10.
        capture_video (bool, optional): Flag indicating whether to capture videos of the environment during training. Defaults to True.
        buffer_size (int, optional): Maximum size of the replay buffer. Defaults to 10000.
        batch_size (int, optional): Size of the mini-batches sampled from the replay buffer during training. Defaults to 32.
        eps_greedy_end (float, optional): Final value of epsilon for epsilon-greedy exploration. Defaults to 0.001.
        eps_greedy_start (float, optional): Initial value of epsilon for epsilon-greedy exploration. Defaults to 1.0.
        eps_greedy_scheduler (str, optional): Type of scheduler used for epsilon-greedy exploration. Defaults to "Linear".
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
        learning_rate (float, optional): Learning rate used by the optimizer. Defaults to 0.001.
        lr_scheduler_method (str, optional): Method used for learning rate scheduling. Defaults to "linear".
        max_timesteps (int, optional): Maximum number of training steps. Defaults to 12000.
        warmup_learn_steps (int, optional): Number of steps before starting to update the model. Defaults to 1000.
        train_frequency (int, optional): Frequency of training updates. Defaults to 200.
        learner_update_times (int, optional): Number of times to update the learner network. Defaults to 5.
        soft_update_tau (float, optional): Interpolation parameter for soft target updates. Defaults to 0.95.
        update_target_frequency (int, optional): Frequency of updating the target network. Defaults to 500.
        work_dir (str, optional): Directory for storing work-related files. Defaults to "work_dirs".
        save_model (bool, optional): Flag indicating whether to save the trained model. Defaults to False.
        model_dir (str, optional): Directory for saving the trained model. Defaults to "model_dir".
        train_log_interval (int, optional): Logging interval during training. Defaults to 1.
        test_log_interval (int, optional): Logging interval during evaluation. Defaults to 5.
        logger (str, optional): Logger to use for recording logs. Defaults to "wandb".
    """

    def __init__(
        self,
        # Common settings
        project: str = 'Gym',
        algo_name: str = 'dqn',
        cuda: bool = True,
        torch_deterministic: bool = True,
        # Environment settings
        seed: int = 123,
        env_id: str = 'CartPole-v1',
        num_envs: int = 1,
        capture_video: bool = True,
        # Buffer settings
        buffer_size: int = 10000,
        batch_size: int = 128,
        # Epsilon-Greedy Scheduler settings
        eps_greedy_end: float = 0.05,
        eps_greedy_start: float = 1.0,
        eps_greedy_scheduler: str = 'linear',
        # Training parameters
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-5,
        lr_scheduler_method: str = 'linear',
        max_timesteps: int = 100000,
        warmup_learn_steps: int = 1000,
        train_frequency: int = 100,
        gradient_steps: int = 5,
        soft_update_tau: float = 0.05,
        target_update_frequency: int = 500,
        # Log and Model Save
        work_dir: str = 'work_dirs',
        save_model: bool = True,
        save_model_frequency: int = 10000,
        train_log_interval: int = 10,
        test_log_interval: int = 100,
        save_interval: int = 1,
        logger: str = 'wandb',
    ) -> None:
        self.project = project
        self.algo_name = algo_name
        self.cuda = cuda
        self.torch_deterministic = torch_deterministic
        self.seed = seed
        self.env_id = env_id
        self.num_envs = num_envs
        self.capture_video = capture_video
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.eps_greedy_end = eps_greedy_end
        self.eps_greedy_start = eps_greedy_start
        self.eps_greedy_scheduler = eps_greedy_scheduler
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.lr_scheduler_method = lr_scheduler_method
        self.max_timesteps = max_timesteps
        self.warmup_learn_steps = warmup_learn_steps
        self.train_frequency = train_frequency
        self.gradient_steps = gradient_steps
        self.soft_update_tau = soft_update_tau
        self.target_update_frequency = target_update_frequency
        self.work_dir = work_dir
        self.save_model = save_model
        self.save_model_frequency = save_model_frequency
        self.train_log_interval = train_log_interval
        self.test_log_interval = test_log_interval
        self.save_interval = save_interval
        self.logger = logger
