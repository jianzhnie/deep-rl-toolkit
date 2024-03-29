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
        max_timesteps: int = 500000,
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
