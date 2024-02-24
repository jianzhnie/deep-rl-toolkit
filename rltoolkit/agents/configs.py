class BaseConfig:
    """Configuration class for the DQN algorithm.

    DQNConfig contains parameters used to instantiate a DQN algorithm.
    These parameters define the algorithm's behavior, network architecture, and training settings.

    Attributes:
        project (str, optional): Name of the project. Defaults to "Gym".
        torch_deterministic (bool, optional): Whether to set torch's random seed for deterministic behavior. Defaults to True.
        seed (int, optional): Seed for environment randomization. Defaults to 123.
        env_name (str, optional): The environment name. Defaults to "CartPole-v0".
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
        max_train_steps (int, optional): Maximum number of training steps. Defaults to 12000.
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
        use_cuda: bool = True,
        optimize_memory_usage: bool = False,
        torch_deterministic: bool = True,
        # Environment settings
        seed: int = 123,
        env_name: str = 'CartPole-v0',
        num_envs: int = 10,
        capture_video: bool = False,
        # Buffer settings
        buffer_size: int = 10000,
        batch_size: int = 128,
        # Epsilon-Greedy Scheduler settings
        eps_greedy_end: float = 0.05,
        eps_greedy_start: float = 1.0,
        eps_greedy_scheduler: str = 'linear',
        # Training parameters
        gamma: float = 0.99,
        learning_rate: float = 2.5e-4,
        max_grad_norm: float = 1.0,
        lr_scheduler_method: str = 'linear',
        max_train_steps: int = 10000,
        warmup_learn_steps: int = 100,
        train_frequency: int = 100,
        repeat_update_times: int = 5,
        soft_update_tau: float = 0.05,
        target_update_frequency: int = 500,
        # Evaluation settings
        eval_frequency: int = 1000,
        eval_episodes: int = 10,
        # Log and Model Save
        work_dir: str = 'work_dirs',
        save_model: bool = False,
        save_model_frequency: int = 10000,
        train_log_interval: int = 10,
        test_log_interval: int = 100,
        save_interval: int = 1,
        logger: str = 'tensorboard',
    ) -> None:
        # Common settings
        self.project = project
        self.algo_name = algo_name
        self.use_cuda = use_cuda
        self.optimize_memory_usage = optimize_memory_usage
        self.torch_deterministic = torch_deterministic
        # Environment parameters
        self.seed = seed
        self.env_name = env_name
        self.num_envs = num_envs
        self.capture_video = capture_video
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        # Epsilon-Greedy Scheduler settings
        self.eps_greedy_end = eps_greedy_end
        self.eps_greedy_start = eps_greedy_start
        self.eps_greedy_scheduler = eps_greedy_scheduler
        # Training parameters
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.lr_scheduler_method = lr_scheduler_method
        self.max_train_steps = max_train_steps
        self.warmup_learn_steps = warmup_learn_steps
        self.train_frequency = train_frequency
        self.repeat_update_times = repeat_update_times
        self.soft_update_tau = soft_update_tau
        self.target_update_frequency = target_update_frequency
        # Evaluation settings
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        # Log and Model Save
        self.work_dir = work_dir
        self.save_model = save_model
        self.save_model_frequency = save_model_frequency
        self.train_log_interval = train_log_interval
        self.test_log_interval = test_log_interval
        self.save_interval = save_interval
        self.logger = logger
