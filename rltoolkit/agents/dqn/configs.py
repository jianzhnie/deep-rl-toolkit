class DQNConfig:
    """Configuration class for DQN algorithm.

    DQNConfig contains parameters used to instantiate a DQN algorithm.
    These parameters define the algorithm's behavior, network architecture, and training settings.

    Args:
        env (str, optional): The environment name.
        hidden_dim (int, optional): Dimension of the hidden layers in the neural network.
        total_steps (int, optional): Total number of training steps.
        memory_size (int, optional): Size of the replay buffer.
        memory_warmup_size (int, optional): Number of episodes before the training starts.
        batch_size (int, optional): Batch size used during training.
        update_target_step (int, optional): Frequency of updating the target network.
        learning_rate (float, optional): Learning rate of the optimizer.
        exploration_start (float, optional): Initial value of epsilon for epsilon-greedy exploration.
        min_exploration (float, optional): Minimum value of epsilon for epsilon-greedy exploration.
        gamma (float, optional): Discount factor for future rewards.
        eval_render (bool, optional): Whether to render the evaluation environment.
        train_log_interval (int, optional): Logging interval during training.
        test_log_interval (int, optional): Logging interval during evaluation.
        log_dir (str, optional): Directory to save logs.
        logger (str, optional): Logger to use for recording logs.
    """

    model_type: str = 'dqn'

    def __init__(
        self,
        env: str = 'CartPole-v0',
        hidden_dim: int = 128,
        total_steps: int = 12000,
        memory_size: int = 10000,
        memory_warmup_size: int = 1000,
        batch_size: int = 32,
        update_target_step: int = 100,
        learning_rate: float = 0.001,
        exploration_start: float = 1.0,
        min_exploration: float = 0.1,
        gamma: float = 0.99,
        eval_render: bool = False,
        train_log_interval: int = 1,
        test_log_interval: int = 5,
        log_dir: str = 'work_dirs',
        logger: str = 'wandb',
    ) -> None:
        # Environment parameters
        self.env = env

        # Network architecture parameters
        self.hidden_dim = hidden_dim

        # Training parameters
        self.total_steps = total_steps
        self.memory_size = memory_size
        self.memory_warmup_size = memory_warmup_size
        self.batch_size = batch_size
        self.update_target_step = update_target_step
        self.learning_rate = learning_rate
        self.exploration_start = exploration_start
        self.min_exploration = min_exploration
        self.gamma = gamma
        self.eval_render = eval_render
        self.train_log_interval = train_log_interval
        self.test_log_interval = test_log_interval

        # Logging parameters
        self.log_dir = log_dir
        self.logger = logger
