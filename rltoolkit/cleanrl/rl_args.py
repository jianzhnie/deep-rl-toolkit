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
    torch_deterministic: Optional[bool] = field(
        default=True,
        metadata={'help': 'Whether to use CUDA. Defaults to True'})
    # Environment settings
    seed: Optional[int] = field(
        default=42,
        metadata={
            'help': 'Seed for environment randomization. Defaults to 123'
        },
    )
    env_id: Optional[str] = field(
        default='CartPole-v0',
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
        default=10000,
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
    optimize_memory_usage: Optional[bool] = field(
        default=False,
        metadata={
            'help': 'Whether to optimize memory usage. Defaults to False'
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
        default=0.1,
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
        default=12000,
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
        default=1e-4,
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
    clip_weights: Optional[bool] = field(
        default=False,
        metadata={
            'help':
            'Flag indicating whether to clip the weights of the model. Defaults to False'
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
        default=1,
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
        default=1.0,
        metadata={
            'help':
            'Interpolation parameter for soft target updates. Defaults to 0.95'
        },
    )
    target_update_frequency: Optional[int] = field(
        default=100,
        metadata={
            'help': 'Frequency of updating the target network. Defaults to 500'
        },
    )
    # Evaluation settings
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
    train_log_interval: Optional[int] = field(
        default=10,
        metadata={'help': 'Logging interval during training. Defaults to 1'},
    )
    test_log_interval: Optional[int] = field(
        default=20,
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


@dataclass
class DQNArguments(RLArguments):
    # DQN Algorithm settings
    hidden_dim: Optional[int] = field(default=128,
                                      metadata={'help': 'The hidden dim'})
    double_dqn: Optional[float] = field(
        default=False,
        metadata={
            'help':
            'Flag indicating whether to use double DQN. Defaults to False'
        },
    )
    dueling_dqn: Optional[float] = field(
        default=False,
        metadata={
            'help':
            'Flag indicating whether to use dueling DQN. Defaults to False'
        },
    )
    n_steps: Optional[int] = field(
        default=5,
        metadata={
            'help':
            'Number of steps to take before updating the target network. Defaults to 1'
        },
    )


@dataclass
class C51Arguments(RLArguments):
    hidden_dim: Optional[int] = field(default=128,
                                      metadata={'help': 'The hidden dim'})
    num_atoms: Optional[float] = field(default=101,
                                       metadata={'help': 'the num atoms'})
    v_min: Optional[float] = field(default=101,
                                   metadata={'help': 'the num atoms'})
    v_max: Optional[float] = field(default=101,
                                   metadata={'help': 'the num atoms'})


@dataclass
class DDPGArguments(RLArguments):
    hidden_dim: Optional[int] = field(default=128,
                                      metadata={'help': 'The hidden dim'})
    actor_lr: Optional[float] = field(
        default=1e-4,
        metadata={
            'help': 'Learning rate used by the optimizer. Defaults to 0.001'
        },
    )
    critic_lr: Optional[float] = field(
        default=1e-4,
        metadata={
            'help': 'Learning rate used by the optimizer. Defaults to 0.001'
        },
    )

    action_bound: Optional[float] = field(default=2,
                                          metadata={'help': 'action bound'})


@dataclass
class PGArguments(RLArguments):
    hidden_dim: Optional[int] = field(default=128,
                                      metadata={'help': 'The hidden dim'})
    with_baseline: Optional[bool] = field(
        default=False, metadata={'help': 'Whether to use a baseline'})


@dataclass
class PPOArguments(RLArguments):
    hidden_dim: Optional[int] = field(default=128,
                                      metadata={'help': 'The hidden dim'})
    actor_lr: Optional[float] = field(
        default=1e-4,
        metadata={
            'help': 'Learning rate used by the optimizer. Defaults to 0.001'
        },
    )
    critic_lr: Optional[float] = field(
        default=1e-4,
        metadata={
            'help': 'Learning rate used by the optimizer. Defaults to 0.001'
        },
    )
    action_bound: Optional[float] = field(default=2,
                                          metadata={'help': 'action bound'})
    lmbda: Optional[float] = field(default=2,
                                   metadata={'help': 'action bound'})
    clip_param: Optional[float] = field(default=0.2,
                                        metadata={'help': 'action bound'})
    target_kl: Optional[float] = field(default=0.01,
                                       metadata={'help': 'action bound'})
    policy_net_iters: Optional[int] = field(default=2,
                                            metadata={'help': 'action bound'})
    critic_net_iters: Optional[int] = field(default=2,
                                            metadata={'help': 'action bound'})
