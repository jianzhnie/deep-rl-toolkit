import random
import sys

import numpy as np
import torch

sys.path.append('../')
from rltoolkit.agents.configs import BaseConfig as DQNConfig
from rltoolkit.runner import Runner

if __name__ == '__main__':
    config = DQNConfig()
    # TRY NOT TO MODIFY: seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    runner = Runner(config)
    runner.run()
