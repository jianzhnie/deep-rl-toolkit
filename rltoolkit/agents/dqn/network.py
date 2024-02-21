import numpy as np
import torch.nn as nn


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):

    def __init__(self, env) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(
                np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)
