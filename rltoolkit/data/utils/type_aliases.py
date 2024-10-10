from typing import NamedTuple

import numpy as np
import torch


class RolloutBufferSamples(NamedTuple):
    obs: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class ReplayBufferSamples(NamedTuple):
    obs: torch.Tensor
    actions: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class PrioritizedReplayBufferSamples(NamedTuple):
    obs: torch.Tensor
    actions: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    weights: np.ndarray
    indices: np.ndarray
