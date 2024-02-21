import torch.nn as nn


def hard_target_update(src_model: nn.Module, tgt_model: nn.Module) -> None:
    """Hard update model parameters.

    Params
    ======
        src_model: PyTorch model (weights will be copied from)
        tgt_model: PyTorch model (weights will be copied to)
    """

    tgt_model.load_state_dict(src_model.state_dict())


def soft_target_update(src_model: nn.Module,
                       tgt_model: nn.Module,
                       tau: float = 0.005) -> None:
    """Soft update model parameters.

    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        src_model: PyTorch model (weights will be copied from)
        tgt_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
    for src_model_param, tgt_model_param in zip(src_model.parameters(),
                                                tgt_model.parameters()):
        tgt_model_param.data.copy_(tau * src_model_param.data +
                                   (1.0 - tau) * tgt_model_param.data)
