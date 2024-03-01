import random
from numbers import Number
from typing import Any, no_type_check

import numpy as np
import torch


# TODO: confusing name, could actually return a batch...
#  Overrides and generic types should be added
@no_type_check
def to_numpy(x: Any) -> np.ndarray:
    """Return an object without torch.Tensor."""
    if isinstance(x, torch.Tensor):  # most often case
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):  # second often case
        return x
    if isinstance(x, (np.number, np.bool_, Number)):
        return np.asanyarray(x)
    if x is None:
        return np.array(None, dtype=object)
    # fallback
    return np.asanyarray(x)


@no_type_check
def to_torch(
    x: Any,
    dtype: torch.dtype | None = None,
    device: str | int | torch.device = 'cpu',
) -> torch.Tensor:
    """Return an object without np.ndarray."""
    if isinstance(x, np.ndarray) and issubclass(
            x.dtype.type,
            np.bool_ | np.number,
    ):  # most often case
        x = torch.from_numpy(x).to(device)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, torch.Tensor):  # second often case
        if dtype is not None:
            x = x.type(dtype)
        return x.to(device)
    if isinstance(x, np.number | np.bool_ | Number):
        return to_torch(np.asanyarray(x), dtype, device)
    # fallback
    raise TypeError(f'object {x} cannot be converted to torch.')


def to_torch_as(x: Any, y: torch.Tensor) -> torch.Tensor:
    """Return an object without np.ndarray.

    Same as ``to_torch(x, dtype=y.dtype, device=y.device)``.
    """
    assert isinstance(y, torch.Tensor)
    return to_torch(x, dtype=y.dtype, device=y.device)


def set_seed(seed) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
