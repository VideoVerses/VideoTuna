"""Small helpers shared by legacy diffusion schedulers."""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch


def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, d: Union[Any, Callable[[], Any]]) -> Any:
    if exists(val):
        return val
    return d() if callable(d) else d


def extract_into_tensor(
    a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size
) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(
    shape: torch.Size,
    device: torch.device,
    repeat: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    dtype = dtype or torch.float32

    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device, dtype=dtype).repeat(
            shape[0], *((1,) * (len(shape) - 1))
        )

    def noise():
        return torch.randn(shape, device=device, dtype=dtype)

    return repeat_noise() if repeat else noise()
