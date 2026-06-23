"""
Unified attention backend selection for VideoTuna model families.

See ``videotuna.settings.PrivTuneSettings`` for VIDEOTUNA_* env vars.
"""

from __future__ import annotations

import importlib
import math
import os
from contextlib import contextmanager
from typing import Literal, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from videotuna.settings import ENV_ATTN_BACKEND, get_settings
from videotuna.utils.device_utils import detect_compute_backend, gpu_is_available

AttnBackend = Literal["flash", "sdpa", "eager"]
AttnLayout = Literal["bsnd", "bhsd"]


def _optional_attr(module_name: str, attr_name: str):
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return getattr(module, attr_name, None)


_FLASH_ATTN_FUNC = _optional_attr("flash_attn", "flash_attn_func")
_FLASH_ATTN_VARLEN_FUNC = _optional_attr("flash_attn", "flash_attn_varlen_func")
_FLASH_ATTN_3_VARLEN_FUNC = _optional_attr(
    "flash_attn_interface", "flash_attn_varlen_func"
)
_FLASH_ATTN_AVAILABLE = _FLASH_ATTN_FUNC is not None


def is_flash_attn_available() -> bool:
    return _FLASH_ATTN_AVAILABLE


def _resolve_auto_backend() -> AttnBackend:
    if detect_compute_backend() == "rocm":
        return "sdpa" if gpu_is_available() else "eager"
    if _FLASH_ATTN_AVAILABLE and gpu_is_available():
        return "flash"
    if gpu_is_available():
        return "sdpa"
    return "eager"


def get_attn_backend_requested() -> str:
    """Return the attention backend requested via env (before fallback)."""
    return get_settings().attn_backend


def get_attn_backend() -> AttnBackend:
    """Resolve the active attention backend from env or auto-detection."""
    settings = get_settings()
    requested = settings.attn_backend
    if requested == "auto":
        return _resolve_auto_backend()
    if requested in ("flash", "sdpa", "eager"):
        if requested == "flash":
            if detect_compute_backend() in ("rocm", "cpu"):
                backend_label = (
                    "AMD ROCm" if detect_compute_backend() == "rocm" else "CPU"
                )
                raise RuntimeError(
                    "VIDEOTUNA_ATTN_BACKEND=flash is not supported on "
                    f"{backend_label}. "
                    "Use VIDEOTUNA_ATTN_BACKEND=sdpa or eager. "
                    "See docs/install-rocm.md or docs/install-cpu.md."
                )
            if not _FLASH_ATTN_AVAILABLE:
                if settings.attn_backend_strict:
                    raise RuntimeError(
                        "VIDEOTUNA_ATTN_BACKEND=flash requires flash-attn. "
                        "Install with: poetry run install-flash-attn"
                    )
                logger.warning(
                    "VIDEOTUNA_ATTN_BACKEND=flash requested but flash-attn is not "
                    "installed; falling back to sdpa. Set "
                    "VIDEOTUNA_ATTN_BACKEND_STRICT=1 to fail instead."
                )
                return "sdpa"
        if requested == "sdpa" and not gpu_is_available():
            return "eager"
        return requested  # type: ignore[return-value]
    raise ValueError(
        f"Invalid {ENV_ATTN_BACKEND}={requested!r}. "
        "Expected auto, flash, sdpa, or eager."
    )


def get_resolved_attn_backend() -> AttnBackend:
    """Alias for get_attn_backend (resolved after auto-detection / fallback)."""
    return get_attn_backend()


def get_torch_compile_mode() -> str:
    return get_settings().torch_compile_mode


def _to_bhsd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: AttnLayout
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if layout == "bhsd":
        return q, k, v
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


def _from_bhsd(x: torch.Tensor, layout: AttnLayout) -> torch.Tensor:
    if layout == "bhsd":
        return x
    return x.transpose(1, 2)


@contextmanager
def _sdpa_context():
    """Prefer flash/mem-efficient SDPA kernels on CUDA when available."""
    if not gpu_is_available():
        yield
        return
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if detect_compute_backend() == "rocm":
            backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        else:
            backends = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        with sdpa_kernel(backends):
            yield
    except (ImportError, AttributeError):
        yield


def attention_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    scale: Optional[float] = None,
    layout: AttnLayout = "bsnd",
) -> torch.Tensor:
    q, k, v = _to_bhsd(q, k, v, layout)
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))

    b, _, s, _ = q.shape
    s1 = k.size(2)
    attn_bias = torch.zeros(b, q.size(1), s, s1, dtype=q.dtype, device=q.device)
    if causal:
        assert attn_mask is None, "Causal mask and attn_mask cannot be used together"
        temp_mask = torch.ones(
            b, q.size(1), s, s, dtype=torch.bool, device=q.device
        ).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            if attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            if attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_bias = attn_bias + attn_mask

    dtype = q.dtype
    attn = (q * scale) @ k.transpose(-2, -1)
    attn = attn + attn_bias
    attn = attn.softmax(dim=-1).to(dtype)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=True)
    out = attn @ v
    return _from_bhsd(out, layout)


def attention_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    scale: Optional[float] = None,
    layout: AttnLayout = "bsnd",
    backend: Optional[AttnBackend] = None,
) -> torch.Tensor:
    """Dense attention with unified backend selection."""
    backend = backend or get_attn_backend()

    if backend == "flash":
        if layout == "bhsd":
            q_f, k_f, v_f = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        else:
            q_f, k_f, v_f = q, k, v
        assert _FLASH_ATTN_FUNC is not None
        return _FLASH_ATTN_FUNC(
            q_f,
            k_f,
            v_f,
            dropout_p=dropout_p,
            softmax_scale=scale,
            causal=causal,
        )

    if backend == "sdpa":
        q_s, k_s, v_s = _to_bhsd(q, k, v, layout)
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q_s.dtype)
        with _sdpa_context():
            out = F.scaled_dot_product_attention(
                q_s,
                k_s,
                v_s,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=scale,
            )
        return _from_bhsd(out, layout)

    return attention_eager(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        causal=causal,
        scale=scale,
        layout=layout,
    )


def attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    dropout_p: float = 0.0,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    batch_size: Optional[int] = None,
    window_size: Tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    prefer_flash3: bool = True,
    backend: Optional[AttnBackend] = None,
) -> torch.Tensor:
    """Variable-length packed attention (flash varlen or dense fallback)."""
    backend = backend or get_attn_backend()

    if backend == "flash":
        if prefer_flash3 and _FLASH_ATTN_3_VARLEN_FUNC is not None:
            out = _FLASH_ATTN_3_VARLEN_FUNC(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_kv,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic,
            )
            if isinstance(out, tuple):
                out = out[0]
        else:
            assert _FLASH_ATTN_VARLEN_FUNC is not None
            out = _FLASH_ATTN_VARLEN_FUNC(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
            )
        if batch_size is not None:
            return out.view(batch_size, max_seqlen_q, out.shape[-2], out.shape[-1])
        return out

    if batch_size is None:
        raise ValueError("batch_size is required for non-flash varlen fallback")

    # Reshape packed varlen tensors back to padded batch for sdpa/eager.
    q.shape[0]
    n_heads = q.shape[1]
    head_dim = q.shape[2]
    q_pad = q.view(batch_size, max_seqlen_q, n_heads, head_dim)
    k_pad = k.view(batch_size, max_seqlen_kv, n_heads, head_dim)
    v_pad = v.view(batch_size, max_seqlen_kv, n_heads, head_dim)
    return attention_dense(
        q_pad,
        k_pad,
        v_pad,
        dropout_p=dropout_p,
        causal=causal,
        scale=softmax_scale,
        layout="bsnd",
        backend=backend,
    )


_DIFFUSERS_BACKEND_MAP = {
    "flash": "flash",
    "sdpa": "native",
    "eager": "_native_math",
}


def apply_diffusers_attention_backend(model) -> None:
    """Map VIDEOTUNA_ATTN_BACKEND to diffusers set_attention_backend."""
    backend = get_attn_backend()
    diffusers_backend = _DIFFUSERS_BACKEND_MAP[backend]
    if backend == "flash" and detect_compute_backend() == "rocm":
        diffusers_backend = "native"

    if hasattr(model, "set_attention_backend"):
        try:
            model.set_attention_backend(diffusers_backend)
            return
        except ValueError:
            if backend == "flash":
                model.set_attention_backend("native")
                return
            raise

    os.environ["DIFFUSERS_ATTN_BACKEND"] = diffusers_backend


_COMPILE_WARNED_ROCM = False


def maybe_compile_denoiser(module: nn.Module) -> nn.Module:
    """Optionally compile a denoiser module when VIDEOTUNA_TORCH_COMPILE=1."""
    global _COMPILE_WARNED_ROCM
    if not get_settings().torch_compile:
        return module
    if not gpu_is_available():
        return module
    if detect_compute_backend() == "rocm" and not _COMPILE_WARNED_ROCM:
        logger.warning(
            "torch.compile on AMD ROCm is experimental in PyTorch 2.6; "
            "set VIDEOTUNA_TORCH_COMPILE=0 to disable."
        )
        _COMPILE_WARNED_ROCM = True
    compile_mode = get_torch_compile_mode()
    logger.info("torch.compile denoiser with mode={}", compile_mode)
    return cast(
        nn.Module,
        torch.compile(module, mode=compile_mode, fullgraph=True),
    )
