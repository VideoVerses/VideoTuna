"""Tests for ROCm-safe attention backend selection."""

import os
from unittest import mock

import pytest

from videotuna.utils import attention


def test_auto_backend_cpu_fallback_eager():
    with mock.patch.object(attention, "detect_compute_backend", return_value="cpu"):
        with mock.patch.object(attention, "gpu_is_available", return_value=False):
            with mock.patch.dict(os.environ, {"VIDEOTUNA_ATTN_BACKEND": "auto"}):
                assert attention.get_attn_backend() == "eager"


def test_flash_rejected_on_cpu():
    with mock.patch.object(attention, "detect_compute_backend", return_value="cpu"):
        with mock.patch.dict(os.environ, {"VIDEOTUNA_ATTN_BACKEND": "flash"}):
            with pytest.raises(RuntimeError, match="not supported on CPU"):
                attention.get_attn_backend()


def test_maybe_compile_noop_without_gpu():
    import torch.nn as nn

    mod = nn.Linear(4, 4)
    with mock.patch.object(attention, "gpu_is_available", return_value=False):
        with mock.patch.dict(os.environ, {"VIDEOTUNA_TORCH_COMPILE": "1"}):
            assert attention.maybe_compile_denoiser(mod) is mod


def test_auto_backend_rocm_prefers_sdpa():
    with mock.patch.object(attention, "detect_compute_backend", return_value="rocm"):
        with mock.patch.object(attention, "gpu_is_available", return_value=True):
            with mock.patch.dict(os.environ, {"VIDEOTUNA_ATTN_BACKEND": "auto"}):
                assert attention.get_attn_backend() == "sdpa"


def test_auto_backend_rocm_cpu_fallback_eager():
    with mock.patch.object(attention, "detect_compute_backend", return_value="rocm"):
        with mock.patch.object(attention, "gpu_is_available", return_value=False):
            with mock.patch.dict(os.environ, {"VIDEOTUNA_ATTN_BACKEND": "auto"}):
                assert attention.get_attn_backend() == "eager"


def test_flash_rejected_on_rocm():
    with mock.patch.object(attention, "detect_compute_backend", return_value="rocm"):
        with mock.patch.object(attention, "_FLASH_ATTN_AVAILABLE", True):
            with mock.patch.dict(os.environ, {"VIDEOTUNA_ATTN_BACKEND": "flash"}):
                with pytest.raises(RuntimeError, match="not supported on AMD ROCm"):
                    attention.get_attn_backend()


def test_auto_backend_cuda_uses_flash_when_available():
    with mock.patch.object(attention, "detect_compute_backend", return_value="cuda"):
        with mock.patch.object(attention, "gpu_is_available", return_value=True):
            with mock.patch.object(attention, "_FLASH_ATTN_AVAILABLE", True):
                with mock.patch.dict(os.environ, {"VIDEOTUNA_ATTN_BACKEND": "auto"}):
                    assert attention.get_attn_backend() == "flash"


def test_sdpa_context_rocm_excludes_flash_kernel():
    with mock.patch.object(attention, "gpu_is_available", return_value=True):
        with mock.patch.object(
            attention, "detect_compute_backend", return_value="rocm"
        ):
            with mock.patch("torch.nn.attention.sdpa_kernel") as mock_sdpa:
                mock_sdpa.return_value.__enter__ = mock.Mock(return_value=None)
                mock_sdpa.return_value.__exit__ = mock.Mock(return_value=False)
                with attention._sdpa_context():
                    pass
                backends = mock_sdpa.call_args[0][0]
                backend_names = [b.name for b in backends]
                assert "FLASH_ATTENTION" not in backend_names
                assert "EFFICIENT_ATTENTION" in backend_names
