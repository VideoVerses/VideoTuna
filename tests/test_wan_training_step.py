"""CPU tests for Wan flow-matching training helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from videotuna.utils.wan_training import (
    build_i2v_mask_and_latent,
    compute_wan_flow_matching_loss,
    is_i2v_task,
)


def test_is_i2v_task():
    assert is_i2v_task("i2v-14B")
    assert not is_i2v_task("t2v-14B")


def test_build_i2v_mask_shape():
    image = torch.zeros(3, 48, 64)
    msk, video = build_i2v_mask_and_latent(
        image,
        num_frames=9,
        lat_h=6,
        lat_w=8,
        vae_stride=(4, 8, 8),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert video.shape == (3, 9, 48, 64)
    assert msk.shape[0] == 4


def test_compute_wan_flow_matching_loss_t2v_mock():
    device = torch.device("cpu")
    latent = torch.randn(16, 21, 60, 104)

    class _TinyDenoiser(torch.nn.Module):
        def forward(self, x, t, context, seq_len, y=None):
            return [torch.randn_like(x[0])]

    denoiser = _TinyDenoiser()
    flow = SimpleNamespace(
        task="t2v-14B",
        device=device,
        cfg=SimpleNamespace(
            boundary=0.875,
            num_train_timesteps=1000,
            param_dtype=torch.float32,
            sample_shift=3.0,
        ),
        low_denoiser=denoiser,
        high_denoiser=denoiser,
    )
    vae = MagicMock()
    vae.encode = lambda videos: [latent.clone() for _ in videos]
    flow.wan_t2v = SimpleNamespace(
        vae=vae,
        vae_stride=(4, 8, 8),
        patch_size=(1, 2, 2),
        text_encoder=lambda texts, dev: [torch.randn(8, 4096)],
    )
    batch = {
        "video": torch.randn(1, 3, 81, 480, 832),
        "caption": ["sks_style test"],
    }
    loss = compute_wan_flow_matching_loss(flow, batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_compute_wan_flow_matching_loss_i2v_requires_image():
    flow = SimpleNamespace(task="i2v-14B", device=torch.device("cpu"))
    flow.wan_i2v = SimpleNamespace(
        vae=MagicMock(encode=lambda v: [torch.randn(16, 21, 60, 104)]),
        vae_stride=(4, 8, 8),
        patch_size=(1, 2, 2),
        text_encoder=lambda texts, dev: [torch.randn(8, 4096)],
    )
    flow.cfg = SimpleNamespace(
        boundary=0.9,
        num_train_timesteps=1000,
        param_dtype=torch.float32,
        sample_shift=3.0,
    )
    flow.low_denoiser = flow.high_denoiser = MagicMock()
    batch = {"video": torch.randn(1, 3, 81, 480, 832), "caption": ["cap"]}
    with pytest.raises(ValueError, match="image"):
        compute_wan_flow_matching_loss(flow, batch)
