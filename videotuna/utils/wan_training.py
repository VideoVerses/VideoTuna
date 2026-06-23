"""Flow-matching training helpers for Wan T2V / I2V native Lightning LoRA."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from videotuna.schedulers.flow_matching import FlowMatchScheduler


def is_i2v_task(task: str) -> bool:
    return "i2v" in task.lower() and "t2v" not in task.lower()


def wan_pipeline_backend(flow: Any) -> Any:
    if hasattr(flow, "wan_t2v"):
        return flow.wan_t2v
    if hasattr(flow, "wan_i2v"):
        return flow.wan_i2v
    raise RuntimeError("WanVideoModelFlow has no wan_t2v or wan_i2v backend")


def _latent_grid(
    num_frames: int,
    height: int,
    width: int,
    vae_stride: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
) -> Tuple[int, int, int, int]:
    lat_t = (num_frames - 1) // vae_stride[0] + 1
    lat_h = height // vae_stride[1]
    lat_w = width // vae_stride[2]
    seq_len = lat_t * lat_h * lat_w // (patch_size[1] * patch_size[2])
    return lat_t, lat_h, lat_w, int(seq_len)


def build_i2v_mask_and_latent(
    image: torch.Tensor,
    num_frames: int,
    lat_h: int,
    lat_w: int,
    vae_stride: Tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build I2V mask + VAE input tensor matching ``WanI2V.generate``.

    ``image``: ``(C, H, W)`` in ``[-1, 1]``.
    Returns ``(mask, video_tensor)`` with ``video_tensor`` shape ``(3, F, H, W)``.
    """
    c, h, w = image.shape
    f = num_frames
    video = torch.cat(
        [
            image.unsqueeze(1),
            torch.zeros(c, f - 1, h, w, device=device, dtype=dtype),
        ],
        dim=1,
    )
    msk = torch.ones(1, f, lat_h, lat_w, device=device, dtype=dtype)
    msk[:, 1:] = 0
    msk = torch.concat(
        [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
        dim=1,
    )
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]
    return msk, video


def encode_i2v_condition(
    vae: Any,
    image: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    vae_stride: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
) -> torch.Tensor:
    """VAE-encode first-frame I2V conditioning for ``WanModel``."""
    device = image.device
    dtype = image.dtype
    _, lat_h, lat_w, _ = _latent_grid(num_frames, height, width, vae_stride, patch_size)
    if image.dim() == 4:
        image = image.squeeze(1)
    msk, video = build_i2v_mask_and_latent(
        image, num_frames, lat_h, lat_w, vae_stride, device, dtype
    )
    if video.shape[-2:] != (height, width):
        video = F.interpolate(
            video.unsqueeze(0),
            size=(num_frames, height, width),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
    encoded = vae.encode([video])[0]
    return torch.concat([msk, encoded], dim=0)


def _encode_videos(vae: Any, videos: torch.Tensor) -> List[torch.Tensor]:
    """Encode a video batch to per-sample latent tensors."""
    latents: List[torch.Tensor] = []
    for idx in range(videos.shape[0]):
        clip = videos[idx]
        latents.append(vae.encode([clip])[0])
    return latents


def _encode_prompts(text_encoder: Any, captions: Sequence[str], device: torch.device):
    if isinstance(captions, str):
        captions = [captions]
    return text_encoder(list(captions), device)


def _select_denoiser(flow: Any, timestep: torch.Tensor) -> Any:
    boundary = flow.cfg.boundary * flow.cfg.num_train_timesteps
    t_val = float(timestep.reshape(-1)[0].item())
    if t_val < boundary:
        return flow.low_denoiser
    return flow.high_denoiser


def _build_flow_scheduler(shift: float) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(
        num_inference_steps=1000,
        num_train_timesteps=1000,
        shift=shift,
    )
    scheduler.set_timesteps(1000, training=True, shift=shift)
    return scheduler


def compute_wan_flow_matching_loss(flow: Any, batch: dict) -> torch.Tensor:
    """
    Flow-matching loss for Wan T2V / I2V LoRA training.

    Expects ``batch`` keys: ``video`` ``(B,C,T,H,W)``, ``caption``, optional ``image``.
    """
    wan = wan_pipeline_backend(flow)
    device = flow.device
    dtype = flow.cfg.param_dtype

    videos = batch["video"].to(device)
    if videos.dim() != 5:
        raise ValueError(f"Expected video batch (B,C,T,H,W), got {videos.shape}")

    batch_size, _, num_frames, height, width = videos.shape
    vae_stride = tuple(wan.vae_stride)
    patch_size = tuple(wan.patch_size)
    _, lat_h, lat_w, seq_len = _latent_grid(
        num_frames, height, width, vae_stride, patch_size
    )

    with torch.no_grad():
        latents = _encode_videos(wan.vae, videos)
        contexts = _encode_prompts(wan.text_encoder, batch["caption"], device)

        y_list: Optional[List[torch.Tensor]] = None
        if is_i2v_task(flow.task):
            images = batch.get("image")
            if images is None:
                raise ValueError("I2V training requires batch['image']")
            images = images.to(device)
            y_list = []
            for idx in range(batch_size):
                img = images[idx]
                if img.dim() == 4:
                    img = img.squeeze(1)
                y_list.append(
                    encode_i2v_condition(
                        wan.vae,
                        img,
                        num_frames,
                        height,
                        width,
                        vae_stride,
                        patch_size,
                    )
                )

    shift = 3.0 if height <= 480 else float(getattr(flow.cfg, "sample_shift", 5.0))
    scheduler = _build_flow_scheduler(shift)

    losses: List[torch.Tensor] = []
    for idx in range(batch_size):
        z = latents[idx].float()
        noise = torch.randn_like(z)
        t_idx = torch.randint(0, len(scheduler.timesteps), (1,), device=device)
        timestep = scheduler.timesteps[t_idx].to(device)
        sigma = scheduler.sigmas[t_idx].to(device)
        noisy = (1.0 - sigma) * z + sigma * noise
        target = noise - z

        denoiser = _select_denoiser(flow, timestep)
        denoiser.train()
        model_input = [noisy.to(dtype)]
        context = [contexts[idx]]
        y_arg = [y_list[idx].to(dtype)] if y_list is not None else None

        autocast_enabled = device.type == "cuda"
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=autocast_enabled,
        ):
            pred_list = denoiser(
                model_input,
                t=timestep,
                context=context,
                seq_len=seq_len,
                y=y_arg,
            )
        pred = pred_list[0].float()
        weight = scheduler.training_weight(timestep).to(device)
        loss = F.mse_loss(pred, target, reduction="mean") * weight
        losses.append(loss)

    return torch.stack(losses).mean()


def init_wan_training_denoisers(flow: Any) -> None:
    """Attach PEFT LoRA to Wan low/high noise experts for dual-expert training."""
    from peft import LoraConfig, get_peft_model

    from videotuna.utils.common_utils import instantiate_from_config
    from videotuna.utils.lora_utils import (
        collect_lora_parameter_names,
        resolve_lora_target_modules,
    )

    wan = wan_pipeline_backend(flow)
    if flow.use_lora and flow.lora_config is not None:
        lora_cfg = instantiate_from_config(flow.lora_config)
        if hasattr(lora_cfg, "target_modules"):
            lora_cfg.target_modules = resolve_lora_target_modules(
                wan.high_noise_model, lora_cfg.target_modules
            )
        flow.high_denoiser = get_peft_model(
            wan.high_noise_model, lora_cfg, autocast_adapter_dtype=False
        )
        low_cfg = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            init_lora_weights=True,
            target_modules=list(lora_cfg.target_modules),
        )
        flow.low_denoiser = get_peft_model(
            wan.low_noise_model, low_cfg, autocast_adapter_dtype=False
        )
        flow.denoiser = flow.high_denoiser
        flow.lora_params = collect_lora_parameter_names(flow.denoiser)
        flow.denoiser.train()
        for name, param in flow.denoiser.named_parameters():
            if name in flow.lora_params:
                param.requires_grad_(True)
        for name, param in flow.low_denoiser.named_parameters():
            if "lora" in name.lower():
                param.requires_grad_(True)
    else:
        flow.low_denoiser = wan.low_noise_model
        flow.high_denoiser = wan.high_noise_model
        flow.denoiser = wan.high_noise_model
