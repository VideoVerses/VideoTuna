"""Accelerate training loop for Flux LoRA fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from videotuna.settings import get_settings
from videotuna.training.flux_lora.checkpoint import (
    checkpoint_step,
    find_latest_checkpoint,
    load_lora_checkpoint,
    prune_checkpoints,
    save_lora_checkpoint,
)
from videotuna.training.flux_lora.config import (
    FluxLoraDataConfig,
    FluxLoraTrainConfig,
    load_train_config,
    stamp_output_dir,
)
from videotuna.training.flux_lora.dataset import (
    FluxBucketBatchSampler,
    FluxLoraImageDataset,
    _load_caption,
)
from videotuna.training.flux_lora.model_utils import load_flux_training_models
from videotuna.training.flux_lora.text_embed_cache import build_or_load_cache
from videotuna.utils.logging_config import bound_logger, resolve_device_label
from videotuna.utils.training_metrics import (
    DEFAULT_FLUX_TRACKIO_PROJECT,
    build_trackio_init_kwargs,
    describe_metrics_backend,
    log_validation_image_to_trackio,
    resolve_accelerate_log_with,
    trackio_enabled,
)

logger = bound_logger(phase="t2i", flow="flux_lora")

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def create_flux_accelerator(
    output_dir: Path,
    *,
    mixed_precision: str,
    gradient_accumulation_steps: int = 1,
    metrics_backend: str | None = None,
) -> Accelerator:
    """Build an Accelerate instance with TensorBoard (and optional Trackio) tracking."""
    if metrics_backend is None:
        metrics_backend = get_settings().metrics_backend
    project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(output_dir / "tensorboard"),
    )
    return Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=resolve_accelerate_log_with(metrics_backend),
        project_config=project_config,
    )


def _flux_tracker_config(config: FluxLoraTrainConfig) -> dict[str, Any]:
    return {
        "lora_rank": config.lora_rank,
        "learning_rate": config.learning_rate,
        "max_train_steps": config.max_train_steps,
        "resolution": config.resolution,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "pretrained_model_name_or_path": config.pretrained_model_name_or_path,
    }


def _apply_runtime_flags(config: FluxLoraTrainConfig) -> None:
    if config.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = not config.disable_benchmark


def _parse_validation_resolution(value: str) -> tuple[int, int]:
    if "x" in value.lower():
        width_str, height_str = value.lower().split("x", 1)
        return int(width_str), int(height_str)
    size = int(value)
    return size, size


def _collect_captions(data_config: FluxLoraDataConfig) -> list[str]:
    data_dir = Path(data_config.instance_data_dir)
    captions: list[str] = []
    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        captions.append(
            _load_caption(
                path,
                data_config.caption_strategy,
                data_config.default_caption,
            )
        )
    return captions


def _resolve_resume_checkpoint(
    config: FluxLoraTrainConfig, output_dir: Path
) -> Path | None:
    if not config.resume_from_checkpoint:
        return None
    if config.resume_from_checkpoint == "latest":
        return find_latest_checkpoint(output_dir)
    candidate = Path(config.resume_from_checkpoint)
    if not candidate.is_absolute():
        candidate = output_dir / candidate
    return candidate if candidate.is_dir() else None


def _create_optimizer(
    transformer, config: FluxLoraTrainConfig
) -> torch.optim.Optimizer:
    if config.optimizer not in {"adamw", "adamw_bf16"}:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    params = transformer.parameters()
    kwargs = {
        "lr": config.learning_rate,
        "betas": (0.9, 0.999),
        "weight_decay": 1e-4,
        "eps": 1e-8,
    }
    if config.optimizer == "adamw_bf16":
        from optimi import AdamW as OptimiAdamW

        return OptimiAdamW(params, **kwargs)
    return torch.optim.AdamW(params, **kwargs)


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    captions = [item["caption"] for item in batch]
    collated: dict[str, Any] = {"pixel_values": pixel_values, "caption": captions}
    if batch and "prompt_embeds" in batch[0]:
        collated["prompt_embeds"] = torch.cat(
            [item["prompt_embeds"] for item in batch], dim=0
        )
        collated["pooled_prompt_embeds"] = torch.cat(
            [item["pooled_prompt_embeds"] for item in batch], dim=0
        )
        collated["text_ids"] = torch.cat([item["text_ids"] for item in batch], dim=0)
    return collated


def _prepare_batch_latents(vae, pixel_values, weight_dtype):
    pixel_values = pixel_values.to(dtype=weight_dtype)
    latents = vae.encode(pixel_values).latent_dist.sample()
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    batch_size, num_channels, height, width = latents.shape
    packed = FluxPipeline._pack_latents(
        latents, batch_size, num_channels, height, width
    )
    return packed, height, width


def _compute_loss(
    pipeline: Any,
    transformer,
    batch,
    weight_dtype,
    accelerator,
) -> torch.Tensor:
    pixel_values = batch["pixel_values"]
    captions = batch["caption"]
    if isinstance(captions, str):
        captions = [captions]

    with torch.no_grad():
        if "prompt_embeds" in batch:
            prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
            text_ids = batch["text_ids"].to(accelerator.device)
        else:
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt=captions,
                prompt_2=captions,
                device=accelerator.device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )
        model_input, latent_height, latent_width = _prepare_batch_latents(
            pipeline.vae, pixel_values, weight_dtype
        )
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        u = torch.rand(bsz, device=accelerator.device)
        sigmas = u
        timesteps = (sigmas * pipeline.scheduler.config.num_train_timesteps).long()
        sigmas = sigmas.view(-1, 1, 1)
        noisy_input = (1.0 - sigmas) * model_input + sigmas * noise
        target = noise - model_input

        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            bsz,
            latent_height // 2,
            latent_width // 2,
            accelerator.device,
            weight_dtype,
        )

    guidance = torch.tensor([1.0], device=accelerator.device, dtype=weight_dtype)
    guidance = guidance.expand(model_input.shape[0])

    model_pred = transformer(
        hidden_states=noisy_input,
        timestep=timesteps / 1000,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        return_dict=False,
    )[0]

    return F.mse_loss(model_pred.float(), target.float(), reduction="mean")


def _run_validation(
    pipeline: FluxPipeline,
    config: FluxLoraTrainConfig,
    output_dir: Path,
    global_step: int,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    log,
    *,
    metrics_backend: str,
) -> None:
    if not config.validation_prompt or not config.validation_steps:
        return
    if global_step % config.validation_steps != 0:
        return
    if not accelerator.is_main_process:
        return

    width, height = _parse_validation_resolution(config.validation_resolution)
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(
        config.validation_seed
    )
    pipeline.transformer.eval()
    with torch.inference_mode():
        result = pipeline(
            prompt=config.validation_prompt,
            height=height,
            width=width,
            num_inference_steps=config.validation_num_inference_steps,
            guidance_scale=config.validation_guidance,
            generator=generator,
        )
    pipeline.transformer.train()
    image = result.images[0]
    image_path = validation_dir / f"step-{global_step:06d}.png"
    image.save(image_path)
    log.info("Saved validation image to {}", image_path)

    tracker = accelerator.trackers[0] if accelerator.trackers else None
    if tracker is not None and hasattr(tracker, "writer"):
        import numpy as np

        array = np.array(image.convert("RGB")).transpose(2, 0, 1)
        tracker.writer.add_image(
            "validation/sample",
            array,
            global_step,
            dataformats="CHW",
        )

    if trackio_enabled(metrics_backend):
        log_validation_image_to_trackio(image, global_step)


def _build_dataloader(
    dataset: FluxLoraImageDataset,
    config: FluxLoraTrainConfig,
) -> DataLoader:
    if config.train_batch_size > 1:
        batch_sampler = FluxBucketBatchSampler(
            dataset.bucket_ids,
            config.train_batch_size,
            shuffle=True,
            seed=config.seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate_batch,
        )
    return DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_batch,
    )


def _build_embed_lookup(
    pipeline: FluxPipeline,
    data_config: FluxLoraDataConfig,
    config: FluxLoraTrainConfig,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    if not data_config.text_embeds or data_config.text_embeds.disabled:
        return {}
    captions = _collect_captions(data_config)
    write_batch_size = (
        data_config.text_embeds.write_batch_size or config.write_batch_size
    )
    return build_or_load_cache(
        pipeline,
        captions,
        data_config.text_embeds.cache_dir,
        write_batch_size,
        device,
    )


def _run_training_loop(
    *,
    config: FluxLoraTrainConfig,
    output_dir: Path,
    pipeline: FluxPipeline,
    transformer,
    dataloader: DataLoader,
    optimizer,
    lr_scheduler,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    global_step: int,
    max_train_steps: int,
    log,
    metrics_backend: str,
) -> None:
    progress = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_main_process,
        desc="Flux LoRA",
        initial=global_step,
        total=max_train_steps,
    )

    while global_step < max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(transformer):
                loss = _compute_loss(
                    pipeline,
                    transformer,
                    batch,
                    weight_dtype,
                    accelerator,
                )
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}", step=global_step)
            accelerator.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                },
                step=global_step,
            )
            _run_validation(
                pipeline,
                config,
                output_dir,
                global_step,
                accelerator,
                weight_dtype,
                log,
                metrics_backend=metrics_backend,
            )
            if (
                global_step % config.checkpointing_steps == 0
                or global_step == max_train_steps
            ) and accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(transformer)
                ckpt = save_lora_checkpoint(unwrapped, output_dir, global_step)
                prune_checkpoints(output_dir, config.checkpoints_total_limit)
                log.info("Saved LoRA checkpoint to {}", ckpt)
            if global_step >= max_train_steps:
                break

    progress.close()


def train(config: FluxLoraTrainConfig, data_config: FluxLoraDataConfig) -> None:
    set_seed(config.seed)
    _apply_runtime_flags(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    metrics_backend = settings.metrics_backend

    accelerator = create_flux_accelerator(
        output_dir,
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        metrics_backend=metrics_backend,
    )
    log = logger.bind(device=resolve_device_label(accelerator.device))
    if accelerator.is_main_process:
        log.info("Training Flux LoRA → {}", output_dir)
        log.info("Metrics backend: {}", describe_metrics_backend(metrics_backend))

    components = load_flux_training_models(
        config.pretrained_model_name_or_path,
        lora_rank=config.lora_rank,
        mixed_precision=config.mixed_precision,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    weight_dtype = components["weight_dtype"]
    transformer = components["transformer"]

    global_step = 0
    resume_path = _resolve_resume_checkpoint(config, output_dir)
    if resume_path is not None:
        load_lora_checkpoint(transformer, resume_path)
        global_step = checkpoint_step(resume_path)
        log.info("Resumed LoRA weights from {} (step {})", resume_path, global_step)
    elif config.resume_from_checkpoint:
        raise ValueError(
            f"No checkpoint found for resume_from_checkpoint="
            f"{config.resume_from_checkpoint!r} under output_dir={output_dir}. "
            "Remove resume_from_checkpoint or set it to null to start fresh, "
            "or point output_dir at a run that contains checkpoint-* directories."
        )

    pipeline = FluxPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=components["vae"],
        text_encoder=components["text_encoder_one"],
        text_encoder_2=components["text_encoder_two"],
        tokenizer=components["tokenizer_one"],
        tokenizer_2=components["tokenizer_two"],
        transformer=transformer,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    pipeline.vae.to(accelerator.device)
    pipeline.text_encoder.to(accelerator.device)
    pipeline.text_encoder_2.to(accelerator.device)

    embed_lookup = _build_embed_lookup(
        pipeline, data_config, config, accelerator.device
    )
    dataset = FluxLoraImageDataset(
        data_config,
        embed_lookup=embed_lookup,
        seed=config.seed,
    )
    dataloader = _build_dataloader(dataset, config)

    optimizer = _create_optimizer(transformer, config)
    opt_impl = (
        "optimi.AdamW" if config.optimizer == "adamw_bf16" else "torch.optim.AdamW"
    )
    log.info("Using optimizer {} ({})", config.optimizer, opt_impl)
    max_train_steps = config.max_train_steps
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    pipeline.transformer = accelerator.unwrap_model(transformer)
    if global_step > 0:
        for _ in range(global_step):
            lr_scheduler.step()
        log.info(
            "Advanced LR scheduler to step {} (optimizer state not restored)",
            global_step,
        )
    trackio_project = settings.trackio_project or DEFAULT_FLUX_TRACKIO_PROJECT
    trackio_init_kwargs = build_trackio_init_kwargs(space_id=settings.trackio_space_id)
    accelerator.init_trackers(
        trackio_project,
        config=_flux_tracker_config(config),
        init_kwargs=trackio_init_kwargs,
    )

    _run_training_loop(
        config=config,
        output_dir=output_dir,
        pipeline=pipeline,
        transformer=transformer,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        weight_dtype=weight_dtype,
        global_step=global_step,
        max_train_steps=max_train_steps,
        log=log,
        metrics_backend=metrics_backend,
    )

    accelerator.end_training()
    if accelerator.is_main_process:
        model_path = config.pretrained_model_name_or_path
        with open(output_dir / "training_config.json", "w") as f:
            json.dump(
                {
                    "pretrained_model_name_or_path": model_path,
                    "lora_rank": config.lora_rank,
                    "max_train_steps": config.max_train_steps,
                    "resolution": config.resolution,
                },
                f,
                indent=2,
            )
        log.info("Training finished. Output: {}", output_dir)


def run_training(
    config_path: str, data_config_path: str, stamp_output: bool = True
) -> None:
    train_cfg, data_cfg = load_train_config(config_path, data_config_path)
    if stamp_output and not train_cfg.resume_from_checkpoint:
        train_cfg.output_dir = stamp_output_dir(train_cfg.output_dir)
        with open(config_path) as f:
            raw = json.load(f)
        raw["--output_dir"] = train_cfg.output_dir
        with open(config_path, "w") as f:
            json.dump(raw, f, indent=4)
    train(train_cfg, data_cfg)
