"""Typed cyclopts option groups for inference entrypoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from typing import Annotated, Any, Literal

from cyclopts import Parameter

from videotuna.utils.inference_profile import MemoryPreset

DtypeChoice = Literal["bf16", "fp16", "fp32"]
DeviceMapChoice = Literal["auto"]


@Parameter(name="*")
@dataclass
class StandardInferenceOptions:
    """Memory, device, and performance flags shared by all inference commands."""

    cpu_smoke: Annotated[bool | None, Parameter(name="cpu-smoke")] = None
    device: Annotated[str | None, Parameter(name="device", alias="--gpu-id")] = None
    min_vram_gb: Annotated[float | None, Parameter(name="min-vram-gb")] = None
    memory_preset: Annotated[MemoryPreset | None, Parameter(name="memory-preset")] = (
        None
    )
    enable_vae_tiling: Annotated[bool | None, Parameter(name="enable_vae_tiling")] = (
        None
    )
    enable_vae_slicing: Annotated[bool | None, Parameter(name="enable_vae_slicing")] = (
        None
    )
    enable_model_cpu_offload: Annotated[
        bool | None, Parameter(name="enable_model_cpu_offload")
    ] = None
    enable_sequential_cpu_offload: Annotated[
        bool | None, Parameter(name="enable_sequential_cpu_offload")
    ] = None
    dtype: DtypeChoice | None = None
    device_map: Annotated[DeviceMapChoice | None, Parameter(name="device-map")] = None
    ulysses_degree: int | None = None
    ring_degree: int | None = None
    compile: Annotated[bool | None, Parameter(name="compile")] = None
    fuse_qkv: bool | None = None
    enable_attention_cache: bool | None = None
    transformer_quant: Annotated[
        str | None,
        Parameter(
            name="transformer-quant",
            help=(
                "Diffusers transformer weight-only quant: "
                "none, int8_wo, int4_wo, fp8_wo (CUDA; fp8_wo needs Ada/Hopper+)."
            ),
        ),
    ] = None
    quant_backend: Annotated[str | None, Parameter(name="quant-backend")] = None


@Parameter(name="*")
@dataclass
class InferenceRunOptions:
    """Model, prompt, and sampling flags for inference."""

    mode: str | None = None
    ckpt_path: str | None = None
    lorackpt: str | None = None
    trained_ckpt: str | None = None
    config: str | None = None
    prompt_file: str | None = None
    prompt_dir: str | None = None
    savedir: str | None = None
    standard_vbench: bool | None = None
    seed: int | None = None
    height: int | None = None
    width: int | None = None
    frames: int | None = None
    fps: int | None = None
    n_samples_prompt: int | None = None
    bs: int | None = None
    ddim_steps: int | None = None
    ddim_eta: float | None = None
    uncond_prompt: str | None = None
    unconditional_guidance_scale: float | None = None
    unconditional_guidance_scale_temporal: float | None = None
    multiple_cond_cfg: bool | None = None
    cfg_img: float | None = None
    timestep_spacing: str | None = None
    guidance_rescale: float | None = None
    loop: bool | None = None
    gfi: bool | None = None
    savefps: str | None = None
    time_shift: float | None = None
    num_inference_steps: int | None = None
    i2v_resolution: str | None = None
    lora_rank: int | None = None


@dataclass(frozen=True)
class InferencePreset:
    """Baked-in defaults for a Poetry inference entrypoint."""

    cli_name: str
    config: str
    enable_model_cpu_offload: bool = False
    require_checkpoint: bool = False
    require_prompt_dir: bool = False


def _non_null_values(options: Any) -> dict[str, Any]:
    return {field.name: getattr(options, field.name) for field in fields(options)}


def inference_options_to_namespace(
    *,
    run: InferenceRunOptions | None = None,
    standard: StandardInferenceOptions | None = None,
    preset: InferencePreset | None = None,
) -> argparse.Namespace:
    """Flatten typed option groups into a namespace for legacy inference code."""
    merged: dict[str, Any] = {}

    if preset is not None:
        merged["config"] = preset.config
        if preset.enable_model_cpu_offload:
            merged["enable_model_cpu_offload"] = True

    for key, value in _non_null_values(run or InferenceRunOptions()).items():
        if value is not None:
            merged[key] = value

    for key, value in _non_null_values(standard or StandardInferenceOptions()).items():
        if value is not None:
            merged[key] = value

    if merged.get("cpu_smoke") is None:
        merged["cpu_smoke"] = False

    bool_defaults = (
        "enable_vae_tiling",
        "enable_vae_slicing",
        "enable_model_cpu_offload",
        "enable_sequential_cpu_offload",
        "compile",
        "fuse_qkv",
        "enable_attention_cache",
    )
    for key in bool_defaults:
        if key not in merged:
            merged[key] = False

    return argparse.Namespace(**merged)


def validate_preset_requirements(
    run: InferenceRunOptions,
    preset: InferencePreset,
) -> None:
    """Enforce checkpoint / prompt requirements for validation entrypoints."""
    import sys

    if preset.require_checkpoint and not (run.trained_ckpt or run.lorackpt):
        print(
            f"Error: {preset.cli_name} requires --trained_ckpt <denoiser.ckpt> "
            "or --lorackpt <checkpoint-dir>",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if preset.require_prompt_dir and not run.prompt_dir:
        print(
            f"Error: {preset.cli_name} requires --prompt_dir <image+prompt pairs>",
            file=sys.stderr,
        )
        raise SystemExit(2)
