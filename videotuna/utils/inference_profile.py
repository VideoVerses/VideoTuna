"""Unified inference memory preset and offload resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

MemoryPreset = Literal["low_vram", "balanced", "max_speed"]
OffloadMode = Literal["sequential", "model", "none"]


@dataclass(frozen=True)
class InferenceProfile:
    memory_preset: MemoryPreset | None
    offload_mode: OffloadMode
    enable_model_cpu_offload: bool
    enable_sequential_cpu_offload: bool
    enable_vae_tiling: bool
    dtype: str | None


def _apply_memory_preset(args: Any) -> None:
    """Mutate *args* in place to apply a named memory preset."""
    preset = getattr(args, "memory_preset", None)
    if not preset:
        return

    if preset == "low_vram":
        args.enable_sequential_cpu_offload = True
        args.enable_model_cpu_offload = False
        args.enable_vae_tiling = True
        if getattr(args, "dtype", None) is None:
            args.dtype = "fp16"
    elif preset == "balanced":
        args.enable_model_cpu_offload = True
        args.enable_sequential_cpu_offload = False
        args.enable_vae_tiling = True
        if getattr(args, "dtype", None) is None:
            args.dtype = "bf16"
    elif preset == "max_speed":
        args.enable_model_cpu_offload = False
        args.enable_sequential_cpu_offload = False
        if getattr(args, "dtype", None) is None:
            args.dtype = "bf16"
    else:
        raise ValueError(
            f"Unknown memory preset {preset!r}. "
            "Expected low_vram, balanced, or max_speed."
        )


def _offload_mode_from_args(args: Any) -> OffloadMode:
    if getattr(args, "enable_sequential_cpu_offload", False):
        return "sequential"
    if getattr(args, "enable_model_cpu_offload", False):
        return "model"
    return "none"


def _profile_from_args(args: Any) -> InferenceProfile:
    return InferenceProfile(
        memory_preset=getattr(args, "memory_preset", None),
        offload_mode=_offload_mode_from_args(args),
        enable_model_cpu_offload=bool(getattr(args, "enable_model_cpu_offload", False)),
        enable_sequential_cpu_offload=bool(
            getattr(args, "enable_sequential_cpu_offload", False)
        ),
        enable_vae_tiling=bool(getattr(args, "enable_vae_tiling", False)),
        dtype=getattr(args, "dtype", None),
    )


def resolve_inference_profile(
    args: Any, *, apply_preset: bool = True
) -> InferenceProfile:
    """Apply memory preset side effects and return the resolved profile."""
    if apply_preset:
        _apply_memory_preset(args)
    return _profile_from_args(args)
