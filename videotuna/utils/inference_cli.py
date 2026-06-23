"""Shared CLI helpers for VideoTuna inference entrypoints."""

from __future__ import annotations

import os
from typing import Any, Optional

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from videotuna.settings import (
    ENV_ATTN_BACKEND,
    ENV_CPU_MODE,
    ENV_TORCH_COMPILE,
    get_settings,
)
from videotuna.utils.inference_profile import resolve_inference_profile


def apply_compile_env(compile_flag: bool) -> None:
    """Set VIDEOTUNA_TORCH_COMPILE before model load when --compile is passed."""
    if get_settings().cpu_mode == "smoke":
        os.environ[ENV_TORCH_COMPILE] = "0"
        return
    os.environ[ENV_TORCH_COMPILE] = "1" if compile_flag else "0"


def apply_cpu_smoke_env(args: Any) -> None:
    """Set environment for CPU smoke mode from --cpu-smoke."""
    if not getattr(args, "cpu_smoke", False):
        return
    os.environ[ENV_CPU_MODE] = "smoke"
    os.environ[ENV_ATTN_BACKEND] = "eager"
    os.environ[ENV_TORCH_COMPILE] = "0"


def validate_cpu_offload_flags(args: Any) -> None:
    """Reject GPU VRAM offload on CPU inference; resolve dual offload flag conflicts."""
    from videotuna.utils.device_utils import (
        detect_compute_backend,
        gpu_is_available,
        resolve_cpu_mode,
    )

    if getattr(args, "enable_sequential_cpu_offload", False) and getattr(
        args, "enable_model_cpu_offload", False
    ):
        logger.warning(
            "Both --enable_sequential_cpu_offload and --enable_model_cpu_offload "
            "were set; using sequential CPU offload (ignoring model offload)."
        )
        args.enable_model_cpu_offload = False

    cpu_mode = resolve_cpu_mode(cli_smoke=getattr(args, "cpu_smoke", False))
    device = (getattr(args, "device", None) or "").strip().lower()
    cpu_inference = (
        cpu_mode in ("smoke", "force")
        or device == "cpu"
        or detect_compute_backend() == "cpu"
        or not gpu_is_available()
    )
    if not cpu_inference:
        return

    offload = (
        getattr(args, "enable_sequential_cpu_offload", False)
        or getattr(args, "enable_model_cpu_offload", False)
        or getattr(args, "memory_preset", None) == "low_vram"
    )
    if offload:
        raise RuntimeError(
            "CPU offload flags (--enable_model_cpu_offload, "
            "--enable_sequential_cpu_offload, --memory-preset low_vram) "
            "require a GPU accelerator to stage weights. "
            "They are not CPU-only inference modes.\n"
            "Install a GPU stack (poetry install --extras cuda) or run "
            "without offload flags."
        )


def apply_cpu_smoke_limits(
    inference_config: DictConfig,
    flow_config: Optional[DictConfig] = None,
) -> None:
    """Cap resolution, frames, and steps for CPU smoke runs."""
    caps = {
        "frames": 2,
        "height": 256,
        "width": 256,
        "num_inference_steps": 4,
        "ddim_steps": 4,
    }
    for key, cap in caps.items():
        current = getattr(inference_config, key, None)
        if current is not None and int(current) > cap:
            logger.warning("CPU smoke: capping {} from {} to {}", key, current, cap)
            inference_config[key] = cap
        elif current is None and key in ("num_inference_steps", "ddim_steps"):
            inference_config[key] = cap

    if getattr(inference_config, "device", None) is None:
        inference_config.device = "cpu"
    if getattr(inference_config, "dtype", None) is None:
        inference_config.dtype = "fp32"

    if flow_config is not None:
        params = flow_config.get("params", OmegaConf.create())
        if OmegaConf.select(params, "height") and int(params.height) > caps["height"]:
            params.height = caps["height"]
        if OmegaConf.select(params, "width") and int(params.width) > caps["width"]:
            params.width = caps["width"]


def resolve_offload_mode(args) -> str:
    """Return offload mode string from parsed args."""
    return resolve_inference_profile(args, apply_preset=False).offload_mode


def prepare_cli_inference_args(args: Any) -> Any:
    """Apply smoke env and validate parallel degrees before config merge."""
    apply_cpu_smoke_env(args)
    ulysses = getattr(args, "ulysses_degree", None)
    ring = getattr(args, "ring_degree", None)
    if ulysses is not None or ring is not None:
        from videotuna.utils.device_utils import validate_sequence_parallel_degrees

        validate_sequence_parallel_degrees(ulysses, ring)
    return args
