"""Diffusers pipeline quantization for Wan 2.2 / Flux inference."""

from __future__ import annotations

import importlib.metadata
from typing import Any, Literal, Optional

import torch
from loguru import logger
from packaging.version import Version

from videotuna.utils.device_utils import detect_compute_backend, gpu_is_available

TransformerQuant = Literal["none", "int8_wo", "int4_wo", "fp8_wo"]
QuantBackend = Literal["torchao", "quanto"]

TRANSFORMER_QUANT_CHOICES = ("none", "int8_wo", "int4_wo", "fp8_wo")
QUANT_BACKEND_CHOICES = ("torchao", "quanto")

_FP8_MIN_COMPUTE_CAPABILITY = (8, 9)
_TORCHAO_MIN_VERSION = "0.15.0"


def normalize_transformer_quant(value: Optional[str]) -> str:
    """Return a validated transformer_quant string (default ``none``)."""
    if value is None or value == "":
        return "none"
    quant = str(value).lower()
    if quant not in TRANSFORMER_QUANT_CHOICES:
        raise ValueError(
            f"Unsupported transformer_quant {value!r}. "
            f"Expected one of: {', '.join(TRANSFORMER_QUANT_CHOICES)}"
        )
    return quant


def normalize_quant_backend(value: Optional[str]) -> str:
    """Return a validated quant_backend string (default ``torchao``)."""
    if value is None or value == "":
        return "torchao"
    backend = str(value).lower()
    if backend not in QUANT_BACKEND_CHOICES:
        raise ValueError(
            f"Unsupported quant_backend {value!r}. "
            f"Expected one of: {', '.join(QUANT_BACKEND_CHOICES)}"
        )
    return backend


def resolve_quant_components(
    model_family: str,
    model_variant: Optional[str],
    mode: str,
) -> list[str]:
    """Pipeline submodules to quantize for the given Diffusers model family."""
    family = model_family.lower()
    variant = (model_variant or "").strip()
    if family == "wan" and variant == "2.2" and mode.lower() in ("t2v", "i2v"):
        return ["transformer", "transformer_2"]
    if family in ("wan", "flux"):
        return ["transformer"]
    return ["transformer"]


def _require_cuda_for_quant(transformer_quant: str) -> None:
    backend = detect_compute_backend()
    if backend == "cpu":
        raise RuntimeError(
            f"transformer_quant={transformer_quant!r} is not supported on CPU. "
            "Use transformer_quant: none or --cpu-smoke without quantization."
        )
    if backend == "rocm":
        raise RuntimeError(
            f"transformer_quant={transformer_quant!r} is not supported on AMD ROCm. "
            "Use memory_preset low_vram with offload instead."
        )
    if backend != "cuda":
        raise RuntimeError(
            f"transformer_quant={transformer_quant!r} requires NVIDIA CUDA; "
            f"detected backend {backend!r}."
        )


def _require_quanto() -> None:
    try:
        import optimum.quanto  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "quant_backend=quanto requires the quanto extra. "
            "Install with: poetry install -E quanto"
        ) from exc


def _require_torchao_min_version(min_version: str = _TORCHAO_MIN_VERSION) -> None:
    try:
        import torchao  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"quant_backend=torchao requires torchao>={min_version}. "
            "Install with: poetry install"
        ) from exc
    installed = importlib.metadata.version("torchao")
    if Version(installed) < Version(min_version):
        raise ImportError(
            f"quant_backend=torchao requires torchao>={min_version}; "
            f"found {installed}. Run: poetry update torchao"
        )


def _require_fp8_gpu() -> None:
    if not gpu_is_available():
        return
    major, minor = torch.cuda.get_device_capability(0)
    min_major, min_minor = _FP8_MIN_COMPUTE_CAPABILITY
    if (major, minor) < (min_major, min_minor):
        raise RuntimeError(
            f"transformer_quant=fp8_wo requires NVIDIA GPU compute capability >= "
            f"{min_major}.{min_minor} (Ada/Hopper); detected {major}.{minor}. "
            "Use int8_wo or int4_wo on older GPUs."
        )
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "transformer_quant=fp8_wo requires torch.float8_e4m3fn (PyTorch 2.6+)."
        )


def validate_transformer_quant(
    *,
    transformer_quant: Optional[str],
    quant_backend: Optional[str],
    offload_mode: str,
    compile_enabled: bool = False,
    has_lora: bool = False,
) -> str:
    """
    Validate quant settings before pipeline load.

    Returns the normalized transformer_quant value.
    """
    quant = normalize_transformer_quant(transformer_quant)
    if quant == "none":
        return quant

    backend = normalize_quant_backend(quant_backend)
    _require_cuda_for_quant(quant)
    if quant == "fp8_wo":
        _require_fp8_gpu()

    if backend == "quanto":
        _require_quanto()
    else:
        _require_torchao_min_version()

    if offload_mode == "sequential":
        logger.warning(
            "transformer_quant={} with sequential CPU offload may be slow or "
            "incompatible; model CPU offload is preferred when quantizing.",
            quant,
        )
    if compile_enabled and offload_mode != "none":
        logger.warning(
            "transformer_quant={} with CPU offload disables torch.compile on "
            "the transformer (compile applies only when offload is none).",
            quant,
        )
    if has_lora:
        logger.info(
            "Attempting native LoRA load on quantized transformers; "
            "use transformer_quant: none if PEFT bridge fails."
        )
    return quant


def maybe_adjust_offload_for_quant(args: Any, transformer_quant: str) -> None:
    """Prefer model offload over sequential when quant is enabled (mutates args)."""
    if transformer_quant == "none":
        return
    if getattr(args, "enable_sequential_cpu_offload", False):
        logger.info(
            "transformer_quant enabled: switching sequential CPU offload to "
            "model CPU offload for Diffusers quant compatibility"
        )
        args.enable_sequential_cpu_offload = False
        args.enable_model_cpu_offload = True


def _build_torchao_component_config(transformer_quant: str) -> Any:
    from diffusers import TorchAoConfig

    if transformer_quant == "int8_wo":
        from torchao.quantization import Int8WeightOnlyConfig, PerGroup

        return TorchAoConfig(Int8WeightOnlyConfig(granularity=PerGroup(128), version=2))
    if transformer_quant == "int4_wo":
        from torchao.quantization import Int4WeightOnlyConfig

        return TorchAoConfig(Int4WeightOnlyConfig(group_size=128, version=2))
    if transformer_quant == "fp8_wo":
        from torchao.quantization import Float8WeightOnlyConfig

        return TorchAoConfig(Float8WeightOnlyConfig())
    raise ValueError(f"Unsupported torchao quant scheme: {transformer_quant}")


def _build_quanto_component_config(transformer_quant: str) -> Any:
    _require_quanto()
    from diffusers import QuantoConfig

    mapping = {
        "int8_wo": "int8",
        "int4_wo": "int4",
        "fp8_wo": "float8",
    }
    weights_dtype = mapping.get(transformer_quant)
    if weights_dtype is None:
        raise ValueError(
            f"quanto backend does not support transformer_quant={transformer_quant!r}"
        )
    return QuantoConfig(weights_dtype=weights_dtype)


def build_pipeline_quantization_config(
    *,
    transformer_quant: str,
    quant_backend: str,
    components: list[str],
) -> Optional[Any]:
    """Build a Diffusers PipelineQuantizationConfig or None when quant is disabled."""
    quant = normalize_transformer_quant(transformer_quant)
    if quant == "none":
        return None

    backend = normalize_quant_backend(quant_backend)
    from diffusers import PipelineQuantizationConfig

    if backend == "torchao":
        component_cfg = _build_torchao_component_config(quant)
    else:
        component_cfg = _build_quanto_component_config(quant)

    quant_mapping = {name: component_cfg for name in components}
    logger.info(
        "Diffusers quant: scheme={} backend={} components={}",
        quant,
        backend,
        list(quant_mapping.keys()),
    )
    return PipelineQuantizationConfig(quant_mapping=quant_mapping)
