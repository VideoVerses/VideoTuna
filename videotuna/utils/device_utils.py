"""Device detection and inference hardware requirements."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Literal

import torch
import torch.version
from loguru import logger

from videotuna.settings import (
    ENV_ALLOW_CPU_INFERENCE,
    ENV_CPU_MODE,
    get_settings,
)

ComputeBackend = Literal["cuda", "rocm", "cpu"]
InferenceDtype = Literal["bf16", "fp16", "fp32"]
FlowCapabilityTier = Literal["cpu_ok", "cpu_smoke", "gpu_required"]
CpuMode = Literal["off", "smoke", "force"]

_DIFFUSERS_FLOW = "videotuna.flow.diffusers_video.DiffusersVideoFlow"
_WAN_FLOW = "videotuna.flow.wanvideo.WanVideoModelFlow"

FLOW_TIERS: dict[str, FlowCapabilityTier] = {
    _DIFFUSERS_FLOW: "cpu_smoke",
    _WAN_FLOW: "gpu_required",
}


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_vram_gb: float
    free_vram_gb: float
    compute_capability: tuple[int, int]
    supports_bf16: bool


def _torch_hip_version() -> str | None:
    hip = getattr(torch.version, "hip", None)
    if hip is None:
        return None
    return str(hip)


def _detect_compute_backend_raw() -> ComputeBackend:
    if not torch.cuda.is_available():
        return "cpu"
    if _torch_hip_version() is not None:
        return "rocm"
    return "cuda"


def detect_compute_backend() -> ComputeBackend:
    """Return the active compute backend (cuda, rocm, or cpu)."""
    requested = get_settings().compute_backend
    if requested == "auto":
        return _detect_compute_backend_raw()
    if requested == "cpu":
        return "cpu"
    if requested == "rocm":
        if _torch_hip_version() is None:
            raise RuntimeError(
                f"VIDEOTUNA_COMPUTE_BACKEND=rocm but PyTorch was not built with HIP. "
                f"Detected: {describe_compute_environment()}\n"
                "Install with: poetry install --extras rocm"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "VIDEOTUNA_COMPUTE_BACKEND=rocm but no ROCm GPU is visible. "
                "Check ROCm driver and HIP_VISIBLE_DEVICES."
            )
        return "rocm"
    # requested == "cuda"
    if _torch_hip_version() is not None:
        raise RuntimeError(
            f"VIDEOTUNA_COMPUTE_BACKEND=cuda but PyTorch reports HIP "
            f"({_torch_hip_version()}). "
            "Use VIDEOTUNA_COMPUTE_BACKEND=rocm or install the CUDA PyTorch wheel."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "VIDEOTUNA_COMPUTE_BACKEND=cuda but torch.cuda.is_available() " "is False."
        )
    return "cuda"


def gpu_is_available() -> bool:
    """True when an accelerator GPU is available (NVIDIA CUDA or AMD ROCm)."""
    return torch.cuda.is_available()


def cuda_is_available() -> bool:
    """Deprecated alias for gpu_is_available()."""
    return gpu_is_available()


def accelerator_device_string() -> str:
    """PyTorch device type string for GPU autocast/offload (cuda for CUDA/ROCm)."""
    return "cuda" if gpu_is_available() else "cpu"


def normalize_device_prefer(prefer: str | int | None) -> str | None:
    """Accept cpu, cuda, cuda:0, cuda:1, 0, 1 → canonical device string."""
    if prefer is None:
        return None
    if isinstance(prefer, int):
        return f"cuda:{prefer}"
    text = str(prefer).strip().lower()
    if not text:
        return None
    if text == "mps":
        raise ValueError(
            f"Invalid device {prefer!r}. MPS is not supported. "
            "PrivTune supports cpu, cuda, and cuda:N. "
            "For config validation on Apple Silicon, use "
            "VIDEOTUNA_CPU_MODE=smoke or --cpu-smoke."
        )
    if text == "cpu":
        return text
    if text.isdigit():
        return f"cuda:{int(text)}"
    if text == "cuda":
        return "cuda"
    if re.match(r"^cuda:\d+$", text):
        return text
    raise ValueError(
        f"Invalid device {prefer!r}. Expected cpu, cuda, cuda:N, "
        "or an integer GPU index."
    )


def resolve_cpu_mode(*, cli_smoke: bool = False) -> CpuMode:
    """Resolve CPU inference mode from CLI flag, env, or legacy allow_cpu."""
    if cli_smoke:
        return "smoke"
    settings = get_settings()
    mode: CpuMode = settings.cpu_mode
    if settings.allow_cpu_inference:
        logger.warning(
            "{} is deprecated; use {}=force or --cpu-smoke instead.",
            ENV_ALLOW_CPU_INFERENCE,
            ENV_CPU_MODE,
        )
        return "force"
    return mode


def _is_production_video_resolution(
    height: int | None,
    width: int | None,
) -> bool:
    """True when H×W matches production video presets (720p-class)."""
    if height is None or width is None:
        return False
    return (height >= 720 or width >= 1280) or (height >= 480 and width >= 720)


def _is_init_smoke_resolution(
    height: int | None,
    width: int | None,
    *,
    frames: int | None = None,
) -> bool:
    """Tiny resolution for native-flow init-only CPU smoke (not full denoise)."""
    if height is None or width is None:
        return False
    if height > 256 or width > 256:
        return False
    if frames is not None and frames > 2:
        return False
    return True


def _diffusers_flow_tier(
    family: str,
    variant: str,
    height: int | None,
    width: int | None,
    base: FlowCapabilityTier,
) -> FlowCapabilityTier:
    """CPU tier for DiffusersVideoFlow from model family and resolution."""
    if family == "flux" and variant in ("schnell", "1-schnell"):
        return "cpu_smoke"
    if family == "flux" and variant in ("1-dev", "dev"):
        if height is not None and height >= 512:
            return "gpu_required"
        return "cpu_smoke"
    if family == "wan":
        if _is_production_video_resolution(height, width):
            return "gpu_required"
        return "cpu_smoke"
    if _is_production_video_resolution(height, width):
        return "gpu_required"
    return base


def get_flow_tier(
    flow_target: str,
    *,
    model_family: str | None = None,
    model_variant: str | None = None,
    height: int | None = None,
    width: int | None = None,
) -> FlowCapabilityTier:
    """Return the CPU capability tier for a flow target and optional model hints."""
    base = FLOW_TIERS.get(flow_target, "cpu_ok")
    if flow_target != _DIFFUSERS_FLOW:
        return base

    family = (model_family or "").lower()
    variant = (model_variant or "").lower()
    return _diffusers_flow_tier(family, variant, height, width, base)


def _validate_cuda_device_index(index: int) -> None:
    if not gpu_is_available():
        raise RuntimeError(
            f"Requested CUDA device index {index} but no GPU accelerator is available."
        )
    count = torch.cuda.device_count()
    if index < 0 or index >= count:
        raise RuntimeError(
            f"Invalid CUDA device index {index}. "
            f"Visible GPU count is {count} (after CUDA_VISIBLE_DEVICES remapping)."
        )


def resolve_inference_device(prefer: str | int | None = None) -> torch.device:
    """Pick the best available torch device for inference."""
    if detect_compute_backend() == "cpu" and prefer is None:
        return torch.device("cpu")

    normalized = normalize_device_prefer(prefer)
    if normalized:
        device = torch.device(normalized)
        if device.type == "cuda":
            if not gpu_is_available():
                raise RuntimeError(
                    f"Requested device {prefer!r} but no GPU accelerator is available."
                )
            index = device.index if device.index is not None else 0
            _validate_cuda_device_index(index)
            torch.cuda.set_device(index)
            return torch.device("cuda", index)
        return device
    if gpu_is_available() and detect_compute_backend() != "cpu":
        torch.cuda.set_device(0)
        return torch.device("cuda", 0)
    return torch.device("cpu")


def get_visible_gpus() -> list[GpuInfo]:
    """Enumerate visible CUDA/ROCm devices with VRAM and compute capability."""
    if not gpu_is_available():
        return []
    gpus: list[GpuInfo] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        major, minor = props.major, props.minor
        gpus.append(
            GpuInfo(
                index=index,
                name=props.name,
                total_vram_gb=total_bytes / (1024**3),
                free_vram_gb=free_bytes / (1024**3),
                compute_capability=(major, minor),
                supports_bf16=major >= 8,
            )
        )
    return gpus


def recommend_dtype(device: torch.device) -> InferenceDtype:
    """CPU → fp32; Ampere+ (sm >= 8.0) → bf16; older NVIDIA GPUs → fp16."""
    if device.type == "cpu":
        return "fp32"
    if device.type != "cuda" or not gpu_is_available():
        return "fp16"
    index = device.index if device.index is not None else 0
    major, _minor = torch.cuda.get_device_capability(index)
    if major >= 8:
        return "bf16"
    return "fp16"


def require_min_vram(
    gb: float,
    *,
    device: torch.device | None = None,
    context: str = "",
) -> None:
    """Fail fast when selected GPU total VRAM is below *gb*."""
    if not gpu_is_available():
        raise RuntimeError(
            _format_hardware_context(context)
            + "No GPU accelerator is available for VRAM check."
        )
    dev = device or resolve_inference_device()
    if dev.type != "cuda":
        return
    index = dev.index if dev.index is not None else 0
    props = torch.cuda.get_device_properties(index)
    total_gb = props.total_memory / (1024**3)
    if total_gb < gb:
        prefix = _format_hardware_context(context, device_index=index)
        raise RuntimeError(
            f"{prefix}"
            f"GPU total VRAM {total_gb:.1f} GB is below required {gb:.1f} GB.\n"
            "Next steps:\n"
            "  - Use --memory-preset low_vram or --enable_sequential_cpu_offload\n"
            "  - Lower resolution or frame count in the config\n"
            "  - Select a GPU with more VRAM via --device / CUDA_VISIBLE_DEVICES"
        )


def _cuda_runtime_version() -> str:
    cuda_ver = getattr(torch.version, "cuda", None)
    return str(cuda_ver) if cuda_ver else "unknown"


def _driver_version() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _format_hardware_context(
    context: str = "",
    *,
    device_index: int = 0,
) -> str:
    lines: list[str] = []
    if context:
        lines.append(context.strip())
        if not lines[-1].endswith("."):
            lines[-1] += "."
    if gpu_is_available():
        props = torch.cuda.get_device_properties(device_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        lines.append(
            f"  GPU: {props.name} "
            f"({total_bytes / (1024**3):.1f} GB total, "
            f"{free_bytes / (1024**3):.1f} GB free)"
        )
        lines.append(
            f"  Driver: {_driver_version()} / "
            f"CUDA runtime: {_cuda_runtime_version()} / "
            f"PyTorch: {torch.__version__}"
        )
    else:
        lines.append(f"  Detected: {describe_compute_environment()}")
    return "\n".join(lines) + "\n"


def log_startup_device_summary(
    device: torch.device,
    dtype: str | None,
    attn_backend: str,
    offload_mode: str,
    *,
    attn_backend_requested: str | None = None,
    memory_preset: str | None = None,
    compile_enabled: bool = False,
    compile_mode: str | None = None,
) -> None:
    """Emit a single structured startup log for inference."""
    gpu_name = "CPU"
    if device.type == "cuda" and gpu_is_available():
        index = device.index if device.index is not None else 0
        gpu_name = torch.cuda.get_device_name(index)
    requested = attn_backend_requested or attn_backend
    resolved_note = f" (resolved {attn_backend})" if requested != attn_backend else ""
    preset_note = f", preset={memory_preset}" if memory_preset else ""
    compile_note = ""
    if compile_enabled:
        compile_note = f", compile={compile_mode or 'reduce-overhead'}"
    logger.info(
        "Inference startup: device={} gpu={} dtype={} attention={}{} offload={}{}{}",
        device,
        gpu_name,
        dtype or "auto",
        requested,
        resolved_note,
        offload_mode,
        preset_note,
        compile_note,
    )


def empty_accelerator_cache() -> None:
    if gpu_is_available():
        torch.cuda.empty_cache()


def synchronize_accelerator() -> None:
    if gpu_is_available():
        torch.cuda.synchronize()


# NVIDIA-oriented aliases (ROCm uses the same torch.cuda API).
empty_cache = empty_accelerator_cache
synchronize_device = synchronize_accelerator


def describe_compute_environment() -> str:
    backend = _detect_compute_backend_raw()
    if backend == "rocm":
        name = torch.cuda.get_device_name(0)
        hip = _torch_hip_version() or "unknown"
        return f"ROCm available ({name}, torch {torch.__version__}, HIP {hip})"
    if backend == "cuda":
        name = torch.cuda.get_device_name(0)
        return f"CUDA available ({name}, torch {torch.__version__})"
    return "No GPU accelerator (CPU-only PyTorch or no GPU driver)"


def snapshot_nvidia_smi() -> str | None:
    """Best-effort nvidia-smi snapshot for failure diagnostics."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _tiered_cpu_error_message(
    flow_target: str,
    tier: FlowCapabilityTier,
    cpu_mode: CpuMode,
) -> str:
    lines = [
        f"This inference command requires a GPU (tier={tier}, cpu_mode={cpu_mode}).\n",
        _format_hardware_context(f"Flow: {flow_target}"),
        "Install options:\n"
        "  - NVIDIA: poetry install --extras cuda\n"
        "  - AMD ROCm: poetry install --extras rocm (see docs/install-rocm.md)\n",
        "What you can do without a GPU:\n"
        "  - Unit tests: poetry run pytest tests/ -m 'not gpu'\n"
        "  - CPU smoke presets: configs/inference/presets/*_cpu_smoke.yaml\n"
        "  - Full matrix: docs/capability-matrix.md\n",
    ]
    if tier == "gpu_required":
        lines.append(
            "  - Debug init only (≤256px, ≤2 frames): --cpu-smoke with a tiny preset\n"
            f"  - Full override (not recommended): {ENV_CPU_MODE}=force\n"
        )
    elif tier == "cpu_smoke" and cpu_mode == "off":
        lines.append(f"  - Enable CPU smoke: --cpu-smoke or {ENV_CPU_MODE}=smoke\n")
    lines.append("See docs/install-cpu.md and docs/capability-matrix.md.")
    return "".join(lines)


def require_accelerator_for_flow(
    flow_target: str,
    *,
    min_vram_gb: float | None = None,
    allow_cpu: bool = False,
    cpu_mode: CpuMode | None = None,
    tier: FlowCapabilityTier | None = None,
    model_family: str | None = None,
    model_variant: str | None = None,
    height: int | None = None,
    width: int | None = None,
    frames: int | None = None,
) -> None:
    """
    Fail fast when a GPU-backed video flow is started without an accelerator.

    Passes when a CUDA or ROCm GPU is available, or when CPU mode permits the tier.
    """
    if allow_cpu:
        logger.warning(
            "allow_cpu=True is deprecated; use --cpu-smoke or VIDEOTUNA_CPU_MODE=force"
        )
        cpu_mode = "force"

    resolved_tier = tier or get_flow_tier(
        flow_target,
        model_family=model_family,
        model_variant=model_variant,
        height=height,
        width=width,
    )
    mode = cpu_mode if cpu_mode is not None else resolve_cpu_mode()

    if resolved_tier == "cpu_ok":
        return

    backend = detect_compute_backend()

    if gpu_is_available() and backend != "cpu":
        logger.info("Inference device: {}", describe_compute_environment())
        if min_vram_gb is not None:
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024**3)
            if total_gb < min_vram_gb:
                logger.warning(
                    "GPU VRAM {:.1f} GB is below recommended {:.1f} GB for {}",
                    total_gb,
                    min_vram_gb,
                    flow_target,
                )
        return

    if mode == "force":
        logger.warning(
            "CPU force mode: skipping GPU requirement for {} (tier={}); "
            "not suitable for production inference",
            flow_target,
            resolved_tier,
        )
        return

    if mode == "smoke" and resolved_tier == "cpu_smoke":
        logger.warning(
            "CPU smoke mode: {} tier=cpu_smoke — tiny resolution/steps only",
            flow_target,
        )
        return

    if (
        mode == "smoke"
        and resolved_tier == "gpu_required"
        and flow_target == _WAN_FLOW
        and _is_init_smoke_resolution(height, width, frames=frames)
    ):
        logger.warning(
            "CPU init smoke: {} at {}x{} (frames={}) — checkpoint load only, "
            "not production 720p denoise",
            flow_target,
            height,
            width,
            frames,
        )
        return

    raise RuntimeError(_tiered_cpu_error_message(flow_target, resolved_tier, mode))


def require_nvidia_cuda_for_flow(
    flow_target: str,
    *,
    allow_cpu: bool = False,
    **kwargs: object,
) -> None:
    """Deprecated alias for require_accelerator_for_flow."""
    require_accelerator_for_flow(flow_target, allow_cpu=allow_cpu, **kwargs)  # type: ignore[arg-type]


def require_xfuser_sequence_parallel(flow_name: str) -> None:
    """Fail when xfuser USP is requested on ROCm (CUDA-only dependency)."""
    if detect_compute_backend() == "rocm":
        raise RuntimeError(
            f"Sequence parallel (ulysses_degree / ring_degree) is not supported on "
            f"AMD ROCm for {flow_name}. xfuser requires NVIDIA CUDA.\n"
            "Use single-GPU inference with VIDEOTUNA_ATTN_BACKEND=sdpa instead."
        )


def validate_sequence_parallel_degrees(
    ulysses_degree: int | None,
    ring_degree: int | None,
    *,
    world_size: int | None = None,
) -> None:
    """Validate xfuser USP degree product matches visible process count."""
    u = ulysses_degree or 1
    r = ring_degree or 1
    if u <= 1 and r <= 1:
        return
    product = u * r
    if world_size is None:
        try:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError:
            world_size = 1
    if world_size != product:
        raise ValueError(
            f"ulysses_degree ({u}) × ring_degree ({r}) = {product} but "
            f"WORLD_SIZE={world_size}. "
            "Launch with torchrun --nproc_per_node=N where N equals the product."
        )


def checkpoints_exist(path: str | None) -> bool:
    if not path:
        return False
    from pathlib import Path

    p = Path(path)
    return p.exists() and (p.is_dir() or p.is_file())


def looks_like_hf_model_id(path: str) -> bool:
    """True for org/model repo ids that are not local paths."""
    if not path or path.startswith(("/", "./", "../")):
        return False
    from pathlib import Path

    if Path(path).exists():
        return False
    parts = path.replace("\\", "/").split("/")
    return len(parts) == 2 and all(parts) and " " not in path


def checkpoint_available(path: str | None, *, flow_target: str = "") -> bool:
    """Local checkpoint exists, or path is a Hugging Face model id."""
    if not path:
        return True
    if checkpoints_exist(path):
        return True
    if "diffusers_video" in flow_target and looks_like_hf_model_id(path):
        return True
    return looks_like_hf_model_id(path)
