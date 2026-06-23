"""Centralized PrivTune environment settings (VIDEOTUNA_* prefix)."""

from __future__ import annotations

from typing import Literal

from loguru import logger
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ComputeBackendSetting = Literal["auto", "cuda", "rocm", "cpu"]
CpuModeSetting = Literal["off", "smoke", "force"]
AttnBackendSetting = Literal["auto", "flash", "sdpa", "eager"]
TorchCompileModeSetting = Literal["reduce-overhead", "max-autotune"]
MetricsOwnerSetting = Literal["script", "flow"]
MetricsBackendSetting = Literal["tensorboard", "trackio"]

ENV_PREFIX = "VIDEOTUNA_"
ENV_COMPUTE_BACKEND = f"{ENV_PREFIX}COMPUTE_BACKEND"
ENV_CPU_MODE = f"{ENV_PREFIX}CPU_MODE"
ENV_ALLOW_CPU_INFERENCE = f"{ENV_PREFIX}ALLOW_CPU_INFERENCE"
ENV_ATTN_BACKEND = f"{ENV_PREFIX}ATTN_BACKEND"
ENV_ATTN_BACKEND_STRICT = f"{ENV_PREFIX}ATTN_BACKEND_STRICT"
ENV_TORCH_COMPILE = f"{ENV_PREFIX}TORCH_COMPILE"
ENV_TORCH_COMPILE_MODE = f"{ENV_PREFIX}TORCH_COMPILE_MODE"
ENV_METRICS_OWNER = f"{ENV_PREFIX}METRICS_OWNER"
ENV_METRICS_BACKEND = f"{ENV_PREFIX}METRICS_BACKEND"
ENV_TRACKIO_SPACE_ID = f"{ENV_PREFIX}TRACKIO_SPACE_ID"
ENV_TRACKIO_PROJECT = f"{ENV_PREFIX}TRACKIO_PROJECT"
ENV_LOG_LEVEL = f"{ENV_PREFIX}LOG_LEVEL"
ENV_BENCH_MODEL = f"{ENV_PREFIX}BENCH_MODEL"

_VALID_COMPILE_MODES = ("reduce-overhead", "max-autotune")


def _parse_bool01(value: object) -> bool:
    """Parse 0/1 env strings; only ``1`` is truthy (legacy os.environ semantics)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip()
    if text == "1":
        return True
    if text == "0":
        return False
    raise ValueError(f"Expected '0' or '1', got {value!r}")


def _normalize_lower(value: object) -> object:
    if isinstance(value, str):
        return value.strip().lower()
    return value


class PrivTuneSettings(BaseSettings):
    """Load all VIDEOTUNA_* environment variables from one settings object."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    compute_backend: ComputeBackendSetting = "auto"
    cpu_mode: CpuModeSetting = "off"
    allow_cpu_inference: bool = False
    attn_backend: AttnBackendSetting = "auto"
    attn_backend_strict: bool = False
    torch_compile: bool = False
    torch_compile_mode: TorchCompileModeSetting = "reduce-overhead"
    metrics_owner: MetricsOwnerSetting = "script"
    metrics_backend: MetricsBackendSetting = "tensorboard"
    trackio_space_id: str | None = None
    trackio_project: str | None = None
    log_level: str = "INFO"
    bench_model: str | None = None

    @field_validator(
        "compute_backend",
        "attn_backend",
        "metrics_owner",
        "metrics_backend",
        mode="before",
    )
    @classmethod
    def _normalize_string_literals(cls, value: object) -> object:
        normalized = _normalize_lower(value)
        if normalized == "mps":
            raise ValueError(
                "VIDEOTUNA_COMPUTE_BACKEND=mps is not supported. "
                "PrivTune supports auto, cuda, rocm, and cpu. "
                "For config validation on Apple Silicon, use "
                "VIDEOTUNA_CPU_MODE=smoke or --cpu-smoke."
            )
        return normalized

    @field_validator("cpu_mode", mode="before")
    @classmethod
    def _normalize_cpu_mode(cls, value: object) -> object:
        if isinstance(value, str):
            text = value.strip().lower()
            return text if text else "off"
        return value

    @field_validator(
        "allow_cpu_inference",
        "attn_backend_strict",
        "torch_compile",
        mode="before",
    )
    @classmethod
    def _parse_bool01_fields(cls, value: object) -> bool:
        return _parse_bool01(value)

    @field_validator("torch_compile_mode", mode="before")
    @classmethod
    def _normalize_compile_mode(cls, value: object) -> str:
        if value is None:
            return "reduce-overhead"
        mode = str(value).strip()
        if mode in _VALID_COMPILE_MODES:
            return mode
        logger.warning(
            "Invalid {}={!r}; using reduce-overhead",
            ENV_TORCH_COMPILE_MODE,
            mode,
        )
        return "reduce-overhead"

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> str:
        if value is None:
            return "INFO"
        return str(value).strip().upper()

    @field_validator(
        "bench_model",
        "trackio_space_id",
        "trackio_project",
        mode="before",
    )
    @classmethod
    def _normalize_optional_string(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


def get_settings() -> PrivTuneSettings:
    """Return settings loaded from the current environment (no cache)."""
    return PrivTuneSettings()
