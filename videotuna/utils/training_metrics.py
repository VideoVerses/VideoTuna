"""Training experiment tracking helpers (TensorBoard + optional Trackio)."""

from __future__ import annotations

import importlib.util
from typing import Any

MetricsBackend = str

DEFAULT_FLUX_TRACKIO_PROJECT = "flux-domain-lora"


def trackio_available() -> bool:
    return importlib.util.find_spec("trackio") is not None


def require_trackio() -> None:
    if trackio_available():
        return
    raise ImportError(
        "VIDEOTUNA_METRICS_BACKEND=trackio requires the trackio extra. "
        "Install with: poetry install -E trackio"
    )


def trackio_enabled(metrics_backend: MetricsBackend) -> bool:
    return metrics_backend == "trackio"


def resolve_accelerate_log_with(metrics_backend: MetricsBackend) -> str | list[str]:
    if metrics_backend == "trackio":
        require_trackio()
        return ["tensorboard", "trackio"]
    return "tensorboard"


def describe_metrics_backend(metrics_backend: MetricsBackend) -> str:
    if trackio_enabled(metrics_backend):
        return "tensorboard + trackio"
    return "tensorboard"


def build_trackio_init_kwargs(
    *,
    space_id: str | None = None,
) -> dict[str, dict[str, Any]] | None:
    if not space_id:
        return None
    return {"trackio": {"space_id": space_id}}


def log_validation_image_to_trackio(image: Any, step: int) -> None:
    require_trackio()
    import trackio

    trackio.log({"validation/sample": trackio.Image(image)}, step=step)
