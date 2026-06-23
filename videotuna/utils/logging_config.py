"""Central loguru configuration with structured phase/flow/device context."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from videotuna.settings import get_settings

if TYPE_CHECKING:
    import torch

DEFAULT_EXTRA = {"phase": "-", "flow": "-", "device": "-"}

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "phase={extra[phase]} flow={extra[flow]} device={extra[device]} | "
    "{message}"
)

_configured = False
_stderr_handler_id: int | None = None
_file_handler_id: int | None = None


def configure_logging(
    *,
    log_file: Path | str | None = None,
    level: str | None = None,
) -> None:
    """Configure loguru once per process (stderr + optional file sink)."""
    global _configured, _stderr_handler_id, _file_handler_id

    log_level = (level or get_settings().log_level).upper()
    logger.configure(extra=DEFAULT_EXTRA)
    logger.remove()

    _stderr_handler_id = logger.add(
        sys.stderr,
        level=log_level,
        format=_LOG_FORMAT,
    )
    _file_handler_id = None
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _file_handler_id = logger.add(
            str(log_path),
            level=log_level,
            format=_LOG_FORMAT,
            mode="w",
        )

    _configured = True


def bound_logger(*, phase: str, flow: str, device: str = "-"):
    """Return a loguru logger with structured context fields."""
    return logger.bind(phase=phase, flow=flow, device=device)


def resolve_device_label(device: torch.device | str | None = None) -> str:
    """Return a compact device label for log context."""
    if device is None:
        return "-"
    if isinstance(device, str):
        return device
    index_suffix = ""
    if device.index is not None:
        index_suffix = f":{device.index}"
    return f"{device.type}{index_suffix}"


def phase_from_wan_task(task: str) -> str:
    """Map Wan task id to logging phase."""
    if task.startswith("i2v"):
        return "i2v"
    if task.startswith("t2i"):
        return "t2i"
    if task.startswith("t2v"):
        return "t2v"
    return "inference"
