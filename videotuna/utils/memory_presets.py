"""Named memory/performance presets for inference CLI."""

from __future__ import annotations

from typing import Any

from videotuna.utils.inference_profile import MemoryPreset, resolve_inference_profile

__all__ = ["MemoryPreset", "apply_memory_preset"]


def apply_memory_preset(args: Any) -> None:
    """Mutate *args* in place to apply a named memory preset."""
    resolve_inference_profile(args)
