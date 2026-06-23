"""Optional 4-bit loading for frozen text encoders via bitsandbytes + accelerate."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def build_transformers_quant_config(load_in_4bit: bool = True) -> Optional[Any]:
    """Return a transformers BitsAndBytesConfig for 4-bit loading, or None."""
    if not load_in_4bit:
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "4-bit loading requires transformers with BitsAndBytesConfig support"
        ) from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def apply_quantization_to_config_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inject quantization kwargs into a model config params dict when load_in_4bit is set.

    Supports transformers from_pretrained-style configs.
    """
    if not params.get("load_in_4bit", False):
        return params

    updated = dict(params)
    quant_config = build_transformers_quant_config(True)
    if quant_config is not None:
        updated["quantization_config"] = quant_config
    updated.setdefault("torch_dtype", torch.bfloat16)
    updated.setdefault("device_map", "auto")
    return updated
