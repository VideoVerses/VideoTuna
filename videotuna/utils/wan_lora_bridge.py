"""Bridge Wan 2.1 native Lightning LoRA checkpoints onto Wan 2.2 Diffusers pipelines."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict

# Native Wan 2.1 PEFT targets (training config in wan_t2v_lora.yaml).
WAN_NATIVE_LORA_TARGETS = ["q", "k", "v", "o", "ffn.0", "ffn.2"]

# Diffusers WanTransformer3DModel PEFT targets (attn1 self-attn + FFN only).
WAN_DIFFUSERS_LORA_TARGETS = [
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
]

DEFAULT_LORA_RANK = 16
HIGH_NOISE_ADAPTER = "domain_high"
LOW_NOISE_ADAPTER = "domain_low"

_SELF_ATTN_RE = re.compile(
    r"^blocks\.(\d+)\.self_attn\.(q|k|v|o)\.(lora_[AB]\.weight)$"
)
_FFN0_RE = re.compile(r"^blocks\.(\d+)\.ffn\.0\.(lora_[AB]\.weight)$")
_FFN2_RE = re.compile(r"^blocks\.(\d+)\.ffn\.2\.(lora_[AB]\.weight)$")
# Legacy / test shorthand: blocks.N.attn.q
_LEGACY_ATTN_RE = re.compile(r"^blocks\.(\d+)\.attn\.(q|k|v|o)\.(lora_[AB]\.weight)$")


@dataclass
class WanLoraLoadReport:
    """Structured result from loading native LoRA onto a Diffusers transformer."""

    expert: str
    rank: int
    source_keys: int
    remapped_keys: int
    loaded_lora_params: int
    missing_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[str] = field(default_factory=list)

    @property
    def remap_ratio(self) -> float:
        if self.source_keys == 0:
            return 0.0
        return self.remapped_keys / self.source_keys

    def as_dict(self) -> Dict[str, Any]:
        return {
            "expert": self.expert,
            "rank": self.rank,
            "source_keys": self.source_keys,
            "remapped_keys": self.remapped_keys,
            "loaded_lora_params": self.loaded_lora_params,
            "remap_ratio": round(self.remap_ratio, 4),
            "missing_keys": len(self.missing_keys),
            "unexpected_keys": len(self.unexpected_keys),
        }


def is_native_wan_lora_ckpt(path: str | Path) -> bool:
    """Return True when path looks like a Wan native Lightning LoRA .ckpt."""
    p = Path(path)
    if not p.is_file():
        return False
    if p.suffix not in (".ckpt", ".pt", ".pth"):
        return False
    try:
        state = load_native_wan_lora_state_dict(p)
    except (OSError, RuntimeError, ValueError):
        return False
    return bool(state) and any("lora" in k.lower() for k in state)


def load_native_wan_lora_state_dict(ckpt_path: str | Path) -> Dict[str, torch.Tensor]:
    """Load and normalize LoRA tensors from a Wan training checkpoint."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    elif isinstance(raw, dict):
        state_dict = raw
    else:
        raise ValueError(f"Unexpected checkpoint type in {ckpt_path}")

    lora_state: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if "lora" not in key.lower():
            continue
        normalized = key
        for prefix in ("denoiser.", "model.", "module.", "base_model.model."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        lora_state[normalized] = tensor
    if not lora_state:
        raise ValueError(f"No LoRA keys found in {ckpt_path}")
    return lora_state


def _infer_lora_rank(state_dict: Dict[str, torch.Tensor]) -> int:
    for key, tensor in state_dict.items():
        if key.endswith(".lora_A.weight") or ".lora_A." in key:
            return int(tensor.shape[0])
        if key.endswith(".lora_B.weight") or ".lora_B." in key:
            return int(tensor.shape[1])
    return DEFAULT_LORA_RANK


def _remap_single_native_key(key: str) -> str:
    """Map one native Wan 2.1 LoRA key to Diffusers WanTransformer3DModel naming."""
    m = _SELF_ATTN_RE.match(key)
    if m:
        idx, proj, suffix = m.groups()
        diff_proj = {"q": "to_q", "k": "to_k", "v": "to_v", "o": "to_out.0"}[proj]
        return f"blocks.{idx}.attn1.{diff_proj}.{suffix}"

    m = _LEGACY_ATTN_RE.match(key)
    if m:
        idx, proj, suffix = m.groups()
        diff_proj = {"q": "to_q", "k": "to_k", "v": "to_v", "o": "to_out.0"}[proj]
        return f"blocks.{idx}.attn1.{diff_proj}.{suffix}"

    m = _FFN0_RE.match(key)
    if m:
        return f"blocks.{m.group(1)}.ffn.net.0.proj.{m.group(2)}"

    m = _FFN2_RE.match(key)
    if m:
        return f"blocks.{m.group(1)}.ffn.net.2.{m.group(2)}"

    # Back-compat: blocks.* -> transformer_blocks.* (older bridge attempt).
    if key.startswith("blocks."):
        return "transformer_blocks." + key[len("blocks.") :]
    return key


def _remap_native_to_diffusers_keys(
    native_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Remap native Wan 2.1 block names to Diffusers WanTransformer3DModel keys."""
    remapped: Dict[str, torch.Tensor] = {}
    for key, tensor in native_state.items():
        remapped[_remap_single_native_key(key)] = tensor
    return remapped


def compute_remap_coverage(
    native_state: Dict[str, torch.Tensor],
) -> Tuple[int, int, float]:
    """Return transformed key count, total keys, and coverage ratio."""
    if not native_state:
        return 0, 0, 0.0
    remapped = _remap_native_to_diffusers_keys(native_state)
    transformed = sum(1 for key in native_state if key != remapped.get(key))
    total = len(native_state)
    return transformed, total, transformed / total


def analyze_native_wan_lora_ckpt(ckpt_path: str | Path) -> Dict[str, Any]:
    """Inventory native checkpoint keys and remapped Diffusers targets."""
    native_state = load_native_wan_lora_state_dict(ckpt_path)
    remapped = _remap_native_to_diffusers_keys(native_state)
    unchanged = [k for k in native_state if k == remapped.get(k)]
    return {
        "path": str(ckpt_path),
        "rank": _infer_lora_rank(native_state),
        "native_key_count": len(native_state),
        "remapped_key_count": len(remapped),
        "unchanged_keys": unchanged[:10],
        "sample_native": sorted(native_state.keys())[:5],
        "sample_remapped": sorted(remapped.keys())[:5],
    }


def _peft_prefix_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        k if k.startswith("base_model.model.") else f"base_model.model.{k}": v
        for k, v in state.items()
    }


def _count_lora_params(module: Any) -> int:
    return sum(1 for name, _ in module.named_parameters() if "lora" in name.lower())


def _apply_lora_to_transformer(
    transformer: Any,
    remapped_state: Dict[str, torch.Tensor],
    *,
    rank: int,
    adapter_name: str,
    expert_label: str,
    source_keys: int,
    remapped_keys: int,
) -> Tuple[Any, WanLoraLoadReport]:
    """Inject PEFT LoRA adapters and load remapped weights onto one transformer."""
    if not hasattr(transformer, "peft_config") or not transformer.peft_config:
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights=True,
            target_modules=WAN_DIFFUSERS_LORA_TARGETS,
        )
        transformer = get_peft_model(
            transformer, lora_config, adapter_name=adapter_name
        )

    peft_state = _peft_prefix_keys(remapped_state)
    result = set_peft_model_state_dict(
        transformer,
        peft_state,
        adapter_name=adapter_name,
    )
    missing: List[str] = []
    unexpected: List[str] = []
    if hasattr(result, "missing_keys"):
        missing = [k for k in result.missing_keys if "lora" in k.lower()]
        unexpected = list(result.unexpected_keys)

    loaded = _count_lora_params(transformer)
    report = WanLoraLoadReport(
        expert=expert_label,
        rank=rank,
        source_keys=source_keys,
        remapped_keys=remapped_keys,
        loaded_lora_params=loaded,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )
    if unexpected:
        logger.warning(
            "Wan LoRA bridge [{}]: {} unexpected keys (first 5): {}",
            expert_label,
            len(unexpected),
            unexpected[:5],
        )
    if missing:
        logger.warning(
            "Wan LoRA bridge [{}]: {} missing LoRA keys (first 5): {}",
            expert_label,
            len(missing),
            missing[:5],
        )
    logger.info("Wan LoRA bridge [{}]: {}", expert_label, report.as_dict())
    return transformer, report


def apply_native_wan_lora_to_pipeline(
    pipeline: Any,
    ckpt_path: str | Path,
    *,
    lora_scale: float = 1.0,
    lora_scale_2: Optional[float] = None,
) -> List[WanLoraLoadReport]:
    """
    Attach Wan 2.1 native LoRA weights to a Wan 2.2 Diffusers pipeline.

    Loads the same adapter onto ``transformer`` (high-noise) and ``transformer_2``
    (low-noise) when both are present, matching Diffusers community practice.
    """
    native_state = load_native_wan_lora_state_dict(ckpt_path)
    rank = _infer_lora_rank(native_state)
    remapped = _remap_native_to_diffusers_keys(native_state)
    remapped_keys, source_keys, _ = compute_remap_coverage(native_state)
    scale_2 = lora_scale if lora_scale_2 is None else lora_scale_2

    reports: List[WanLoraLoadReport] = []
    adapters: List[str] = []
    scales: List[float] = []

    pipeline.transformer, report_high = _apply_lora_to_transformer(
        pipeline.transformer,
        remapped,
        rank=rank,
        adapter_name=HIGH_NOISE_ADAPTER,
        expert_label="transformer",
        source_keys=source_keys,
        remapped_keys=remapped_keys,
    )
    reports.append(report_high)
    adapters.append(HIGH_NOISE_ADAPTER)
    scales.append(lora_scale)

    transformer_2 = getattr(pipeline, "transformer_2", None)
    if transformer_2 is not None:
        pipeline.transformer_2, report_low = _apply_lora_to_transformer(
            transformer_2,
            remapped,
            rank=rank,
            adapter_name=LOW_NOISE_ADAPTER,
            expert_label="transformer_2",
            source_keys=source_keys,
            remapped_keys=remapped_keys,
        )
        reports.append(report_low)
        adapters.append(LOW_NOISE_ADAPTER)
        scales.append(scale_2)

    total_loaded = sum(r.loaded_lora_params for r in reports)
    if total_loaded == 0:
        raise RuntimeError(
            f"Wan LoRA bridge failed to load any parameters from {ckpt_path}. "
            "Run tools/spike_wan_lora_bridge.py for a key inventory."
        )

    if hasattr(pipeline, "set_adapters"):
        pipeline.set_adapters(adapters, adapter_weights=scales)
    elif hasattr(pipeline, "fuse_lora"):
        pipeline.fuse_lora(lora_scale=lora_scale)

    min_remap = min(r.remap_ratio for r in reports)
    if min_remap < 0.9 and remapped:
        logger.warning(
            "Wan LoRA bridge: remap ratio {:.1%} below 90% — visual QA recommended",
            min_remap,
        )

    logger.info(
        "Wan LoRA bridge: rank={} experts={} total_lora_params={} scales={}",
        rank,
        [r.expert for r in reports],
        total_loaded,
        scales,
    )
    return reports


def apply_native_wan_lora_to_i2v_pipeline(
    pipeline: Any,
    ckpt_path: str | Path,
    *,
    lora_scale: float = 1.0,
    lora_scale_2: Optional[float] = None,
) -> List[WanLoraLoadReport]:
    """
    Attach Wan 2.1 native I2V LoRA weights to a Wan 2.2 I2V Diffusers pipeline.

    Uses the same block-level key remap as T2V; both transformer experts receive
    identical adapter weights when ``transformer_2`` is present.
    """
    return apply_native_wan_lora_to_pipeline(
        pipeline,
        ckpt_path,
        lora_scale=lora_scale,
        lora_scale_2=lora_scale_2,
    )


def export_diffusers_lora_state_dicts(
    ckpt_path: str | Path,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Export remapped LoRA tensors for Diffusers ``load_lora_weights``.

    Returns a dict with ``high_noise`` and optionally ``low_noise`` entries
    (same weights; Wan 2.2 loads low-noise expert via ``load_into_transformer_2``).
    """
    native_state = load_native_wan_lora_state_dict(ckpt_path)
    remapped = _remap_native_to_diffusers_keys(native_state)
    exports = {"high_noise": remapped, "low_noise": dict(remapped)}
    return exports


def strip_peft_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove PEFT prefixes for safetensors export."""
    cleaned: Dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        k = key
        for prefix in ("base_model.model.", "base_model."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
        cleaned[k] = tensor
    return cleaned
