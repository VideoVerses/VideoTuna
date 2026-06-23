#!/usr/bin/env python3
"""GPU/CPU spike: inventory native Wan LoRA keys and optional smoke inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from videotuna.testing.wan_lora_ckpt import build_synthetic_wan_lora_ckpt  # noqa: E402
from videotuna.utils.wan_lora_bridge import (  # noqa: E402
    analyze_native_wan_lora_ckpt,
    apply_native_wan_lora_to_pipeline,
    compute_remap_coverage,
    is_native_wan_lora_ckpt,
    load_native_wan_lora_state_dict,
)


def _build_synthetic_ckpt(path: Path, *, num_blocks: int = 2, rank: int = 16) -> None:
    """Write a synthetic denoiser ckpt with production-style key names."""
    build_synthetic_wan_lora_ckpt(path, num_blocks=num_blocks, rank=rank)
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=False)
    count = len(raw.get("state_dict", {}))
    print(f"Wrote synthetic checkpoint: {path} ({count} tensors)")


def _inventory_only(ckpt: Path) -> int:
    info = analyze_native_wan_lora_ckpt(ckpt)
    native = load_native_wan_lora_state_dict(ckpt)
    transformed, total, coverage = compute_remap_coverage(native)
    print(json.dumps(info, indent=2))
    print(f"Remap coverage: {transformed}/{total} keys transformed ({coverage:.1%})")
    return 0


def _load_on_pipeline(ckpt: Path, model_id: str) -> int:
    import torch
    from diffusers import WanPipeline

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Loading pipeline {model_id} (dtype={dtype})...")
    pipeline = WanPipeline.from_pretrained(model_id, torch_dtype=dtype)
    reports = apply_native_wan_lora_to_pipeline(pipeline, ckpt)
    for report in reports:
        print(json.dumps(report.as_dict(), indent=2))
        if report.loaded_lora_params == 0:
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Spike: Wan 2.1 native LoRA → 2.2 Diffusers bridge inventory/load test."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Native denoiser .ckpt path (or omit with --synthetic)",
    )
    parser.add_argument(
        "--synthetic",
        type=Path,
        metavar="PATH",
        help="Write a synthetic production-key ckpt to PATH and use it",
    )
    parser.add_argument(
        "--load-pipeline",
        action="store_true",
        help="Load Wan 2.2 pipeline and apply bridge (requires GPU + weights)",
    )
    parser.add_argument(
        "--model-id",
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="Diffusers model id for --load-pipeline",
    )
    args = parser.parse_args()

    ckpt = args.input
    if args.synthetic:
        _build_synthetic_ckpt(args.synthetic)
        ckpt = args.synthetic

    if ckpt is None:
        parser.error("Provide --input or --synthetic")

    if not is_native_wan_lora_ckpt(ckpt):
        print(f"Not a native Wan LoRA checkpoint: {ckpt}", file=sys.stderr)
        return 1

    rc = _inventory_only(ckpt)
    if args.load_pipeline:
        rc = _load_on_pipeline(ckpt, args.model_id) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
