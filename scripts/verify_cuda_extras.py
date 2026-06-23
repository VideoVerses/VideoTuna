#!/usr/bin/env python3
"""Verify NVIDIA CUDA optional dependencies and runtime environment."""

from __future__ import annotations

import argparse
import importlib
import sys

import torch
import torch.version

from videotuna.utils.device_utils import (
    _driver_version,
    describe_compute_environment,
    detect_compute_backend,
    get_visible_gpus,
)


def _check_import(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        return True, str(version)
    except ImportError as exc:
        return False, str(exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify NVIDIA CUDA extras for VideoTuna."
    )
    parser.add_argument(
        "--expect-flash",
        action="store_true",
        help="Fail when flash-attn is not importable.",
    )
    args = parser.parse_args(argv)

    errors: list[str] = []
    backend = detect_compute_backend()

    print(f"Compute backend: {backend}")
    print(describe_compute_environment())
    print(f"Driver: {_driver_version()}")
    print(f"CUDA runtime (torch): {getattr(torch.version, 'cuda', 'n/a')}")
    print(f"PyTorch: {torch.__version__}")

    if backend == "rocm":
        errors.append(
            "Active backend is ROCm; run verify on an NVIDIA CUDA install "
            "(poetry install -E cuda)."
        )
    elif backend == "cpu":
        errors.append("No GPU visible; CUDA verification requires an NVIDIA GPU.")

    gpus = get_visible_gpus()
    for gpu in gpus:
        print(
            f"  [{gpu.index}] {gpu.name}: "
            f"{gpu.total_vram_gb:.1f} GB total, "
            f"{gpu.free_vram_gb:.1f} GB free, "
            f"sm {gpu.compute_capability[0]}.{gpu.compute_capability[1]}, "
            f"bf16={gpu.supports_bf16}"
        )

    optional = ["xformers", "flash_attn", "triton", "xfuser", "bitsandbytes"]
    for pkg in optional:
        ok, detail = _check_import(pkg)
        status = "OK" if ok else "MISSING"
        print(f"  {pkg}: {status} ({detail})")
        if args.expect_flash and pkg == "flash_attn" and not ok:
            errors.append("flash-attn not installed (--expect-flash)")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print("CUDA extras verification OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
