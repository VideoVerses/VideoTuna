#!/usr/bin/env python3
"""Verify CPU-only PyTorch install for VideoTuna dev/CI."""

from __future__ import annotations

import argparse
import importlib
import sys

import torch
from torch import version as torch_version

from videotuna.utils.device_utils import (
    describe_compute_environment,
    detect_compute_backend,
)


def _check_import(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "installed")
        return True, str(version)
    except ImportError:
        return False, "not installed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify CPU-only PyTorch install for VideoTuna."
    )
    parser.parse_args(argv)

    errors: list[str] = []
    backend = detect_compute_backend()

    print(f"Compute backend: {backend}")
    print(describe_compute_environment())
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {getattr(torch_version, 'cuda', None)}")
    print(f"HIP build: {getattr(torch_version, 'hip', None)}")

    if getattr(torch_version, "cuda", None) is not None:
        errors.append("PyTorch was built with CUDA; run: poetry run install-cpu-torch")
    if getattr(torch_version, "hip", None) is not None:
        errors.append("PyTorch reports HIP (ROCm); expected CPU-only wheel.")

    if backend != "cpu":
        errors.append(
            f"Expected detect_compute_backend()=cpu, got {backend!r}. "
            "Set VIDEOTUNA_COMPUTE_BACKEND=cpu or use a CPU torch wheel."
        )

    cuda_only = ["xformers", "xfuser", "bitsandbytes"]
    for pkg in cuda_only:
        ok, detail = _check_import(pkg)
        status = "PRESENT" if ok else "absent"
        print(f"  {pkg}: {status} ({detail})")
        if ok:
            errors.append(
                f"CUDA-only package {pkg} is installed; "
                "re-run poetry run install-cpu-torch"
            )

    triton_ok, triton_detail = _check_import("triton")
    print(f"  triton: {'PRESENT' if triton_ok else 'absent'} ({triton_detail})")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print("CPU torch verification OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
