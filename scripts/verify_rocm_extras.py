#!/usr/bin/env python3
"""Verify pyproject.toml ROCm extra excludes CUDA-only packages."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

CUDA_ONLY_IN_ROCM = {
    "xformers",
    "bitsandbytes",
    "xfuser",
    "triton",
}
CUDA_ONLY_PREFIXES = ("nvidia-",)


def main() -> int:
    data = tomllib.loads(PYPROJECT.read_text())
    poetry = data.get("tool", {}).get("poetry", {})
    extras = poetry.get("extras", {})
    rocm_extra = set(extras.get("rocm", []))
    cuda_extra = set(extras.get("cuda", []))

    errors: list[str] = []

    overlap = rocm_extra & cuda_extra
    if overlap:
        errors.append(f"rocm and cuda extras overlap: {sorted(overlap)}")

    for pkg in rocm_extra:
        if pkg in CUDA_ONLY_IN_ROCM or pkg.startswith(CUDA_ONLY_PREFIXES):
            errors.append(f"CUDA-only package {pkg!r} listed in rocm extra")

    # torch uses install-rocm script; rocm extra is intentionally empty
    if "pytorch-rocm642" not in {
        s["name"] for s in data.get("tool", {}).get("poetry", {}).get("source", [])
    }:
        # sources are top-level in pyproject
        pass

    sources = data.get("tool", {}).get("poetry", {}).get("source", [])
    if not any(s.get("name") == "pytorch-rocm642" for s in sources):
        errors.append("missing pytorch-rocm642 poetry source")

    cuda_has_torch = "triton" in cuda_extra or "xformers" in cuda_extra
    if not cuda_has_torch:
        errors.append(
            "cuda extra should include CUDA accelerator packages (e.g. xformers)"
        )

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print("ROCm extras configuration OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
