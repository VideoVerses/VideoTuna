#!/usr/bin/env python3
"""
Compare Wan 2.2 Diffusers peak VRAM: baseline vs torchao int8/fp8 vs quanto int8.

Manual rental-GPU spike — not run in CI. Documents quant evaluation per runbook.

Usage:
  poetry run python tools/spike_wan_quant_compare.py \\
    --prompt "A cat walking on grass" \\
    --frames 17 --height 480 --width 848

  # fp8 on Ada/Hopper (sm >= 8.9):
  poetry run python tools/spike_wan_quant_compare.py \\
    --schemes none,fp8_wo --prompt "A cat walking on grass"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_FP8_MIN_COMPUTE_CAPABILITY = (8, 9)


def _fp8_gpu_supported() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    return (major, minor) >= _FP8_MIN_COMPUTE_CAPABILITY


def _run_variant(
    *,
    label: str,
    transformer_quant: str,
    quant_backend: str,
    args: argparse.Namespace,
) -> dict:
    import torch
    from diffusers import WanPipeline

    from videotuna.utils.common_utils import monitor_resources
    from videotuna.utils.diffusers_optimizations import apply_diffusers_optimizations
    from videotuna.utils.diffusers_quantization import (
        build_pipeline_quantization_config,
        resolve_quant_components,
        validate_transformer_quant,
    )

    validate_transformer_quant(
        transformer_quant=transformer_quant,
        quant_backend=quant_backend,
        offload_mode="model",
    )
    dtype = torch.bfloat16 if transformer_quant == "fp8_wo" else torch.float16
    model_id = args.model_id
    load_kwargs: dict = {"torch_dtype": dtype}
    if transformer_quant != "none":
        components = resolve_quant_components("wan", "2.2", "t2v")
        qcfg = build_pipeline_quantization_config(
            transformer_quant=transformer_quant,
            quant_backend=quant_backend,
            components=components,
        )
        if qcfg is not None:
            load_kwargs["quantization_config"] = qcfg

    pipe = WanPipeline.from_pretrained(model_id, **load_kwargs)
    ns = argparse.Namespace(
        enable_sequential_cpu_offload=False,
        enable_model_cpu_offload=True,
        enable_vae_tiling=True,
        enable_vae_slicing=False,
        fuse_qkv=False,
        enable_attention_cache=False,
        device=None,
        device_map=None,
    )
    apply_diffusers_optimizations(pipe, ns, model_family="wan")

    @monitor_resources(return_metrics=True, frames=args.frames)
    def _gen():
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        pipe(
            prompt=args.prompt,
            num_frames=args.frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=5.0,
            generator=generator,
        )
        return {"ok": True}

    metrics = _gen()
    return {
        "label": label,
        "transformer_quant": transformer_quant,
        "quant_backend": quant_backend,
        "peak_vram_gb": metrics.get("peak_vram_gb", -1),
        "wall_time_s": metrics.get("wall_time_s", -1),
    }


def _build_variants(cli: argparse.Namespace) -> list[tuple[str, str, str]]:
    scheme_to_label = {
        "none": "baseline",
        "int8_wo": "torchao_int8",
        "int4_wo": "torchao_int4",
        "fp8_wo": "torchao_fp8",
    }
    requested = [s.strip() for s in cli.schemes.split(",") if s.strip()]
    variants: list[tuple[str, str, str]] = []
    for scheme in requested:
        if scheme not in scheme_to_label:
            raise ValueError(
                f"Unsupported scheme {scheme!r}. "
                f"Expected one of: {', '.join(scheme_to_label)}"
            )
        label = scheme_to_label[scheme]
        backend = "torchao"
        if scheme == "int8_wo" and cli.include_quanto:
            variants.append((label, scheme, backend))
            variants.append(("quanto_int8", scheme, "quanto"))
            continue
        if scheme == "fp8_wo" and not _fp8_gpu_supported():
            print(
                f"Skipping {label}: fp8_wo requires NVIDIA GPU sm >= "
                f"{_FP8_MIN_COMPUTE_CAPABILITY[0]}.{_FP8_MIN_COMPUTE_CAPABILITY[1]}",
                file=sys.stderr,
            )
            continue
        variants.append((label, scheme, backend))
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Wan 2.2 quant VRAM spike")
    parser.add_argument(
        "--model-id",
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    )
    parser.add_argument("--prompt", default="A cat walking on grass")
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--schemes",
        default="none,int8_wo,fp8_wo",
        help="Comma-separated transformer_quant schemes to benchmark",
    )
    parser.add_argument(
        "--include-int4",
        action="store_true",
        help="Add torchao int4_wo variant to the run list",
    )
    parser.add_argument(
        "--include-quanto",
        action="store_true",
        help="Also run optimum-quanto int8 when int8_wo is requested",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/spike_wan_quant.json"),
    )
    cli = parser.parse_args()

    import torch
    import torchao

    if not torch.cuda.is_available():
        print("CUDA required for spike_wan_quant_compare", file=sys.stderr)
        sys.exit(1)

    schemes = cli.schemes
    if cli.include_int4 and "int4_wo" not in schemes:
        schemes = f"{schemes},int4_wo"
    cli.schemes = schemes

    try:
        variants = _build_variants(cli)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    if cli.include_quanto:
        try:
            import optimum.quanto  # noqa: F401
        except ImportError:
            print("optimum-quanto not installed; skipping quanto variant")

    results: list[dict] = [{"torchao_version": torchao.__version__}]
    for label, quant, backend in variants:
        if backend == "quanto":
            try:
                import optimum.quanto  # noqa: F401
            except ImportError:
                continue
        print(f"Running {label}...")
        try:
            metrics = _run_variant(
                label=label,
                transformer_quant=quant,
                quant_backend=backend,
                args=cli,
            )
        except Exception as exc:
            metrics = {"label": label, "error": str(exc)}
        results.append(metrics)
        print(json.dumps(metrics, indent=2))

    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {cli.out}")


if __name__ == "__main__":
    main()
