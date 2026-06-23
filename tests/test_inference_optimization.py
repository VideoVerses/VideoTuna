"""Tests for inference CLI, metrics, and optimization."""

import argparse
import json
import os
import tempfile
from unittest import mock

import pytest

from videotuna.cli.inference_options import (
    StandardInferenceOptions,
    inference_options_to_namespace,
)
from videotuna.utils.common_utils import monitor_resources, save_metrics
from videotuna.utils.inference_cli import (
    apply_compile_env,
    prepare_cli_inference_args,
)
from videotuna.utils.inference_profile import resolve_inference_profile


def test_standard_inference_options_to_namespace():
    standard = StandardInferenceOptions(
        device="cuda:1",
        min_vram_gb=24.0,
        memory_preset="low_vram",
        enable_vae_tiling=True,
        enable_sequential_cpu_offload=True,
        dtype="bf16",
        ulysses_degree=2,
        ring_degree=2,
        compile=True,
        transformer_quant="int8_wo",
        quant_backend="torchao",
    )
    args = inference_options_to_namespace(standard=standard)
    assert args.device == "cuda:1"
    assert args.min_vram_gb == 24.0
    assert args.memory_preset == "low_vram"
    assert args.enable_vae_tiling is True
    assert args.enable_sequential_cpu_offload is True
    assert args.dtype == "bf16"
    assert args.ulysses_degree == 2
    assert args.compile is True
    assert args.transformer_quant == "int8_wo"
    assert args.quant_backend == "torchao"
    assert not hasattr(args, "enable_fp8")


def test_resolve_inference_profile():
    args = argparse.Namespace(
        memory_preset="low_vram",
        enable_model_cpu_offload=False,
        enable_sequential_cpu_offload=False,
        enable_vae_tiling=False,
        dtype=None,
    )
    profile = resolve_inference_profile(args)
    assert profile.offload_mode == "sequential"
    assert profile.enable_sequential_cpu_offload is True
    assert profile.enable_model_cpu_offload is False
    assert profile.enable_vae_tiling is True
    assert profile.dtype == "fp16"
    assert profile.memory_preset == "low_vram"
    assert args.enable_sequential_cpu_offload is True
    assert args.enable_vae_tiling is True
    assert args.dtype == "fp16"

    args = argparse.Namespace(
        memory_preset=None,
        enable_model_cpu_offload=True,
        enable_sequential_cpu_offload=False,
        enable_vae_tiling=False,
        dtype="bf16",
    )
    profile = resolve_inference_profile(args, apply_preset=False)
    assert profile.offload_mode == "model"
    assert profile.dtype == "bf16"


def test_prepare_cli_inference_args_validates_parallel():
    args = argparse.Namespace(
        memory_preset=None,
        ulysses_degree=2,
        ring_degree=2,
        cpu_smoke=False,
        device=None,
        enable_sequential_cpu_offload=False,
        enable_model_cpu_offload=False,
    )
    with mock.patch.dict(os.environ, {"WORLD_SIZE": "3"}):
        with pytest.raises(ValueError, match="ulysses_degree"):
            prepare_cli_inference_args(args)


def test_validate_cpu_offload_rejected_on_cpu_smoke():
    from videotuna.utils.inference_cli import validate_cpu_offload_flags

    args = argparse.Namespace(
        cpu_smoke=True,
        device=None,
        enable_sequential_cpu_offload=True,
        enable_model_cpu_offload=False,
        memory_preset=None,
    )
    with pytest.raises(RuntimeError, match="CPU offload flags"):
        validate_cpu_offload_flags(args)


def test_validate_cpu_offload_both_flags_sequential_wins():
    from videotuna.utils.inference_cli import validate_cpu_offload_flags

    args = argparse.Namespace(
        cpu_smoke=False,
        device="cuda:0",
        enable_sequential_cpu_offload=True,
        enable_model_cpu_offload=True,
        memory_preset=None,
    )
    with (
        mock.patch("videotuna.utils.device_utils.gpu_is_available", return_value=True),
        mock.patch(
            "videotuna.utils.device_utils.detect_compute_backend", return_value="cuda"
        ),
        mock.patch("videotuna.utils.device_utils.resolve_cpu_mode", return_value="off"),
        mock.patch("videotuna.utils.inference_cli.logger.warning") as warn,
    ):
        validate_cpu_offload_flags(args)

    assert args.enable_sequential_cpu_offload is True
    assert args.enable_model_cpu_offload is False
    warn.assert_called_once()
    assert "sequential" in warn.call_args[0][0].lower()
    profile = resolve_inference_profile(args, apply_preset=False)
    assert profile.offload_mode == "sequential"


def test_apply_cpu_smoke_env():
    from videotuna.utils.inference_cli import apply_cpu_smoke_env

    args = argparse.Namespace(cpu_smoke=True)
    with mock.patch.dict(os.environ, {}, clear=True):
        apply_cpu_smoke_env(args)
        assert os.environ["VIDEOTUNA_CPU_MODE"] == "smoke"
        assert os.environ["VIDEOTUNA_ATTN_BACKEND"] == "eager"
        assert os.environ["VIDEOTUNA_TORCH_COMPILE"] == "0"


@mock.patch.dict(
    os.environ,
    {"VIDEOTUNA_ATTN_BACKEND": "flash", "VIDEOTUNA_ATTN_BACKEND_STRICT": "0"},
)
def test_attn_flash_fallback_to_sdpa():
    from videotuna.utils import attention

    with mock.patch.object(attention, "_FLASH_ATTN_AVAILABLE", False):
        with mock.patch.object(
            attention, "detect_compute_backend", return_value="cuda"
        ):
            with mock.patch.object(attention, "gpu_is_available", return_value=True):
                assert attention.get_attn_backend() == "sdpa"


@mock.patch.dict(
    os.environ,
    {"VIDEOTUNA_ATTN_BACKEND": "flash", "VIDEOTUNA_ATTN_BACKEND_STRICT": "1"},
)
def test_attn_flash_strict_raises():
    from videotuna.utils import attention

    with mock.patch.object(attention, "_FLASH_ATTN_AVAILABLE", False):
        with mock.patch.object(
            attention, "detect_compute_backend", return_value="cuda"
        ):
            with pytest.raises(RuntimeError, match="flash-attn"):
                attention.get_attn_backend()


def test_attn_auto_resolves():
    import torch

    from videotuna.utils.attention import get_attn_backend

    backend = get_attn_backend()
    if torch.cuda.is_available():
        assert backend in ("flash", "sdpa", "eager")
    else:
        assert backend in ("sdpa", "eager")


def test_apply_compile_env():
    apply_compile_env(True)
    assert os.environ["VIDEOTUNA_TORCH_COMPILE"] == "1"
    apply_compile_env(False)
    assert os.environ["VIDEOTUNA_TORCH_COMPILE"] == "0"


@mock.patch.dict(os.environ, {"VIDEOTUNA_ATTN_BACKEND": "eager"})
def test_monitor_resources_returns_extended_keys():
    @monitor_resources(return_metrics=True, frames=10)
    def dummy():
        return "ok"

    out = dummy()
    assert out["result"] == "ok"
    assert "peak_vram_gb" in out
    assert "seconds_per_frame" in out
    assert out["attention_backend"] == "eager"
    assert "torch_compile" in out


def test_benchmark_defaults_wan_only():
    from scripts.benchmark_attn_backends import DEFAULT_MODEL, DEFAULT_NUM_FRAMES

    assert DEFAULT_MODEL == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert DEFAULT_NUM_FRAMES == 17


def test_inference_new_imports_argparse():
    import torch

    from videotuna.utils.device_utils import require_accelerator_for_flow

    if torch.cuda.is_available():
        require_accelerator_for_flow("videotuna.flow.wanvideo.WanVideoModelFlow")
        return
    with pytest.raises(RuntimeError, match="requires a GPU"):
        require_accelerator_for_flow("videotuna.flow.wanvideo.WanVideoModelFlow")


def test_require_nvidia_cuda_alias_raises_without_gpu():
    import torch

    from videotuna.utils.device_utils import require_nvidia_cuda_for_flow

    if torch.cuda.is_available():
        require_nvidia_cuda_for_flow("videotuna.flow.wanvideo.WanVideoModelFlow")
        return
    with pytest.raises(RuntimeError, match="requires a GPU"):
        require_nvidia_cuda_for_flow("videotuna.flow.wanvideo.WanVideoModelFlow")


def test_save_metrics_writes_metrics_json():
    with tempfile.TemporaryDirectory() as tmp:
        save_metrics(
            savedir=tmp,
            gpu=[1.5],
            time=[10.0],
            frames=5,
        )
        path = os.path.join(tmp, "metrics.json")
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "per_sample" in data
        assert os.path.exists(os.path.join(tmp, "metric.json"))


def test_apply_diffusers_optimizations_compiles_when_no_offload():
    from unittest.mock import MagicMock

    from videotuna.utils import diffusers_optimizations

    transformer = MagicMock(name="transformer")
    compiled = MagicMock(name="compiled_transformer")
    pipe = MagicMock()
    pipe.transformer = transformer
    args = argparse.Namespace(
        enable_sequential_cpu_offload=False,
        enable_model_cpu_offload=False,
        enable_vae_slicing=False,
        enable_vae_tiling=False,
        fuse_qkv=False,
        enable_attention_cache=False,
        device=None,
        device_map=None,
    )
    with mock.patch.object(
        diffusers_optimizations, "maybe_compile_denoiser", return_value=compiled
    ) as compile_mock:
        with mock.patch.object(
            diffusers_optimizations, "apply_diffusers_attention_backend"
        ):
            with mock.patch.object(diffusers_optimizations, "resolve_inference_device"):
                with mock.patch.object(pipe, "to"):
                    diffusers_optimizations.apply_diffusers_optimizations(pipe, args)
    compile_mock.assert_called_once_with(transformer)
    assert pipe.transformer is compiled


def test_apply_diffusers_optimizations_skips_compile_with_offload():
    from unittest.mock import MagicMock

    from videotuna.utils import diffusers_optimizations

    pipe = MagicMock()
    pipe.transformer = MagicMock(name="transformer")
    args = argparse.Namespace(
        enable_sequential_cpu_offload=False,
        enable_model_cpu_offload=True,
        enable_vae_slicing=False,
        enable_vae_tiling=False,
        fuse_qkv=False,
        enable_attention_cache=False,
        device=None,
        device_map=None,
    )
    with mock.patch.object(
        diffusers_optimizations, "maybe_compile_denoiser"
    ) as compile_mock:
        with mock.patch.object(
            diffusers_optimizations, "apply_diffusers_attention_backend"
        ):
            diffusers_optimizations.apply_diffusers_optimizations(pipe, args)
    compile_mock.assert_not_called()


def test_require_accelerator_for_flow_raises_without_gpu():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "inference_new.py"
    ).read_text(encoding="utf-8")
    assert "generic_inference_entry" in source
    assert "import argparse" not in source


@pytest.mark.gpu
def test_wan_domain_low_vram_quanto_int4_gpu_smoke():
    """GPU integration: optimum-quanto int4_wo on Wan 2.2 low-VRAM preset."""
    pytest.importorskip("optimum.quanto")

    from pathlib import Path

    import torch
    import yaml
    from diffusers import WanPipeline

    from videotuna.utils.device_utils import detect_compute_backend
    from videotuna.utils.diffusers_optimizations import apply_diffusers_optimizations
    from videotuna.utils.diffusers_quantization import (
        build_pipeline_quantization_config,
        maybe_adjust_offload_for_quant,
        resolve_quant_components,
        validate_transformer_quant,
    )

    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    if detect_compute_backend() == "rocm":
        pytest.skip("quanto quant is CUDA-only")

    repo_root = Path(__file__).resolve().parents[1]
    preset_path = (
        repo_root
        / "configs"
        / "inference"
        / "presets"
        / "wan_domain_lora_smoke_22_low_vram.yaml"
    )
    cfg = yaml.safe_load(preset_path.read_text(encoding="utf-8"))
    flow_params = cfg["flow"]["params"]
    inference = cfg["inference"]
    model_id = flow_params["pretrained_model_name_or_path"]
    model_variant = flow_params["model_variant"]

    args = argparse.Namespace(
        transformer_quant="int4_wo",
        quant_backend="quanto",
        enable_sequential_cpu_offload=inference["enable_sequential_cpu_offload"],
        enable_model_cpu_offload=False,
        enable_vae_tiling=inference["enable_vae_tiling"],
        enable_vae_slicing=False,
        fuse_qkv=False,
        enable_attention_cache=False,
        device=None,
        device_map=None,
    )
    maybe_adjust_offload_for_quant(args, "int4_wo")
    assert args.enable_sequential_cpu_offload is False
    assert args.enable_model_cpu_offload is True

    validate_transformer_quant(
        transformer_quant="int4_wo",
        quant_backend="quanto",
        offload_mode="model",
    )
    components = resolve_quant_components("wan", model_variant, "t2v")
    assert components == ["transformer", "transformer_2"]
    quant_config = build_pipeline_quantization_config(
        transformer_quant="int4_wo",
        quant_backend="quanto",
        components=components,
    )
    assert quant_config is not None
    assert set(quant_config.quant_mapping.keys()) == {"transformer", "transformer_2"}

    pipe = WanPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        quantization_config=quant_config,
    )
    apply_diffusers_optimizations(pipe, args, model_family="wan")

    generator = torch.Generator(device="cuda").manual_seed(42)
    output = pipe(
        prompt="sks_style, slow camera push-in, soft lighting",
        num_frames=5,
        height=256,
        width=448,
        num_inference_steps=2,
        guidance_scale=5.0,
        generator=generator,
    )
    assert output is not None
    assert len(output.frames[0]) == 5
