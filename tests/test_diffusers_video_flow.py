"""Unit tests for the PrivTune Diffusers inference flow (Flux + Wan 2.2)."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest import mock

import torch
from omegaconf import OmegaConf

from videotuna.flow.diffusers_video import (
    MODEL_REGISTRY,
    DiffusersVideoFlow,
    resolve_model_id,
    resolve_torch_dtype,
)
from videotuna.utils.diffusers_optimizations import (
    apply_diffusers_optimizations,
    transformer_cache_context,
)


def test_resolve_model_id_defaults():
    assert (
        resolve_model_id("flux", "t2i", None, model_variant="1-dev")
        == "black-forest-labs/FLUX.1-dev"
    )
    assert (
        resolve_model_id("wan", "t2v", None, model_variant="2.2")
        == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    )
    assert resolve_model_id("wan", "t2v", "custom/model") == "custom/model"


def test_resolve_torch_dtype():
    assert resolve_torch_dtype("fp16") == torch.float16
    assert resolve_torch_dtype("bf16") == torch.bfloat16
    assert resolve_torch_dtype(None) == torch.bfloat16


def test_model_registry_covers_domain_families():
    assert ("flux", "t2i") in MODEL_REGISTRY
    assert ("wan", "t2v") in MODEL_REGISTRY
    assert ("cogvideox", "t2v") not in MODEL_REGISTRY


def test_apply_diffusers_optimizations_mock_pipe():
    pipe = mock.MagicMock()
    pipe.vae = mock.MagicMock()
    del pipe.enable_vae_tiling
    args = argparse.Namespace(
        enable_sequential_cpu_offload=False,
        enable_model_cpu_offload=True,
        enable_vae_slicing=True,
        enable_vae_tiling=True,
        fuse_qkv=True,
        enable_attention_cache=False,
    )
    apply_diffusers_optimizations(pipe, args)
    pipe.enable_model_cpu_offload.assert_called_once()
    pipe.vae.enable_slicing.assert_called_once()
    pipe.vae.enable_tiling.assert_called_once()
    pipe.fuse_qkv_projections.assert_called_once()
    pipe.set_progress_bar_config.assert_called_once()


def test_transformer_cache_context_noop_without_transformer():
    pipe = SimpleNamespace(transformer=None)
    with transformer_cache_context(pipe):
        pass


def test_diffusers_video_flow_instantiate_pipeline_only():
    flow = DiffusersVideoFlow(
        model_family="wan",
        mode="t2v",
        pretrained_model_name_or_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    )
    assert flow.pipeline_only is True
    assert flow.pipeline is None


@mock.patch("videotuna.flow.diffusers_video.export_to_video")
@mock.patch.object(DiffusersVideoFlow, "_generate_sample")
def test_inference_t2v_saves_video(mock_generate, mock_export):
    mock_generate.return_value = {
        "result": [{"frame": 0}],
        "peak_vram_gb": 1.0,
        "wall_time_s": 2.0,
    }
    flow = DiffusersVideoFlow(model_family="wan", mode="t2v")
    flow.pipeline = mock.MagicMock()
    args = OmegaConf.create(
        {
            "savedir": "/tmp/privtune-test",
            "prompt_file": "inputs/t2v/prompts.txt",
            "frames": 49,
            "num_inference_steps": 4,
            "unconditional_guidance_scale": 6.0,
            "seed": 1,
            "savefps": 8,
        }
    )
    with mock.patch.object(
        DiffusersVideoFlow, "load_inference_inputs", return_value=["hello"]
    ):
        with mock.patch.object(flow, "save_metrics"):
            metrics = flow.inference(args)
    assert len(metrics["per_sample"]) == 1
    mock_export.assert_called_once()


def test_yaml_wan22_instantiates_flow():
    from videotuna.utils.common_utils import instantiate_from_config

    cfg = OmegaConf.load("configs/inference/presets/balanced_wan2_2_720p.yaml")
    flow = instantiate_from_config(cfg.flow, resolve=True)
    assert isinstance(flow, DiffusersVideoFlow)
    assert flow.model_variant == "2.2"
