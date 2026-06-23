"""CPU smoke tests for Wan 2.2 Diffusers inference preset YAMLs."""

from pathlib import Path

import yaml
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]

LOW_VRAM_PRESET = (
    REPO_ROOT / "configs" / "inference" / "presets" / "low_vram_wan2_2_720p.yaml"
)
BALANCED_PRESET = (
    REPO_ROOT / "configs" / "inference" / "presets" / "balanced_wan2_2_720p.yaml"
)
MAX_SPEED_PRESET = (
    REPO_ROOT / "configs" / "inference" / "presets" / "max_speed_wan2_2_720p.yaml"
)
CPU_SMOKE_PRESET = (
    REPO_ROOT / "configs" / "inference" / "presets" / "wan2_2_cpu_smoke.yaml"
)

WAN_MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_wan_low_vram_preset():
    cfg = _load_yaml(LOW_VRAM_PRESET)
    assert cfg["flow"]["params"]["model_variant"] == "2.2"
    assert cfg["inference"]["memory_preset"] == "low_vram"
    assert cfg["inference"]["enable_sequential_cpu_offload"] is True
    assert cfg["inference"]["dtype"] == "fp16"
    assert cfg["inference"]["min_vram_gb"] == 10
    assert cfg["inference"]["height"] == 720
    assert cfg["inference"]["width"] == 1280


def test_wan_low_vram_int8_quant_preset():
    path = (
        REPO_ROOT
        / "configs"
        / "inference"
        / "presets"
        / "low_vram_wan2_2_720p_int8.yaml"
    )
    cfg = _load_yaml(path)
    assert cfg["inference"]["transformer_quant"] == "int8_wo"
    assert cfg["inference"]["quant_backend"] == "torchao"
    assert cfg["inference"]["enable_model_cpu_offload"] is True


def test_wan_low_vram_fp8_quant_preset():
    path = (
        REPO_ROOT
        / "configs"
        / "inference"
        / "presets"
        / "low_vram_wan2_2_720p_fp8.yaml"
    )
    cfg = _load_yaml(path)
    assert cfg["inference"]["transformer_quant"] == "fp8_wo"
    assert cfg["inference"]["quant_backend"] == "torchao"
    assert cfg["inference"]["enable_model_cpu_offload"] is True
    assert cfg["inference"]["dtype"] == "bf16"


def test_wan_balanced_preset():
    cfg = _load_yaml(BALANCED_PRESET)
    assert cfg["flow"]["params"]["pretrained_model_name_or_path"] == WAN_MODEL_ID
    assert cfg["inference"]["memory_preset"] == "balanced"
    assert cfg["inference"]["enable_model_cpu_offload"] is True
    assert cfg["inference"]["enable_vae_tiling"] is True
    assert cfg["inference"]["dtype"] == "bf16"
    assert cfg["inference"]["min_vram_gb"] == 20


def test_wan_max_speed_preset():
    cfg = _load_yaml(MAX_SPEED_PRESET)
    assert cfg["inference"]["memory_preset"] == "max_speed"
    assert cfg["inference"]["dtype"] == "bf16"
    assert cfg["inference"]["min_vram_gb"] == 38
    assert "enable_model_cpu_offload" not in cfg["inference"]
    assert "enable_sequential_cpu_offload" not in cfg["inference"]


def test_wan_cpu_smoke_preset():
    cfg = OmegaConf.load(CPU_SMOKE_PRESET)
    assert cfg.flow.params.model_family == "wan"
    assert cfg.inference.device == "cpu"
    assert cfg.inference.frames == 2
    assert cfg.inference.height == 256
    assert cfg.inference.width == 448
    assert cfg.inference.num_inference_steps == 4
    assert cfg.inference.dtype == "fp32"
