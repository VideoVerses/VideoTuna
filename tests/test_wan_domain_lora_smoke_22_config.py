"""CPU smoke tests for Wan 2.2 domain LoRA validation preset."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

WAN_DOMAIN_SMOKE_22 = (
    REPO_ROOT / "configs" / "inference" / "presets" / "wan_domain_lora_smoke_22.yaml"
)
WAN_DOMAIN_SMOKE_22_LOW_VRAM = (
    REPO_ROOT
    / "configs"
    / "inference"
    / "presets"
    / "wan_domain_lora_smoke_22_low_vram.yaml"
)


def test_wan_domain_lora_smoke_22_yaml():
    cfg = yaml.safe_load(WAN_DOMAIN_SMOKE_22.read_text(encoding="utf-8"))
    assert cfg["flow"]["target"] == "videotuna.flow.diffusers_video.DiffusersVideoFlow"
    assert cfg["flow"]["params"]["model_variant"] == "2.2"
    inf = cfg["inference"]
    assert inf["height"] == 720
    assert inf["width"] == 1280
    assert inf["frames"] == 81
    assert inf["num_inference_steps"] == 4
    assert inf["savefps"] == 16
    assert inf["enable_model_cpu_offload"] is True
    assert "sks_style" in inf["prompt_file"]


def test_wan_domain_lora_smoke_22_low_vram_yaml():
    cfg = yaml.safe_load(WAN_DOMAIN_SMOKE_22_LOW_VRAM.read_text(encoding="utf-8"))
    inf = cfg["inference"]
    assert inf["height"] == 720
    assert inf["enable_sequential_cpu_offload"] is True
    assert inf["min_vram_gb"] == 10
