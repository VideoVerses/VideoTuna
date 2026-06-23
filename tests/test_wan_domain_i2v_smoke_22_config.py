"""CPU smoke tests for Wan 2.2 domain I2V validation preset."""

from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
WAN_I2V_SMOKE_22 = (
    REPO_ROOT / "configs" / "inference" / "presets" / "wan_domain_i2v_smoke_22.yaml"
)


def test_wan_domain_i2v_smoke_22_yaml():
    cfg = OmegaConf.load(WAN_I2V_SMOKE_22)
    assert cfg.flow.params.mode == "i2v"
    assert cfg.flow.params.model_variant == "2.2"
    assert cfg.inference.height == 720
    assert cfg.inference.width == 1280
    assert cfg.inference.frames == 81
    assert cfg.inference.num_inference_steps == 4
    assert "Wan2.2-I2V" in cfg.flow.params.pretrained_model_name_or_path
