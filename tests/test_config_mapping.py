"""Tests for config path mapping validation and application."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from videotuna.utils.config_mapping import (
    ConfigMappingError,
    ConfigPathMappings,
    apply_config_mappings,
    config_path_exists,
    get_config_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WAN_T2V_CONFIG = REPO_ROOT / "configs" / "domain" / "wan_t2v_lora.yaml"


def _minimal_cfg(*, include_mapping: bool = True) -> OmegaConf:
    cfg = OmegaConf.create(
        {
            "train": {
                "ckpt": "checkpoints/wan/Wan2.1-T2V-14B",
            },
            "flow": {
                "params": {
                    "ckpt_path": "checkpoints/wan/old-path",
                }
            },
        }
    )
    if include_mapping:
        cfg.train.mapping = {"train.ckpt": "flow.params.ckpt_path"}
    return cfg


def test_config_path_exists_valid_and_missing():
    cfg = _minimal_cfg(include_mapping=False)

    assert config_path_exists(cfg, "train.ckpt") is True
    assert config_path_exists(cfg, "flow.params.ckpt_path") is True
    assert config_path_exists(cfg, "flow.params.missing") is False
    assert config_path_exists(cfg, "missing.top") is False
    assert config_path_exists(cfg, "train.ckpt.bad") is False
    assert config_path_exists(cfg, "") is False


def test_get_config_path_raises_for_missing_path():
    cfg = _minimal_cfg(include_mapping=False)

    assert get_config_path(cfg, "train.ckpt") == "checkpoints/wan/Wan2.1-T2V-14B"
    with pytest.raises(ConfigMappingError, match="does not exist"):
        get_config_path(cfg, "train.missing")


def test_apply_train_mapping_valid():
    cfg = _minimal_cfg()

    apply_config_mappings(cfg, section="train")

    assert cfg.flow.params.ckpt_path == "checkpoints/wan/Wan2.1-T2V-14B"


def test_apply_train_mapping_missing_source():
    cfg = _minimal_cfg()
    cfg.train.mapping = {"train.no_such": "flow.params.ckpt_path"}

    with pytest.raises(ConfigMappingError) as exc_info:
        apply_config_mappings(cfg, section="train")

    message = str(exc_info.value)
    assert "train.mapping" in message
    assert "source path" in message
    assert "train.no_such" in message
    assert "flow.params.ckpt_path" in message


def test_apply_train_mapping_missing_target():
    cfg = _minimal_cfg()
    cfg.train.mapping = {"train.ckpt": "flow.params.no_such"}

    with pytest.raises(ConfigMappingError) as exc_info:
        apply_config_mappings(cfg, section="train")

    message = str(exc_info.value)
    assert "train.mapping" in message
    assert "target path" in message
    assert "flow.params.no_such" in message
    assert "train.ckpt" in message


def test_apply_train_mapping_invalid_shape():
    cfg = _minimal_cfg()
    cfg.train.mapping = ["train.ckpt", "flow.params.ckpt_path"]

    with pytest.raises(ConfigMappingError, match="must be a mapping"):
        apply_config_mappings(cfg, section="train")


def test_apply_train_mapping_invalid_dot_path():
    cfg = _minimal_cfg()
    cfg.train.mapping = {"train..ckpt": "flow.params.ckpt_path"}

    with pytest.raises(ConfigMappingError, match="invalid dot paths"):
        apply_config_mappings(cfg, section="train")


def test_config_path_mappings_rejects_invalid_dot_path():
    with pytest.raises(ValidationError, match="invalid source path"):
        ConfigPathMappings(root={"train..ckpt": "flow.params.ckpt_path"})


def test_apply_inference_mapping_missing_source():
    cfg = OmegaConf.create(
        {
            "inference": {
                "ckpt_path": "checkpoints/wan/Wan2.1-T2V-14B",
                "mapping": {"inference.no_such": "flow.params.ckpt_path"},
            },
            "flow": {"params": {"ckpt_path": "checkpoints/wan/Wan2.1-T2V-14B"}},
        }
    )

    with pytest.raises(ConfigMappingError) as exc_info:
        apply_config_mappings(cfg, section="inference")

    message = str(exc_info.value)
    assert "inference.mapping" in message
    assert "source path" in message
    assert "inference.no_such" in message


def test_apply_mapping_noop_when_absent():
    cfg = _minimal_cfg(include_mapping=False)
    original_ckpt_path = cfg.flow.params.ckpt_path

    apply_config_mappings(cfg, section="train")

    assert cfg.flow.params.ckpt_path == original_ckpt_path


def test_wan_domain_yaml_mapping_paths_exist():
    cfg = OmegaConf.load(WAN_T2V_CONFIG)

    apply_config_mappings(cfg, section="train")
    apply_config_mappings(cfg, section="inference")

    assert cfg.flow.params.ckpt_path == cfg.train.ckpt
