"""Unit tests for Wan domain LoRA Pydantic config loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from videotuna.training.wan_lora.config import WanLoraTrainConfig, load_wan_lora_config

REPO_ROOT = Path(__file__).resolve().parents[1]
WAN_T2V_CONFIG = REPO_ROOT / "configs" / "domain" / "wan_t2v_lora.yaml"


def test_invalid_task_raises_validation_error():
    cfg = load_wan_lora_config(WAN_T2V_CONFIG)
    payload = cfg.model_dump(mode="json")
    payload["flow"]["params"]["task"] = "t2v-A14B"
    with pytest.raises(ValidationError):
        WanLoraTrainConfig.model_validate(payload)


def test_round_trip_revalidates():
    cfg = load_wan_lora_config(WAN_T2V_CONFIG)
    round_tripped = WanLoraTrainConfig.model_validate(cfg.model_dump(mode="json"))
    assert round_tripped.train.name == cfg.train.name
    assert round_tripped.flow.params.ckpt_path == cfg.flow.params.ckpt_path
