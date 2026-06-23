"""CPU smoke tests for domain adult fine-tuning configs (no GPU weights)."""

import json
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from videotuna.training.flux_lora.config import load_train_config
from videotuna.training.wan_lora.config import load_wan_lora_config

REPO_ROOT = Path(__file__).resolve().parents[1]

FLUX_TRAIN_CONFIG = REPO_ROOT / "configs" / "domain" / "flux_t2i.json"
FLUX_DATA_CONFIG = REPO_ROOT / "configs" / "domain" / "flux_t2i_data.json"
FLUX_CLOUD_SMOKE_CONFIG = REPO_ROOT / "configs" / "domain" / "flux_t2i_cloud_smoke.json"
WAN_DOMAIN_CONFIG = REPO_ROOT / "configs" / "domain" / "wan_t2v_lora.yaml"
WAN_CLOUD_SMOKE_CONFIG = (
    REPO_ROOT / "configs" / "domain" / "wan_t2v_lora_cloud_smoke.yaml"
)
WAN_I2V_DOMAIN_CONFIG = REPO_ROOT / "configs" / "domain" / "wan_i2v_lora.yaml"
FLUX_INFER_SMOKE = (
    REPO_ROOT / "configs" / "inference" / "presets" / "flux_domain_lora_smoke.yaml"
)
WAN_INFER_SMOKE = (
    REPO_ROOT / "configs" / "inference" / "presets" / "wan_domain_lora_smoke.yaml"
)
WAN_INFER_SMOKE_22 = (
    REPO_ROOT / "configs" / "inference" / "presets" / "wan_domain_lora_smoke_22.yaml"
)


def test_flux_domain_train_config_loads():
    train_cfg, data_cfg = load_train_config(FLUX_TRAIN_CONFIG, FLUX_DATA_CONFIG)
    assert train_cfg.pretrained_model_name_or_path == "black-forest-labs/FLUX.1-dev"
    assert train_cfg.output_dir == "results/train/flux-domain-adult"
    assert train_cfg.max_train_steps == 2000
    assert train_cfg.checkpointing_steps == 250
    assert train_cfg.validation_prompt is not None
    assert train_cfg.validation_steps == 40
    assert train_cfg.gradient_checkpointing is True
    assert "sks_style" in train_cfg.validation_prompt
    assert data_cfg.instance_data_dir == "data/t2i/domain"
    assert data_cfg.caption_strategy == "filename"
    assert data_cfg.text_embeds is not None


def test_flux_domain_data_backend_json():
    backends = json.loads(FLUX_DATA_CONFIG.read_text(encoding="utf-8"))
    image_backend = next(b for b in backends if b.get("dataset_type") != "text_embeds")
    assert image_backend["instance_data_dir"] == "data/t2i/domain"
    assert "caption" not in image_backend


def test_wan_domain_yaml_parses():
    cfg = load_wan_lora_config(WAN_DOMAIN_CONFIG)
    assert cfg.train.name == "train_wan_domain_t2v_lora"
    csv_path = cfg.train.data.params["train"]["params"]["csv_path"]
    assert csv_path == "data/t2v/domain/metadata.csv"
    assert cfg.train.lightning.trainer.max_epochs == 50
    ckpt_cb = cfg.train.lightning.callbacks.model_checkpoint.params
    assert ckpt_cb["every_n_train_steps"] == 25
    assert cfg.flow.params.ckpt_path == "checkpoints/wan/Wan2.1-T2V-14B"


def test_flux_cloud_smoke_train_config_loads():
    train_cfg, data_cfg = load_train_config(FLUX_CLOUD_SMOKE_CONFIG, FLUX_DATA_CONFIG)
    assert train_cfg.max_train_steps == 50
    assert train_cfg.checkpointing_steps == 25
    assert train_cfg.output_dir == "results/train/flux-cloud-smoke"
    assert train_cfg.pretrained_model_name_or_path == "checkpoints/flux/FLUX.1-dev"
    assert data_cfg.instance_data_dir == "data/t2i/domain"


def test_wan_cloud_smoke_yaml_parses():
    cfg = load_wan_lora_config(WAN_CLOUD_SMOKE_CONFIG)
    assert cfg.train.name == "train_wan_cloud_smoke"
    assert cfg.train.lightning.trainer.max_epochs == 1
    ckpt_cb = cfg.train.lightning.callbacks.model_checkpoint.params
    assert ckpt_cb["every_n_train_steps"] == 5


def test_wan_i2v_domain_yaml_parses():
    cfg = load_wan_lora_config(WAN_I2V_DOMAIN_CONFIG)
    assert cfg.train.name == "train_wan_domain_i2v_lora"
    assert cfg.flow.params.task == "i2v-14B"
    csv_path = cfg.train.data.params["train"]["params"]["csv_path"]
    assert csv_path == "data/i2v/domain/metadata.csv"
    assert cfg.train.data.params["train"]["params"]["image_to_video"] is False
    assert cfg.inference.mode == "i2v"
    assert cfg.flow.params.ckpt_path == "checkpoints/wan/Wan2.1-I2V-14B-480P"


def test_flux_domain_inference_smoke_yaml():
    cfg = yaml.safe_load(FLUX_INFER_SMOKE.read_text(encoding="utf-8"))
    assert cfg["flow"]["params"]["model_variant"] == "1-dev"
    assert cfg["inference"]["height"] == 512
    assert cfg["inference"]["num_inference_steps"] == 8
    assert cfg["inference"]["enable_model_cpu_offload"] is True


def test_wan_domain_inference_smoke_yaml():
    cfg = OmegaConf.load(WAN_INFER_SMOKE)
    assert cfg.inference.height == 480
    assert cfg.inference.width == 832
    assert cfg.inference.frames == 81
    assert cfg.inference.num_inference_steps == 20
    assert cfg.flow.params.offload_model is True


def test_wan_domain_inference_smoke_22_yaml():
    cfg = OmegaConf.load(WAN_INFER_SMOKE_22)
    assert cfg.inference.height == 720
    assert cfg.inference.width == 1280
    assert cfg.inference.frames == 81
    assert cfg.inference.num_inference_steps == 4
    assert cfg.flow.params.model_variant == "2.2"
    assert "DiffusersVideoFlow" in cfg.flow.target
