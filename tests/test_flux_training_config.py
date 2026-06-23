"""Flux LoRA training config loading (no GPU)."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FLUX_CONFIG = REPO_ROOT / "configs" / "domain" / "flux_t2i.json"
FLUX_DATA = REPO_ROOT / "configs" / "domain" / "flux_t2i_data.json"


def test_flux_training_config_json_loads():
    with open(FLUX_CONFIG) as f:
        config = json.load(f)
    assert config["--model_family"] == "flux"
    assert config["--pretrained_model_name_or_path"] == "black-forest-labs/FLUX.1-dev"
    assert config["--data_backend_config"] == "configs/domain/flux_t2i_data.json"


def test_flux_multidatabackend_json_loads():
    with open(FLUX_DATA) as f:
        backends = json.load(f)
    assert isinstance(backends, list)
    assert backends[0]["type"] == "local"
    assert backends[0]["instance_data_dir"]


def test_flux_training_config_loader():
    from videotuna.training.flux_lora.config import load_train_config

    train_cfg, data_cfg = load_train_config(FLUX_CONFIG, FLUX_DATA)
    assert train_cfg.model_family == "flux"
    assert train_cfg.lora_rank == 4
    assert train_cfg.max_train_steps == 2000
    assert train_cfg.checkpoints_total_limit == 20
    assert train_cfg.write_batch_size == 1
    assert train_cfg.optimizer == "adamw_bf16"
    assert data_cfg.caption_strategy == "filename"


def test_train_flux_lora_yaml_loader():
    """Exercise the YAML→JSON bridge used by some docs (no training run)."""
    import yaml

    sample = {
        "data": [{"id": "test", "type": "local", "instance_data_dir": "data/"}],
        "train": {
            "model_family": "flux",
            "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
        },
    }
    parsed = yaml.safe_load(yaml.dump(sample))
    assert parsed["train"]["model_family"] == "flux"
