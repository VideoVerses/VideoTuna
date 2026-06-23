"""CPU smoke tests for the first-party Flux LoRA trainer."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from videotuna.training.flux_lora.config import FluxLoraDataConfig, load_train_config
from videotuna.training.flux_lora.dataset import FluxLoraImageDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
FLUX_CONFIG = REPO_ROOT / "configs" / "domain" / "flux_t2i.json"
FLUX_DATA = REPO_ROOT / "configs" / "domain" / "flux_t2i_data.json"


@pytest.fixture
def tiny_image_dataset(tmp_path):
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    img.save(tmp_path / "sample.png")
    (tmp_path / "sample.txt").write_text("a photo of sample", encoding="utf-8")
    return FluxLoraDataConfig(
        instance_data_dir=str(tmp_path),
        caption_strategy="filename",
        resolution=64,
    )


def test_dataset_loads_local_images(tiny_image_dataset):
    dataset = FluxLoraImageDataset(tiny_image_dataset, seed=0)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["caption"] == "a photo of sample"
    assert sample["pixel_values"].shape[0] == 3
    assert sample["pixel_values"].shape[1] % 64 == 0
    assert sample["pixel_values"].shape[2] % 64 == 0


def test_text_embeds_backend_requires_cache_dir(tmp_path):
    data_path = tmp_path / "backends.json"
    data_path.write_text(
        '[{"type":"local","instance_data_dir":"data","caption_strategy":"filename"},'
        '{"type":"local","dataset_type":"text_embeds","disabled":false}]',
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"--pretrained_model_name_or_path":"black-forest-labs/FLUX.1-dev",'
        '"--output_dir":"results/train/test",'
        '"--max_train_steps":10}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cache_dir"):
        load_train_config(config_path, data_path)


def test_text_embeds_backend_parses_with_cache_dir(tmp_path):
    data_path = tmp_path / "backends.json"
    data_path.write_text(
        '[{"type":"local","instance_data_dir":"data","caption_strategy":"filename"},'
        '{"type":"local","dataset_type":"text_embeds","cache_dir":"cache/text",'
        '"disabled":false}]',
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"--pretrained_model_name_or_path":"black-forest-labs/FLUX.1-dev",'
        '"--output_dir":"results/train/test",'
        '"--max_train_steps":10}',
        encoding="utf-8",
    )
    train_cfg, data_cfg = load_train_config(config_path, data_path)
    assert train_cfg.max_train_steps == 10
    assert data_cfg.text_embeds is not None
    assert data_cfg.text_embeds.cache_dir == "cache/text"


def test_load_train_config_from_repo_defaults():
    train_cfg, data_cfg = load_train_config(FLUX_CONFIG, FLUX_DATA)
    assert train_cfg.pretrained_model_name_or_path == "black-forest-labs/FLUX.1-dev"
    assert data_cfg.resolution == 512
    assert train_cfg.num_workers == 0
    assert train_cfg.gradient_checkpointing is True
    assert train_cfg.validation_steps == 40


def test_load_train_config_num_workers_from_json(tmp_path):
    data_path = tmp_path / "backends.json"
    data_path.write_text(
        '[{"type":"local","instance_data_dir":"data","caption_strategy":"filename"}]',
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"--pretrained_model_name_or_path":"black-forest-labs/FLUX.1-dev",'
        '"--output_dir":"results/train/test",'
        '"--num_workers": 4}',
        encoding="utf-8",
    )
    train_cfg, _ = load_train_config(config_path, data_path)
    assert train_cfg.num_workers == 4


def test_flux_lora_target_modules():
    from videotuna.training.flux_lora.model_utils import FLUX_LORA_TARGET_MODULES

    assert "to_q" in FLUX_LORA_TARGET_MODULES
    assert len(FLUX_LORA_TARGET_MODULES) == 4


def test_checkpoint_save_with_mock_transformer(tmp_path):
    pytest.importorskip("peft")
    from typing import Any, cast

    from diffusers import FluxTransformer2DModel
    from peft import LoraConfig, get_peft_model

    try:
        transformer = FluxTransformer2DModel(
            in_channels=64,
            out_channels=64,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=64,
            num_attention_heads=4,
            joint_attention_dim=64,
            pooled_projection_dim=64,
            guidance_embeds=True,
        )
    except Exception as exc:
        pytest.skip(f"Could not construct FluxTransformer2DModel stub: {exc}")

    lora_config = LoraConfig(r=4, lora_alpha=4, target_modules=["to_q"])
    transformer = get_peft_model(cast(Any, transformer), lora_config)
    from videotuna.training.flux_lora.checkpoint import save_lora_checkpoint

    path = save_lora_checkpoint(transformer, tmp_path, step=1)
    assert path.is_dir()
    assert any(path.iterdir())


def test_create_optimizer_adamw_uses_torch(monkeypatch):
    from unittest import mock

    import torch.nn as nn

    from videotuna.training.flux_lora.config import FluxLoraTrainConfig
    from videotuna.training.flux_lora.train import _create_optimizer

    model = nn.Linear(4, 4)
    config = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir="results/train/test",
        instance_data_dir="data/t2i/domain",
        optimizer="adamw",
    )
    optimi_mock = mock.MagicMock()
    monkeypatch.setitem(
        __import__("sys").modules,
        "optimi",
        mock.MagicMock(AdamW=optimi_mock),
    )
    opt = _create_optimizer(model, config)
    assert isinstance(opt, torch.optim.AdamW)
    optimi_mock.assert_not_called()


def test_create_optimizer_adamw_bf16_uses_optimi(monkeypatch):
    from unittest import mock

    import torch.nn as nn

    from videotuna.training.flux_lora.config import FluxLoraTrainConfig
    from videotuna.training.flux_lora.train import _create_optimizer

    model = nn.Linear(4, 4)
    config = FluxLoraTrainConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        output_dir="results/train/test",
        instance_data_dir="data/t2i/domain",
        optimizer="adamw_bf16",
    )
    optimi_cls = mock.MagicMock(return_value=mock.MagicMock())
    monkeypatch.setitem(
        __import__("sys").modules,
        "optimi",
        mock.MagicMock(AdamW=optimi_cls),
    )
    opt = _create_optimizer(model, config)
    optimi_cls.assert_called_once()
    assert opt is optimi_cls.return_value


def test_create_flux_accelerator_uses_tensorboard(tmp_path, monkeypatch):
    from unittest import mock

    from videotuna.training.flux_lora.train import create_flux_accelerator

    captured: dict = {}

    def fake_accelerator(**kwargs):
        captured.update(kwargs)
        return mock.MagicMock()

    monkeypatch.setattr(
        "videotuna.training.flux_lora.train.Accelerator",
        fake_accelerator,
    )
    create_flux_accelerator(
        tmp_path,
        mixed_precision="bf16",
        metrics_backend="tensorboard",
    )
    assert captured["log_with"] == "tensorboard"
    assert captured["project_config"].logging_dir == str(tmp_path / "tensorboard")
    assert captured["project_config"].project_dir == str(tmp_path)


def test_create_flux_accelerator_uses_trackio_dual_logging(tmp_path, monkeypatch):
    from unittest import mock

    from videotuna.training.flux_lora.train import create_flux_accelerator

    captured: dict = {}

    def fake_accelerator(**kwargs):
        captured.update(kwargs)
        return mock.MagicMock()

    monkeypatch.setattr(
        "videotuna.training.flux_lora.train.Accelerator",
        fake_accelerator,
    )
    monkeypatch.setattr(
        "videotuna.utils.training_metrics.trackio_available",
        lambda: True,
    )
    create_flux_accelerator(tmp_path, mixed_precision="bf16", metrics_backend="trackio")
    assert captured["log_with"] == ["tensorboard", "trackio"]
