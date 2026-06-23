"""Dataset behavior for Flux LoRA bucketing and caption dropout."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from videotuna.training.flux_lora.config import FluxLoraDataConfig
from videotuna.training.flux_lora.dataset import FluxLoraImageDataset

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_image_dataset(tmp_path):
    img = Image.new("RGB", (640, 480), color=(128, 64, 32))
    img.save(tmp_path / "sample.png")
    (tmp_path / "sample.txt").write_text("a photo of sample", encoding="utf-8")
    return FluxLoraDataConfig(
        instance_data_dir=str(tmp_path),
        caption_strategy="filename",
        resolution=512,
        resolution_type="pixel_area",
        aspect_bucket_rounding=2,
        minimum_image_size=0,
    )


def test_dataset_loads_local_images(tiny_image_dataset):
    dataset = FluxLoraImageDataset(tiny_image_dataset, seed=0)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["caption"] == "a photo of sample"
    height, width = sample["pixel_values"].shape[1:]
    assert width % 64 == 0
    assert height % 64 == 0


def test_dataset_filters_small_images(tmp_path):
    img = Image.new("RGB", (128, 128), color=(10, 20, 30))
    img.save(tmp_path / "small.png")
    (tmp_path / "small.txt").write_text("small", encoding="utf-8")
    data_cfg = FluxLoraDataConfig(
        instance_data_dir=str(tmp_path),
        caption_strategy="filename",
        resolution=512,
        minimum_image_size=512,
    )
    with pytest.raises(ValueError, match="No training images found"):
        FluxLoraImageDataset(data_cfg)


def test_caption_dropout(tmp_path):
    img = Image.new("RGB", (512, 512), color=(1, 2, 3))
    img.save(tmp_path / "sample.png")
    (tmp_path / "sample.txt").write_text("drop me", encoding="utf-8")
    data_cfg = FluxLoraDataConfig(
        instance_data_dir=str(tmp_path),
        caption_strategy="filename",
        resolution=512,
        caption_dropout_probability=1.0,
    )
    dataset = FluxLoraImageDataset(data_cfg, seed=0)
    assert dataset[0]["caption"] == ""
