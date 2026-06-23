"""Local image + caption dataset for Flux LoRA training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TypedDict

import torch
from PIL import Image
from torch.utils.data import BatchSampler, Dataset
from torchvision import transforms

from videotuna.training.flux_lora.bucketing import (
    bucket_dimensions_for_image,
    meets_minimum_size,
)
from videotuna.training.flux_lora.config import FluxLoraDataConfig
from videotuna.utils.logging_config import bound_logger

logger = bound_logger(phase="t2i", flow="flux_lora")

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class FluxLoraSample(TypedDict, total=False):
    pixel_values: torch.Tensor
    caption: str
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    text_ids: torch.Tensor


def _load_caption(
    image_path: Path, caption_strategy: str, default_caption: str | None
) -> str:
    if caption_strategy == "filename":
        txt_path = image_path.with_suffix(".txt")
        if txt_path.is_file():
            return txt_path.read_text(encoding="utf-8").strip()
        if default_caption:
            return default_caption
        raise ValueError(
            f"Missing caption file for {image_path} (caption_strategy=filename)"
        )
    if default_caption:
        return default_caption
    raise ValueError(
        f"Unsupported caption_strategy={caption_strategy!r} without default caption"
    )


def _center_square_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


class FluxBucketBatchSampler(BatchSampler):
    """Batch indices grouped by target bucket dimensions."""

    def __init__(
        self,
        bucket_ids: list[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.bucket_ids = bucket_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        buckets: dict[int, list[int]] = {}
        for index, bucket_id in enumerate(self.bucket_ids):
            buckets.setdefault(bucket_id, []).append(index)
        batches: list[list[int]] = []
        for indices in buckets.values():
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start : start + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches
        self._epoch += 1

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self._grouped_indices().values()
        )

    def _grouped_indices(self) -> dict[int, list[int]]:
        buckets: dict[int, list[int]] = {}
        for index, bucket_id in enumerate(self.bucket_ids):
            buckets.setdefault(bucket_id, []).append(index)
        return buckets


class FluxLoraImageDataset(Dataset):
    def __init__(
        self,
        data_config: FluxLoraDataConfig,
        *,
        embed_lookup: dict[str, dict[str, torch.Tensor]] | None = None,
        seed: int = 42,
    ):
        self.data_dir = Path(data_config.instance_data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Training data directory not found: {self.data_dir}"
            )

        self.caption_strategy = data_config.caption_strategy
        self.default_caption = data_config.default_caption
        self.resolution = data_config.resolution
        self.crop = data_config.crop
        self.resolution_type = data_config.resolution_type
        self.aspect_bucket_rounding = data_config.aspect_bucket_rounding
        self.minimum_image_size = data_config.minimum_image_size
        self.caption_dropout_probability = data_config.caption_dropout_probability
        self.embed_lookup = embed_lookup or {}
        self._rng = random.Random(seed)

        self.samples: list[tuple[Path, str, tuple[int, int]]] = []
        self.bucket_ids: list[int] = []
        bucket_map: dict[tuple[int, int], int] = {}
        filtered = 0

        for path in sorted(self.data_dir.iterdir()):
            if path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            with Image.open(path) as image:
                width, height = image.size
            if not meets_minimum_size(
                width,
                height,
                self.minimum_image_size,
                self.resolution_type,
            ):
                filtered += 1
                continue
            if self.crop:
                side = min(width, height)
                width = height = side
            bucket_w, bucket_h = bucket_dimensions_for_image(
                width,
                height,
                self.resolution,
                self.resolution_type,
                self.aspect_bucket_rounding,
            )
            caption = _load_caption(path, self.caption_strategy, self.default_caption)
            bucket_key = (bucket_w, bucket_h)
            if bucket_key not in bucket_map:
                bucket_map[bucket_key] = len(bucket_map)
            self.samples.append((path, caption, bucket_key))
            self.bucket_ids.append(bucket_map[bucket_key])

        if filtered:
            logger.info("Filtered {} images below minimum_image_size", filtered)

        if not self.samples:
            raise ValueError(f"No training images found in {self.data_dir}")

        logger.info(
            "Loaded {} training images from {} across {} aspect buckets",
            len(self.samples),
            self.data_dir,
            len(bucket_map),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _maybe_dropout_caption(self, caption: str) -> str:
        if (
            self.caption_dropout_probability > 0.0
            and self._rng.random() < self.caption_dropout_probability
        ):
            return ""
        return caption

    def __getitem__(self, index: int) -> FluxLoraSample:
        path, caption, (bucket_w, bucket_h) = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.crop:
            image = _center_square_crop(image)
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (bucket_h, bucket_w),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        pixel_values = transform(image)
        caption = self._maybe_dropout_caption(caption)
        sample: FluxLoraSample = {"pixel_values": pixel_values, "caption": caption}
        cached = self.embed_lookup.get(caption)
        if cached is not None:
            sample["prompt_embeds"] = cached["prompt_embeds"]
            sample["pooled_prompt_embeds"] = cached["pooled_prompt_embeds"]
            sample["text_ids"] = cached["text_ids"]
        return sample
