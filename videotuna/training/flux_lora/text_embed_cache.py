"""On-disk text embedding cache for Flux LoRA training."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from videotuna.utils.logging_config import bound_logger

logger = bound_logger(phase="t2i", flow="flux_lora")


def _caption_cache_path(cache_dir: Path, caption: str) -> Path:
    digest = hashlib.sha256(caption.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.pt"


def _load_cached_embed(path: Path) -> dict[str, torch.Tensor] | None:
    if not path.is_file():
        return None
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        return None
    return data


def _save_cached_embed(path: Path, embeds: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prompt_embeds": embeds["prompt_embeds"].cpu(),
            "pooled_prompt_embeds": embeds["pooled_prompt_embeds"].cpu(),
            "text_ids": embeds["text_ids"].cpu(),
        },
        path,
    )


def build_or_load_cache(
    pipeline: Any,
    captions: list[str],
    cache_dir: str | Path,
    write_batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Build or load cached prompt embeddings keyed by caption text."""
    cache_root = Path(cache_dir)
    unique_captions = list(dict.fromkeys(captions))
    lookup: dict[str, dict[str, torch.Tensor]] = {}
    pending: list[str] = []

    for caption in unique_captions:
        path = _caption_cache_path(cache_root, caption)
        cached = _load_cached_embed(path)
        if cached is not None:
            lookup[caption] = cached
        else:
            pending.append(caption)

    if not pending:
        logger.info("Text embed cache hit for all {} captions", len(unique_captions))
        return lookup

    logger.info(
        "Encoding {} / {} captions into cache (write_batch_size={})",
        len(pending),
        len(unique_captions),
        write_batch_size,
    )
    pipeline.text_encoder.to(device)
    pipeline.text_encoder_2.to(device)

    for start in range(0, len(pending), write_batch_size):
        batch = pending[start : start + write_batch_size]
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt=batch,
                prompt_2=batch,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )
        for idx, caption in enumerate(batch):
            embeds = {
                "prompt_embeds": prompt_embeds[idx : idx + 1].cpu(),
                "pooled_prompt_embeds": pooled_prompt_embeds[idx : idx + 1].cpu(),
                "text_ids": text_ids[idx : idx + 1].cpu(),
            }
            lookup[caption] = embeds
            _save_cached_embed(_caption_cache_path(cache_root, caption), embeds)

    return lookup
