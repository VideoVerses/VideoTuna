"""Save Flux LoRA checkpoints in Diffusers-compatible format."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from diffusers import FluxPipeline
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def save_lora_checkpoint(transformer, output_dir: str | Path, step: int) -> Path:
    save_path = Path(output_dir) / f"checkpoint-{step}"
    save_path.mkdir(parents=True, exist_ok=True)

    transformer_lora = get_peft_model_state_dict(transformer)
    FluxPipeline.save_lora_weights(
        save_directory=str(save_path),
        transformer_lora_layers=transformer_lora,
    )
    return save_path


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    if not root.is_dir():
        return None
    best_step = -1
    best_path: Path | None = None
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = _CHECKPOINT_RE.match(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if step > best_step:
            best_step = step
            best_path = path
    return best_path


def checkpoint_step(path: Path) -> int:
    match = _CHECKPOINT_RE.match(path.name)
    if match is None:
        raise ValueError(f"Not a checkpoint directory: {path}")
    return int(match.group(1))


def load_lora_checkpoint(transformer, checkpoint_dir: str | Path) -> None:
    path = Path(checkpoint_dir)
    lora_state_dict = FluxPipeline.lora_state_dict(str(path))
    set_peft_model_state_dict(transformer, lora_state_dict)


def prune_checkpoints(output_dir: str | Path, limit: int | None) -> None:
    if limit is None or limit <= 0:
        return
    root = Path(output_dir)
    checkpoints: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = _CHECKPOINT_RE.match(path.name)
        if match is None:
            continue
        checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0])
    while len(checkpoints) > limit:
        _, path = checkpoints.pop(0)
        shutil.rmtree(path)
