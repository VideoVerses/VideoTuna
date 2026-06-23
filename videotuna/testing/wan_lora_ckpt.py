"""Synthetic Wan 2.1 native LoRA checkpoints for tests and bridge spikes."""

from __future__ import annotations

from pathlib import Path

import torch


def build_synthetic_wan_lora_ckpt(
    path: Path,
    *,
    num_blocks: int = 2,
    rank: int = 16,
) -> Path:
    """Write a synthetic denoiser ckpt with production-style key names."""
    dim_in, dim_mid, dim_out = 5120, 13824, 5120
    state: dict[str, torch.Tensor] = {}
    for i in range(num_blocks):
        for p in ("q", "k", "v", "o"):
            state[f"denoiser.blocks.{i}.self_attn.{p}.lora_A.weight"] = torch.randn(
                rank, dim_in
            )
            state[f"denoiser.blocks.{i}.self_attn.{p}.lora_B.weight"] = torch.randn(
                dim_in, rank
            )
        state[f"denoiser.blocks.{i}.ffn.0.lora_A.weight"] = torch.randn(rank, dim_in)
        state[f"denoiser.blocks.{i}.ffn.0.lora_B.weight"] = torch.randn(dim_mid, rank)
        state[f"denoiser.blocks.{i}.ffn.2.lora_A.weight"] = torch.randn(rank, dim_mid)
        state[f"denoiser.blocks.{i}.ffn.2.lora_B.weight"] = torch.randn(dim_out, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state}, path)
    return path
