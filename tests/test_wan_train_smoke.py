"""Smoke benchmark for training data path (CPU-only, no checkpoints)."""

import time

import torch
from torch.utils.data import Dataset

from videotuna.data.lightningdata import DataModuleFromConfig


class _SmokeDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return {
            "video": torch.randn(3, 8, 64, 64),
            "caption": f"cap-{idx}",
        }


def test_dataloader_epoch_smoke_benchmark():
    """Pseudo-epoch iteration with hardened DataLoader settings."""
    dm = DataModuleFromConfig(
        batch_size=2,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
        train={
            "target": "tests.test_wan_train_smoke._SmokeDataset",
            "params": {},
        },
    )
    dm.setup()
    loader = dm.train_dataloader()

    start = time.perf_counter()
    batches = list(loader)
    elapsed = time.perf_counter() - start

    assert len(batches) == 4
    assert batches[0]["video"].shape == (2, 3, 8, 64, 64)
    assert elapsed < 30.0
