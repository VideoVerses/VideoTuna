"""Tests for training DataLoader configuration."""

import pytest
import torch
from torch.utils.data import Dataset

from videotuna.data.lightningdata import DataModuleFromConfig


class _TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {"video": torch.zeros(3, 2, 8, 8), "caption": f"item-{idx}"}


@pytest.fixture
def tiny_datamodule_config():
    return {
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "train": {
            "target": "tests.test_training_dataloader._TinyDataset",
            "params": {},
        },
    }


def test_datamodule_dataloader_kwargs(tiny_datamodule_config):
    dm = DataModuleFromConfig(**tiny_datamodule_config)
    dm.setup()
    loader = dm.train_dataloader()
    assert loader.batch_size == 2
    assert loader.pin_memory is True
    assert loader.num_workers == 0


def test_datamodule_collate_default_batch(tiny_datamodule_config):
    dm = DataModuleFromConfig(**tiny_datamodule_config)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert batch["video"].shape[0] == 2
    assert len(batch["caption"]) == 2


def test_default_num_workers_not_batch_scaled():
    dm = DataModuleFromConfig(
        batch_size=8,
        train={
            "target": "tests.test_training_dataloader._TinyDataset",
            "params": {},
        },
    )
    assert dm.num_workers == 4
