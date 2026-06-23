"""Tests for Wan checkpoint loading."""

import pytest

from videotuna.models.wan.wan.modules.model import WanModel


def test_wan_from_pretrained_missing_dir():
    with pytest.raises(FileNotFoundError, match="Wan checkpoint directory not found"):
        WanModel.from_pretrained("/nonexistent/wan/checkpoint")
