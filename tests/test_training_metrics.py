"""Tests for training metrics backend resolution."""

from __future__ import annotations

from unittest import mock

import pytest

from videotuna.utils.training_metrics import (
    build_trackio_init_kwargs,
    describe_metrics_backend,
    log_validation_image_to_trackio,
    require_trackio,
    resolve_accelerate_log_with,
    trackio_enabled,
)


def test_resolve_accelerate_log_with_tensorboard_default():
    assert resolve_accelerate_log_with("tensorboard") == "tensorboard"


def test_resolve_accelerate_log_with_trackio_dual_mode(monkeypatch):
    monkeypatch.setattr(
        "videotuna.utils.training_metrics.trackio_available",
        lambda: True,
    )
    assert resolve_accelerate_log_with("trackio") == ["tensorboard", "trackio"]


def test_require_trackio_raises_when_missing(monkeypatch):
    monkeypatch.setattr(
        "videotuna.utils.training_metrics.trackio_available",
        lambda: False,
    )
    with pytest.raises(ImportError, match="poetry install -E trackio"):
        require_trackio()


def test_trackio_enabled():
    assert trackio_enabled("trackio") is True
    assert trackio_enabled("tensorboard") is False


def test_describe_metrics_backend():
    assert describe_metrics_backend("tensorboard") == "tensorboard"
    assert describe_metrics_backend("trackio") == "tensorboard + trackio"


def test_build_trackio_init_kwargs_without_space_id():
    assert build_trackio_init_kwargs(space_id=None) is None


def test_build_trackio_init_kwargs_with_space_id():
    assert build_trackio_init_kwargs(space_id="user/privtune-trackio") == {
        "trackio": {"space_id": "user/privtune-trackio"}
    }


def test_log_validation_image_to_trackio(monkeypatch):
    fake_image = mock.MagicMock()
    fake_trackio = mock.MagicMock()
    fake_trackio.Image = mock.MagicMock(side_effect=lambda img: ("image", img))

    monkeypatch.setattr(
        "videotuna.utils.training_metrics.trackio_available",
        lambda: True,
    )
    monkeypatch.setitem(__import__("sys").modules, "trackio", fake_trackio)

    log_validation_image_to_trackio(fake_image, step=42)

    fake_trackio.Image.assert_called_once_with(fake_image)
    fake_trackio.log.assert_called_once_with(
        {"validation/sample": ("image", fake_image)},
        step=42,
    )
