"""Tests for central loguru logging configuration."""

from __future__ import annotations

import io

import pytest
from loguru import logger

from videotuna.utils import logging_config
from videotuna.utils.logging_config import (
    bound_logger,
    configure_logging,
    phase_from_wan_task,
    resolve_device_label,
)


@pytest.fixture(autouse=True)
def reset_loguru_handlers():
    logger.remove()
    logging_config._configured = False
    logging_config._stderr_handler_id = None
    logging_config._file_handler_id = None
    yield
    logger.remove()
    logging_config._configured = False
    logging_config._stderr_handler_id = None
    logging_config._file_handler_id = None


def test_configure_logging_respects_log_level(monkeypatch, capsys):
    monkeypatch.setenv("VIDEOTUNA_LOG_LEVEL", "WARNING")
    configure_logging()

    bound_logger(phase="t2i", flow="flux_lora").debug("hidden")
    bound_logger(phase="t2i", flow="flux_lora").warning("visible")

    captured = capsys.readouterr()
    assert "hidden" not in captured.err
    assert "visible" in captured.err


def test_bound_logger_emits_structured_context():
    configure_logging()

    output = io.StringIO()
    handler_id = logger.add(
        output,
        level="INFO",
        format=(
            "phase={extra[phase]} flow={extra[flow]} "
            "device={extra[device]} | {message}"
        ),
    )

    bound_logger(phase="t2v", flow="wanvideo", device="cuda:0").info("hello")

    logger.remove(handler_id)
    text = output.getvalue()
    assert "phase=t2v" in text
    assert "flow=wanvideo" in text
    assert "device=cuda:0" in text
    assert "hello" in text


def test_configure_logging_is_idempotent():
    configure_logging()
    first_count = len(logger._core.handlers)  # noqa: SLF001

    configure_logging()
    second_count = len(logger._core.handlers)  # noqa: SLF001

    assert first_count == second_count


def test_configure_logging_adds_file_sink(tmp_path):
    log_file = tmp_path / "train.log"
    configure_logging(log_file=log_file)

    bound_logger(phase="t2i", flow="flux_lora").info("file test")

    text = log_file.read_text(encoding="utf-8")
    assert text
    assert "file test" in text


def test_resolve_device_label():
    import torch

    assert resolve_device_label(None) == "-"
    assert resolve_device_label("cuda:1") == "cuda:1"
    assert resolve_device_label(torch.device("cpu")) == "cpu"
    assert resolve_device_label(torch.device("cuda", 0)) == "cuda:0"


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("t2v-14B", "t2v"),
        ("i2v-14B", "i2v"),
        ("t2i-14B", "t2i"),
        ("unknown", "inference"),
    ],
)
def test_phase_from_wan_task(task, expected):
    assert phase_from_wan_task(task) == expected


def test_configure_logging_adds_stderr_handler():
    configure_logging()
    assert len(logger._core.handlers) >= 1  # noqa: SLF001
