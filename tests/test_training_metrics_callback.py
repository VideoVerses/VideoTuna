"""Tests for training metrics callback."""

import json
import os
import tempfile
from unittest import mock

from videotuna.utils.callbacks import TrainingMetricsCallback


def test_training_metrics_callback_writes_metrics_json():
    callback = TrainingMetricsCallback()
    trainer = mock.MagicMock()
    trainer.current_epoch = 0
    trainer.global_rank = 0
    trainer.global_step = 42
    pl_module = mock.MagicMock()
    pl_module.logdir = None
    trainer.logger = mock.MagicMock()

    with tempfile.TemporaryDirectory() as tmpdir:
        callback.save_dir = tmpdir
        callback.on_train_epoch_start(trainer, pl_module)
        callback.on_train_epoch_end(trainer, pl_module)

        metrics_path = os.path.join(tmpdir, "metrics.json")
        assert os.path.isfile(metrics_path)
        with open(metrics_path) as f:
            data = json.load(f)
        assert len(data["epochs"]) == 1
        assert "epoch_time_s" in data["epochs"][0]
        assert "peak_vram_gb" in data["epochs"][0]
        trainer.logger.log_metrics.assert_called_once()
        logged = trainer.logger.log_metrics.call_args[0][0]
        assert "epoch_time_s" in logged
        assert "peak_vram_gb" in logged
