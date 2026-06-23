"""Tests for Wan 2.1 native I2V LoRA → Wan 2.2 I2V Diffusers bridge."""

from unittest.mock import MagicMock, patch

from videotuna.utils.wan_lora_bridge import apply_native_wan_lora_to_i2v_pipeline


def test_i2v_bridge_delegates_to_t2v_bridge():
    pipeline = MagicMock()
    with patch(
        "videotuna.utils.wan_lora_bridge.apply_native_wan_lora_to_pipeline"
    ) as mock_apply:
        mock_apply.return_value = []
        apply_native_wan_lora_to_i2v_pipeline(
            pipeline, "/tmp/denoiser.ckpt", lora_scale=0.8
        )
        mock_apply.assert_called_once_with(
            pipeline, "/tmp/denoiser.ckpt", lora_scale=0.8, lora_scale_2=None
        )
