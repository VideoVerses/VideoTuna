"""Tests for unified compute backend detection."""

from unittest import mock

import pytest
import torch

from videotuna.utils import device_utils


def test_gpu_is_available_alias():
    assert device_utils.cuda_is_available() == device_utils.gpu_is_available()


def test_normalize_device_prefer():
    assert device_utils.normalize_device_prefer(None) is None
    assert device_utils.normalize_device_prefer("cuda") == "cuda"
    assert device_utils.normalize_device_prefer("cuda:1") == "cuda:1"
    assert device_utils.normalize_device_prefer(1) == "cuda:1"
    assert device_utils.normalize_device_prefer("0") == "cuda:0"
    assert device_utils.normalize_device_prefer("cpu") == "cpu"
    with pytest.raises(ValueError, match="MPS is not supported"):
        device_utils.normalize_device_prefer("mps")


def test_normalize_device_prefer_invalid():
    with pytest.raises(ValueError, match="Invalid device"):
        device_utils.normalize_device_prefer("invalid")


def test_resolve_inference_device_explicit_cpu():
    with mock.patch.object(device_utils, "gpu_is_available", return_value=True):
        dev = device_utils.resolve_inference_device("cpu")
    assert dev == torch.device("cpu")


def test_resolve_inference_device_cpu_when_no_gpu():
    with mock.patch.object(device_utils, "gpu_is_available", return_value=False):
        assert device_utils.resolve_inference_device() == torch.device("cpu")


def test_resolve_inference_device_cuda_when_gpu():
    with mock.patch.object(device_utils, "gpu_is_available", return_value=True):
        with mock.patch.object(
            device_utils, "detect_compute_backend", return_value="cuda"
        ):
            with mock.patch.object(device_utils.torch.cuda, "set_device"):
                with mock.patch.object(
                    device_utils.torch.cuda, "device_count", return_value=2
                ):
                    dev = device_utils.resolve_inference_device()
    assert dev == torch.device("cuda", 0)


def test_resolve_inference_device_indexed():
    with mock.patch.object(device_utils, "gpu_is_available", return_value=True):
        with mock.patch.object(device_utils.torch.cuda, "set_device") as set_dev:
            with mock.patch.object(
                device_utils.torch.cuda, "device_count", return_value=2
            ):
                dev = device_utils.resolve_inference_device("cuda:1")
    assert dev == torch.device("cuda", 1)
    set_dev.assert_called_with(1)


def test_resolve_inference_device_rejects_cuda_without_gpu():
    with mock.patch.object(device_utils, "gpu_is_available", return_value=False):
        with pytest.raises(RuntimeError, match="no GPU accelerator"):
            device_utils.resolve_inference_device("cuda")


def test_recommend_dtype_cpu():
    dev = torch.device("cpu")
    assert device_utils.recommend_dtype(dev) == "fp32"


def test_recommend_dtype_ampere():
    dev = torch.device("cuda", 0)
    with mock.patch.object(device_utils, "gpu_is_available", return_value=True):
        with mock.patch.object(
            device_utils.torch.cuda, "get_device_capability", return_value=(8, 6)
        ):
            assert device_utils.recommend_dtype(dev) == "bf16"


def test_recommend_dtype_turing():
    dev = torch.device("cuda", 0)
    with mock.patch.object(device_utils, "gpu_is_available", return_value=True):
        with mock.patch.object(
            device_utils.torch.cuda, "get_device_capability", return_value=(7, 5)
        ):
            assert device_utils.recommend_dtype(dev) == "fp16"


def test_require_min_vram_raises():
    dev = torch.device("cuda", 0)
    props = mock.Mock()
    props.total_memory = 8 * 1024**3
    with mock.patch.object(device_utils, "gpu_is_available", return_value=True):
        with mock.patch.object(
            device_utils.torch.cuda, "get_device_properties", return_value=props
        ):
            with mock.patch.object(
                device_utils, "_format_hardware_context", return_value=""
            ):
                with pytest.raises(RuntimeError, match="below required"):
                    device_utils.require_min_vram(16.0, device=dev, context="test")


def test_get_visible_gpus_empty_without_gpu():
    with mock.patch.object(device_utils, "gpu_is_available", return_value=False):
        assert device_utils.get_visible_gpus() == []


def test_get_visible_gpus_mocked():
    props = mock.Mock()
    props.name = "RTX 4090"
    props.major = 8
    props.minor = 9
    props.total_memory = 24 * 1024**3
    with mock.patch.object(device_utils, "gpu_is_available", return_value=True):
        with mock.patch.object(device_utils.torch.cuda, "device_count", return_value=1):
            with mock.patch.object(
                device_utils.torch.cuda, "get_device_properties", return_value=props
            ):
                with mock.patch.object(
                    device_utils.torch.cuda,
                    "mem_get_info",
                    return_value=(8 * 1024**3, 24 * 1024**3),
                ):
                    gpus = device_utils.get_visible_gpus()
    assert len(gpus) == 1
    assert gpus[0].name == "RTX 4090"
    assert gpus[0].supports_bf16 is True


def test_empty_cache_aliases():
    assert device_utils.empty_cache is device_utils.empty_accelerator_cache
    assert device_utils.synchronize_device is device_utils.synchronize_accelerator


def test_detect_compute_backend_cpu():
    with mock.patch.object(device_utils.torch.cuda, "is_available", return_value=False):
        assert device_utils.detect_compute_backend() == "cpu"


def test_detect_compute_backend_cuda():
    with mock.patch.object(device_utils.torch.cuda, "is_available", return_value=True):
        with mock.patch.object(device_utils, "_torch_hip_version", return_value=None):
            assert device_utils.detect_compute_backend() == "cuda"


def test_detect_compute_backend_rocm():
    with mock.patch.object(device_utils.torch.cuda, "is_available", return_value=True):
        with mock.patch.object(
            device_utils, "_torch_hip_version", return_value="6.2.4"
        ):
            assert device_utils.detect_compute_backend() == "rocm"


def test_describe_compute_environment_rocm():
    with mock.patch.object(
        device_utils, "_detect_compute_backend_raw", return_value="rocm"
    ):
        with mock.patch.object(
            device_utils.torch.cuda, "get_device_name", return_value="gfx1100"
        ):
            with mock.patch.object(
                device_utils, "_torch_hip_version", return_value="6.2.4"
            ):
                with mock.patch.object(device_utils.torch, "__version__", "2.6.0"):
                    desc = device_utils.describe_compute_environment()
    assert "ROCm available" in desc
    assert "gfx1100" in desc
    assert "HIP 6.2.4" in desc


def test_require_accelerator_for_flow_raises_without_gpu():
    with mock.patch.object(device_utils, "gpu_is_available", return_value=False):
        with pytest.raises(RuntimeError, match="requires a GPU"):
            device_utils.require_accelerator_for_flow(
                "videotuna.flow.wanvideo.WanVideoModelFlow"
            )


def test_get_flow_tier_wan_720p_gpu_required():
    tier = device_utils.get_flow_tier(
        device_utils._DIFFUSERS_FLOW,
        model_family="wan",
        height=720,
        width=1280,
    )
    assert tier == "gpu_required"


def test_get_flow_tier_flux_schnell_cpu_smoke():
    tier = device_utils.get_flow_tier(
        device_utils._DIFFUSERS_FLOW,
        model_family="flux",
        model_variant="schnell",
        height=768,
        width=1360,
    )
    assert tier == "cpu_smoke"


def test_get_flow_tier_flux_dev_cpu_smoke_at_512():
    tier = device_utils.get_flow_tier(
        device_utils._DIFFUSERS_FLOW,
        model_family="flux",
        model_variant="1-dev",
        height=256,
        width=256,
    )
    assert tier == "cpu_smoke"


def test_get_flow_tier_flux_dev_gpu_required_above_512():
    tier = device_utils.get_flow_tier(
        device_utils._DIFFUSERS_FLOW,
        model_family="flux",
        model_variant="dev",
        height=768,
        width=512,
    )
    assert tier == "gpu_required"


def test_get_flow_tier_unknown_family_defaults():
    tier = device_utils.get_flow_tier(
        device_utils._DIFFUSERS_FLOW,
        model_family="unknown",
        height=256,
        width=256,
    )
    assert tier == "cpu_smoke"


def test_require_accelerator_wan_native_720p_blocks_cpu_smoke():
    with mock.patch.object(device_utils, "gpu_is_available", return_value=False):
        with pytest.raises(RuntimeError, match="requires a GPU"):
            device_utils.require_accelerator_for_flow(
                device_utils._WAN_FLOW,
                cpu_mode="smoke",
                height=720,
                width=1280,
                frames=81,
            )


def test_resolve_cpu_mode_smoke_cli():
    with mock.patch.dict("os.environ", {}, clear=True):
        assert device_utils.resolve_cpu_mode(cli_smoke=True) == "smoke"


def test_resolve_cpu_mode_legacy_allow_cpu():
    with mock.patch.dict(
        "os.environ",
        {"VIDEOTUNA_ALLOW_CPU_INFERENCE": "1"},
        clear=True,
    ):
        assert device_utils.resolve_cpu_mode() == "force"


def test_require_accelerator_for_flow_allow_cpu():
    device_utils.require_accelerator_for_flow(
        "videotuna.flow.wanvideo.WanVideoModelFlow",
        allow_cpu=True,
    )


def test_compute_backend_env_rocm_mismatch():
    with mock.patch.dict("os.environ", {"VIDEOTUNA_COMPUTE_BACKEND": "rocm"}):
        with mock.patch.object(device_utils, "_torch_hip_version", return_value=None):
            with pytest.raises(RuntimeError, match="not built with HIP"):
                device_utils.detect_compute_backend()


def test_compute_backend_env_mps_rejected():
    with mock.patch.dict("os.environ", {"VIDEOTUNA_COMPUTE_BACKEND": "mps"}):
        with pytest.raises(ValueError, match="mps is not supported"):
            device_utils.detect_compute_backend()


def test_require_xfuser_sequence_parallel_on_rocm():
    with mock.patch.object(device_utils, "detect_compute_backend", return_value="rocm"):
        with pytest.raises(RuntimeError, match="xfuser requires NVIDIA CUDA"):
            device_utils.require_xfuser_sequence_parallel("TestFlow")


def test_validate_sequence_parallel_degrees_mismatch():
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        device_utils.validate_sequence_parallel_degrees(2, 2, world_size=3)


def test_validate_sequence_parallel_degrees_ok():
    device_utils.validate_sequence_parallel_degrees(2, 2, world_size=4)


def test_accelerator_helpers_noop_on_cpu():
    with mock.patch.object(device_utils, "gpu_is_available", return_value=False):
        assert device_utils.accelerator_device_string() == "cpu"
        device_utils.empty_accelerator_cache()
        device_utils.synchronize_accelerator()
