import importlib

import pytest
import torch
from packaging.version import Version

INFERENCE_BACKENDS = [
    "videotuna.flow.diffusers_video",
    "videotuna.flow.wanvideo",
]

TRAINING_BACKENDS = [
    ("videotuna.training.flux_lora.config", None),
]

GPU_BACKENDS = [
    "videotuna.flow.wanvideo",
]


def test_wan_t5_encoder_no_cuda_default_arg():
    """T5EncoderModel must not use torch.cuda at class definition time."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (root / "videotuna/models/wan/wan/modules/t5.py").read_text()
    assert "device=torch.cuda.current_device()" not in source
    assert "device=None" in source


@pytest.mark.parametrize("module", INFERENCE_BACKENDS)
def test_inference_backend_import(module):
    importlib.import_module(module)


@pytest.mark.parametrize("module,extra", TRAINING_BACKENDS)
def test_training_backend_import(module, extra):
    if extra is not None:
        pytest.importorskip(extra)
    importlib.import_module(module)


@pytest.mark.parametrize("module", GPU_BACKENDS)
def test_gpu_backend_import(module):
    importlib.import_module(module)


def test_core_ml_stack_versions():
    import accelerate
    import diffusers
    import peft
    import transformers

    assert (
        Version(torch.__version__).major == 2 and Version(torch.__version__).minor >= 6
    )
    assert Version(diffusers.__version__) >= Version("0.38.0")
    assert Version(transformers.__version__) >= Version("4.48.0")
    assert Version(accelerate.__version__) >= Version("1.14.0")
    assert Version(peft.__version__) >= Version("0.17.0")


def test_training_stack_versions():
    pytest.importorskip("deepspeed")
    import deepspeed

    assert Version(deepspeed.__version__) >= Version("0.19.0")
