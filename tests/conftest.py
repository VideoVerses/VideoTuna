import warnings

import pytest

try:
    from sentry_sdk.hub import SentryHubDeprecationWarning
except ImportError:
    SentryHubDeprecationWarning = DeprecationWarning  # type: ignore[misc,assignment]


@pytest.fixture(autouse=True)
def _suppress_third_party_import_warnings():
    """Optional third-party deps emit noisy warnings on import-only smoke tests."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SentryHubDeprecationWarning)
        warnings.filterwarnings(
            "ignore",
            message="builtin type SwigPyPacked has no __module__ attribute",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="builtin type SwigPyObject has no __module__ attribute",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="User provided device_type of 'cuda', but CUDA is not available",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="`torch.cuda.amp.autocast",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="`torch.utils._pytree._register_pytree_node` is deprecated",
            category=FutureWarning,
            module=r"colossalai\..*",
        )
        warnings.filterwarnings(
            "ignore",
            message="is already registered as pytree node",
            category=UserWarning,
            module=r"torch\..*",
        )
        yield


def pytest_collection_modifyitems(config, items):
    try:
        import torch
        from torch import version as torch_version
    except (ImportError, OSError, ValueError):
        return

    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    is_rocm = (
        torch.cuda.is_available() and getattr(torch_version, "hip", None) is not None
    )
    if not is_rocm:
        skip_rocm = pytest.mark.skip(reason="ROCm not available")
        for item in items:
            if "rocm" in item.keywords:
                item.add_marker(skip_rocm)
