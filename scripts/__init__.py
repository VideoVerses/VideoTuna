"""
Poetry commands
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime

current_time = datetime.now().strftime("%Y%m%d%H%M%S")

FLUX_T2I_CONFIG = "configs/domain/flux_t2i.json"
FLUX_T2I_DATA_CONFIG = "configs/domain/flux_t2i_data.json"
FLUX_T2I_CLOUD_SMOKE = "configs/domain/flux_t2i_cloud_smoke.json"
WAN_T2V_LORA_CONFIG = "configs/domain/wan_t2v_lora.yaml"
WAN_T2V_LORA_CLOUD_SMOKE = "configs/domain/wan_t2v_lora_cloud_smoke.yaml"
WAN_I2V_LORA_CONFIG = "configs/domain/wan_i2v_lora.yaml"

# Single source of truth for CPU CI smoke tests (see .github/workflows/cpu.yml).
CI_SMOKE_TESTS = [
    "tests/test_import_smoke.py",
    "tests/test_domain_finetune_configs.py",
    "tests/test_flux_lora_train_smoke.py",
    "tests/test_wan_lora_bridge.py",
    "tests/test_wan_i2v_lora_bridge.py",
    "tests/test_wan_domain_lora_smoke_22_config.py",
    "tests/test_wan_domain_i2v_smoke_22_config.py",
    "tests/test_wan_i2v_dataset.py",
    "tests/test_wan_training_step.py",
    "tests/test_poetry_scripts.py",
    "tests/test_diffusers_quantization.py",
]

# Line-coverage floor for videotuna/training/ + videotuna/utils/
# (CI smoke baseline ~36%).
COVERAGE_GATE_FAIL_UNDER = 33


def _require_cuda_backend(installer_name: str) -> None:
    """Abort when the active PyTorch build is ROCm (CUDA-only installer)."""
    try:
        from videotuna.utils.device_utils import detect_compute_backend

        if detect_compute_backend() == "rocm":
            print(
                f"{installer_name} is not supported on AMD ROCm.\n"
                "Use VIDEOTUNA_ATTN_BACKEND=sdpa for attention on ROCm.\n"
                "See docs/install-rocm.md.",
                file=sys.stderr,
            )
            sys.exit(1)
    except ImportError:
        pass


def install_deepspeed():
    """
    Install DeepSpeed with CUDA 12.6 toolkit support
    (rebuilds against the active torch).

    When conda is unavailable, skips the CUDA toolkit step and installs via pip.
    If deepspeed>=0.19.2 is already importable, exits successfully without rebuilding.
    """
    _require_cuda_backend("install-deepspeed")
    try:
        import deepspeed
        from packaging.version import Version

        if Version(deepspeed.__version__) >= Version("0.19.2"):
            print(
                f"deepspeed {deepspeed.__version__} already installed "
                "(>= 0.19.2); skipping rebuild."
            )
            return
    except ImportError:
        pass

    if shutil.which("conda"):
        command_install_cuda_toolkit = [
            "conda",
            "install",
            "cuda-toolkit=12.6",
            "-c",
            "conda-forge",
            "-c",
            "nvidia",
            "-y",
        ] + sys.argv[1:]
        result_cuda_toolkit = subprocess.run(command_install_cuda_toolkit, check=False)
        if result_cuda_toolkit.returncode != 0:
            print(
                "conda cuda-toolkit install failed; continuing with pip-only "
                "deepspeed install.",
                file=sys.stderr,
            )
    else:
        print(
            "conda not found; skipping cuda-toolkit install. "
            "If the pip build fails, install CUDA/nvcc or use conda.",
            file=sys.stderr,
        )

    pip = [sys.executable, "-m", "pip"]
    subprocess.run([*pip, "uninstall", "deepspeed", "-y"], check=False)

    env = os.environ.copy()
    env["DS_BUILD_CPU_ADAM"] = "1"
    env["BUILD_UTILS"] = "1"
    result_deepspeed = subprocess.run(
        [*pip, "install", "deepspeed==0.19.2"],
        check=False,
        env=env,
    )
    exit(result_deepspeed.returncode)


def _python_wheel_tag() -> str:
    major, minor = sys.version_info[:2]
    return f"cp{major}{minor}"


def _torch_cuda_wheel_tag() -> str:
    """Map torch.version.cuda to flash-attn wheel tag (e.g. cu126)."""
    try:
        import torch
        import torch.version

        cuda = getattr(torch.version, "cuda", None)
        if cuda is None:
            return "cu12"
        parts = str(cuda).split(".")
        if len(parts) >= 2:
            return f"cu{parts[0]}{parts[1]}"
    except ImportError:
        pass
    return "cu126"


def _torch_minor_for_flash() -> str:
    import torch

    return ".".join(torch.__version__.split(".")[:2])


def _flash_attn_wheel_url() -> str:
    wheel_tag = _python_wheel_tag()
    cuda_tag = _torch_cuda_wheel_tag()
    torch_minor = _torch_minor_for_flash()
    return (
        "https://github.com/Dao-AILab/flash-attention/releases/download/"
        f"v2.7.4.post1/flash_attn-2.7.4.post1+{cuda_tag}torch{torch_minor}cxx11abiTRUE-"
        f"{wheel_tag}-{wheel_tag}-linux_x86_64.whl"
    )


def install_flash_attn():
    """
    Install flash-attn for PyTorch 2.6 + CUDA 12.6 (cxx11 ABI wheels).

    Tries a prebuilt wheel first (no compiler or conda required). Falls back to a
    source build only when the wheel is unavailable.
    """
    _require_cuda_backend("install-flash-attn")
    try:
        import torch
        import torch.version

        if getattr(torch.version, "hip", None) is not None:
            print(
                "install-flash-attn requires an NVIDIA CUDA PyTorch build. "
                "Detected ROCm/HIP. See docs/install-rocm.md.",
                file=sys.stderr,
            )
            sys.exit(1)
        if getattr(torch.version, "cuda", None) is None:
            print(
                "install-flash-attn requires a CUDA PyTorch build. "
                "Run: poetry run install-cpu-torch is not compatible.",
                file=sys.stderr,
            )
            sys.exit(1)
    except ImportError:
        pass

    subprocess.run([sys.executable, "-m", "pip", "install", "ninja"], check=False)

    flash_attn_wheel = _flash_attn_wheel_url()
    result_wheel = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            flash_attn_wheel,
            "--no-build-isolation",
        ],
        check=False,
    )
    if result_wheel.returncode == 0:
        exit(0)

    if shutil.which("conda"):
        result_nvcc = subprocess.run(
            [
                "conda",
                "install",
                "-c",
                "nvidia",
                "cuda-nvcc=12.6",
                "-y",
            ]
            + sys.argv[1:],
            check=False,
        )
        if result_nvcc.returncode != 0:
            exit(result_nvcc.returncode)
    elif shutil.which("nvcc") is None:
        print(
            "Prebuilt flash-attn wheel install failed and nvcc was not found.\n"
            "Install the CUDA toolkit (nvcc), or use conda:\n"
            "  conda install -c nvidia cuda-nvcc=12.6",
            file=sys.stderr,
        )
        exit(result_wheel.returncode)

    result_flash = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "flash-attn==2.7.4.post1",
            "--no-build-isolation",
        ],
        check=False,
    )
    exit(result_flash.returncode)


_ROCM_TORCH_INDEX = "https://download.pytorch.org/whl/rocm6.2.4"
_CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
# Re-pin after pip torch installs (ROCm/CPU indexes may upgrade these transitively).
_POETRY_PINNED_DEPS = ("pillow==10.4.0", "numpy>=1.26,<2.3")
_CUDA_ONLY_PACKAGES = (
    "xformers",
    "bitsandbytes",
    "xfuser",
    "triton",
    "nvidia-cublas-cu12",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-cusparselt-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-nvtx-cu12",
)
# Keep triton on CPU installs — torchao/diffusers import torch._inductor which needs it.
_CPU_UNINSTALL_PACKAGES = tuple(p for p in _CUDA_ONLY_PACKAGES if p != "triton")


def _reconcile_poetry_pinned_deps(pip: list[str]) -> None:
    """Restore numpy/pillow versions required by videotuna and scipy."""
    subprocess.run([*pip, "install", *_POETRY_PINNED_DEPS], check=False)


def install_rocm():
    """
    Install PyTorch 2.6 + torchvision 0.21 for ROCm 6.2.4 and remove CUDA-only wheels.

    Uninstalls existing torch/torchvision first so pip does not keep a mismatched
    CUDA torchvision wheel (e.g. 0.21.0+cu126) alongside ROCm torch.

    Run after: poetry install -E rocm
    Re-run after any plain `poetry install` on AMD machines (lockfile pins CUDA torch).
    """
    pip = [sys.executable, "-m", "pip"]
    for pkg in (*_CUDA_ONLY_PACKAGES, "torch", "torchvision"):
        subprocess.run([*pip, "uninstall", pkg, "-y"], check=False)
    result = subprocess.run(
        [
            *pip,
            "install",
            "torch==2.6.0",
            "torchvision==0.21.0",
            "--index-url",
            _ROCM_TORCH_INDEX,
            "--force-reinstall",
            "--no-deps",
            "--no-cache-dir",
        ],
        check=False,
    )
    if result.returncode != 0:
        exit(result.returncode)
    subprocess.run(
        [
            *pip,
            "install",
            "pytorch-triton-rocm==3.2.0",
            "--index-url",
            _ROCM_TORCH_INDEX,
            "--no-deps",
            "--no-cache-dir",
        ],
        check=False,
    )
    _reconcile_poetry_pinned_deps(pip)

    import torch
    import torch.version
    import torchvision

    torch_build = torch.__version__
    tv_build = torchvision.__version__
    hip = getattr(torch.version, "hip", None)
    if hip is None:
        print(
            "WARNING: torch installed but torch.version.hip is None. "
            "Expected a ROCm wheel from the rocm6.2.4 index.",
            file=sys.stderr,
        )
    if "+cu" in tv_build:
        print(
            f"ERROR: torch/torchvision build mismatch: torch={torch_build}, "
            f"torchvision={tv_build}. Re-run: poetry run install-rocm",
            file=sys.stderr,
        )
        exit(1)

    print(f"torch {torch_build}, torchvision {tv_build}, HIP {hip}")
    try:
        from videotuna.utils.device_utils import describe_compute_environment

        print(describe_compute_environment())
    except ImportError:
        print(f"torch.cuda.is_available()={torch.cuda.is_available()}, " f"hip={hip}")
    exit(0)


def install_cpu_torch():
    """Install CPU-only PyTorch 2.6 wheels (no CUDA/ROCm)."""
    pip = [sys.executable, "-m", "pip"]
    for pkg in (*_CPU_UNINSTALL_PACKAGES, "torch", "torchvision"):
        subprocess.run([*pip, "uninstall", pkg, "-y"], check=False)
    result = subprocess.run(
        [
            *pip,
            "install",
            "torch==2.6.0",
            "torchvision==0.21.0",
            "--index-url",
            _CPU_TORCH_INDEX,
            "--force-reinstall",
            "--no-deps",
            "--no-cache-dir",
        ],
        check=False,
    )
    if result.returncode != 0:
        exit(result.returncode)
    subprocess.run(
        [*pip, "install", "triton==3.2.0", "--no-cache-dir"],
        check=False,
    )
    _reconcile_poetry_pinned_deps(pip)
    exit(result.returncode)


def install_flash_attn_rocm():
    """
    flash-attn is not officially supported on ROCm in VideoTuna.

    Use VIDEOTUNA_ATTN_BACKEND=sdpa instead. See docs/install-rocm.md.
    """
    print(
        "flash-attn is not supported on AMD ROCm in VideoTuna.\n"
        "Use: export VIDEOTUNA_ATTN_BACKEND=sdpa\n"
        "For experimental upstream builds, see "
        "https://github.com/Dao-AILab/flash-attention",
        file=sys.stderr,
    )
    sys.exit(1)


_FORMAT_TARGETS = ["."]
_LINT_TARGETS = ["videotuna", "tests", "scripts", "tools"]


def code_format(check=False):
    """
    Run the code formatting
    """
    cmds = (
        [
            ["ruff", "format", "--check", *_FORMAT_TARGETS],
            ["ruff", "check", "--select", "I", *_FORMAT_TARGETS],
        ]
        if check
        else [
            ["ruff", "check", "--fix", *_LINT_TARGETS],
            ["ruff", "check", "--select", "I", "--fix", *_FORMAT_TARGETS],
            ["ruff", "format", *_FORMAT_TARGETS],
        ]
    )
    return_code = 0

    for command in cmds:
        process = subprocess.run(command, check=False)
        if process.returncode != 0:
            return_code = process.returncode
            break

    exit(return_code)


def code_format_check():
    """
    Check the code formatting (useful with CI)
    """
    code_format(check=True)


def lint():
    """
    Run the linter
    """
    result = subprocess.run(
        ["ruff", "check", *_LINT_TARGETS] + sys.argv[1:], check=False
    )
    exit(result.returncode)


def test():  # pragma: no cover
    """
    Run all unittests
    """
    os.environ["ENV"] = "test"
    args = sys.argv[1:]
    has_targets = any(
        not a.startswith("-")
        and (a.endswith(".py") or "::" in a or a.startswith("tests"))
        for a in args
    )
    cmd = ["pytest"] + (args if has_targets else ["tests"] + args)
    result = subprocess.run(cmd, check=False)
    exit(result.returncode)


def coverage_report():
    """
    Run all unittests with coverage
    """
    os.environ["ENV"] = "test"
    result = subprocess.run(
        ["coverage", "run", "-m", "pytest", "--junitxml", "report.xml"], check=False
    )
    if result.returncode > 0:
        exit(result.returncode)
    result = subprocess.run(["coverage", "report", "-m"], check=False)
    exit(result.returncode)


def coverage_gate():
    """
    Run CI smoke tests with coverage and enforce a modest floor on core modules.
    """
    os.environ["ENV"] = "test"
    run = subprocess.run(
        ["coverage", "run", "-m", "pytest", "-q", *CI_SMOKE_TESTS],
        check=False,
    )
    if run.returncode != 0:
        exit(run.returncode)
    report = subprocess.run(
        [
            "coverage",
            "report",
            "--include=videotuna/training/*,videotuna/utils/*",
            f"--fail-under={COVERAGE_GATE_FAIL_UNDER}",
        ],
        check=False,
    )
    exit(report.returncode)


def type_check():
    """
    Run the type checking
    """
    result = subprocess.run(["mypy", "videotuna", "tests"], check=False)
    exit(result.returncode)


def inference_flux_lora():
    from videotuna.cli.inference_app import inference_flux_lora_entry

    inference_flux_lora_entry()


def inference_wan2_2_t2v_720p():
    from videotuna.cli.inference_app import inference_wan2_2_t2v_720p_entry

    inference_wan2_2_t2v_720p_entry()


def train_flux_lora():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config_path = FLUX_T2I_CONFIG
    data_config_path = FLUX_T2I_DATA_CONFIG
    result = subprocess.run(
        [
            "accelerate",
            "launch",
            "--mixed_precision=bf16",
            "--num_processes=1",
            "--num_machines=1",
            "scripts/train_flux_lora.py",
            "--config_path",
            config_path,
            "--data_config_path",
            data_config_path,
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_wan2_1_t2v_lora():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    ckpt = "checkpoints/wan/Wan2.1-T2V-14B"
    config = WAN_T2V_LORA_CONFIG
    resroot = "results/train"
    expname = "train_wan_domain_t2v_lora"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_new.py",
            "-t",
            "--ckpt",
            ckpt,
            "--base",
            config,
            "--logdir",
            resroot,
            "--name",
            f"{expname}_{current_time}",
            "--devices",
            "0,",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_domain_t2i():
    """Canonical alias for Flux T2I domain LoRA training."""
    train_flux_lora()


def train_domain_t2v():
    """Canonical alias for Wan 2.1 T2V domain LoRA training."""
    train_wan2_1_t2v_lora()


def train_wan2_1_i2v_lora():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    ckpt = "checkpoints/wan/Wan2.1-I2V-14B-480P"
    config = WAN_I2V_LORA_CONFIG
    resroot = "results/train"
    expname = "train_wan_domain_i2v_lora"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_new.py",
            "-t",
            "--ckpt",
            ckpt,
            "--base",
            config,
            "--logdir",
            resroot,
            "--name",
            f"{expname}_{current_time}",
            "--devices",
            "0,",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_domain_i2v():
    """Canonical alias for Wan 2.1 I2V domain LoRA training."""
    train_wan2_1_i2v_lora()


def inference_domain_t2i():
    """Canonical alias for Flux domain LoRA smoke inference."""
    inference_flux_lora()


def validate_domain_t2v():
    """Canonical Wan 2.2 domain LoRA validation after training."""
    from videotuna.cli.inference_app import validate_domain_t2v_entry

    validate_domain_t2v_entry()


def validate_domain_i2v():
    """Canonical Wan 2.2 domain I2V LoRA validation after training."""
    from videotuna.cli.inference_app import validate_domain_i2v_entry

    validate_domain_i2v_entry()


def inference_wan2_2_i2v_720p():
    from videotuna.cli.inference_app import inference_wan2_2_i2v_720p_entry

    inference_wan2_2_i2v_720p_entry()


def benchmark_attn_backends():
    """Benchmark eager vs sdpa vs flash on Wan Diffusers inference."""
    from scripts.benchmark_attn_backends import main

    raise SystemExit(main())
