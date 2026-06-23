# CPU-only development install

PrivTune supports CPU-only installs for **unit tests**, **config validation**, and **tiny smoke inference**. CPU is not practical for domain LoRA training or 14B video generation — use NVIDIA CUDA or AMD ROCm for production training.

## Prerequisites

- Linux, macOS, or Windows
- Python 3.11+
- No GPU required

## Install

```bash
poetry install -E cpu --with dev --with training
poetry run install-cpu-torch
```

`install-cpu-torch` removes CUDA/ROCm wheels and installs CPU-only `torch==2.6.0` + `torchvision==0.21.0` from `https://download.pytorch.org/whl/cpu`.

**Important:** The committed `poetry.lock` pins NVIDIA CUDA torch. Any later plain `poetry install` may restore `+cu126` wheels — re-run `poetry run install-cpu-torch` on CPU-only machines.

Verify:

```bash
poetry run verify-cpu-torch
poetry run python -c "from videotuna.utils.device_utils import describe_compute_environment; print(describe_compute_environment())"
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `VIDEOTUNA_COMPUTE_BACKEND` | Set `cpu` to force CPU even when a GPU is visible |
| `VIDEOTUNA_CPU_MODE` | `off` (default), `smoke` (tiny runs), `force` (debug init; deprecated alias: `VIDEOTUNA_ALLOW_CPU_INFERENCE=1`) |
| `VIDEOTUNA_ATTN_BACKEND` | Use `eager` or `sdpa` on CPU (`flash` is not supported) |
| `VIDEOTUNA_TORCH_COMPILE` | Keep `0` on CPU (compile is GPU-only) |

## CPU inference vs GPU + CPU offload

| | CPU-only inference | GPU inference + CPU offload |
|--|-------------------|----------------------------|
| **Purpose** | Dev, CI, smoke tests | Reduce VRAM on a GPU machine |
| **Requires GPU** | No | Yes |
| **Flags** | `--cpu-smoke`, `device: cpu` | `--enable_model_cpu_offload`, `--memory-preset low_vram` |
| **Practical for 720p 14B** | No | Yes (slow) |

CPU offload flags move weights between GPU and **host RAM**. They do not replace a GPU for large models.

## Smoke tests

```bash
export VIDEOTUNA_ATTN_BACKEND=eager
poetry run lint
poetry run format-check
poetry run test tests/test_import_smoke.py -q
poetry run test tests/test_domain_finetune_configs.py -q
poetry run test tests/test_flux_lora_train_smoke.py -q
poetry run test tests/test_poetry_scripts.py -q
```

Optional CPU inference smoke (downloads Wan 2.2 weights on first run):

```bash
poetry run inference-wan2.2-t2v-720p \
  --config configs/inference/presets/wan2_2_cpu_smoke.yaml \
  --cpu-smoke --num_inference_steps 2
```

Preset: [`configs/inference/presets/wan2_2_cpu_smoke.yaml`](../configs/inference/presets/wan2_2_cpu_smoke.yaml). See [MODEL_VERSIONS.md](MODEL_VERSIONS.md).

## Model tiers on CPU

| Tier | Models | Status |
|------|--------|--------|
| **cpu_ok** | Import smoke, config parse, unit tests | Always |
| **cpu_smoke** | Flux dev (≤512px), Wan 2.2 tiny preset | `--cpu-smoke` required |
| **gpu_required** | Production 720p Wan 2.2, native Wan 2.1 720p | Clear error; use `wan2_2_cpu_smoke.yaml` or config validation only |

## NVIDIA install (default)

```bash
poetry install -E cuda --with training
poetry run install-deepspeed
poetry run install-flash-attn   # optional
```

## AMD ROCm

See [install-rocm.md](install-rocm.md).

## Apple Silicon

Linux CPU install is separate from Apple Silicon. For Mac arm64, use the Compose **`privtune`** image for containerized dev — see [Docker (optional)](../README.md#docker-optional) in the README. Native CPU training on Mac remains limited (same constraints as Linux CPU smoke).

## Troubleshooting

**`verify-cpu-torch` reports CUDA build**

Re-run `poetry run install-cpu-torch` after any `poetry install`.

**`flash` / `xformers` import errors on CPU**

Expected — these are CUDA-only optional deps. Use `VIDEOTUNA_ATTN_BACKEND=eager`.

**Offload flags rejected on CPU**

`--enable_model_cpu_offload` and `--memory-preset low_vram` need a GPU. Remove them for CPU smoke runs.

**Wan blocked at production resolution**

Production 720p configs are `gpu_required` on CPU. Use `wan2_2_cpu_smoke.yaml` with `--cpu-smoke`, or stick to config/unit tests.
