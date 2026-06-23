# AMD ROCm install

PrivTune supports AMD GPUs on Linux x86_64 via PyTorch ROCm wheels. ROCm uses the same `torch.cuda` API as NVIDIA CUDA (HIP backend).

## Prerequisites

- Linux x86_64
- AMD GPU (e.g. RX 7900 XTX, MI300 series)
- ROCm driver **≥ 6.2** (PyTorch 2.6 wheels target **ROCm 6.2.4**)
- Python 3.11+

Verify the driver:

```bash
rocminfo | head
```

## Install

```bash
poetry install -E rocm
poetry run install-rocm
```

`install-rocm` removes CUDA-only packages, uninstalls any existing torch/torchvision wheels, then installs matching **ROCm** builds of `torch==2.6.0` and `torchvision==0.21.0` from `https://download.pytorch.org/whl/rocm6.2.4`.

**Important:** The committed `poetry.lock` pins NVIDIA CUDA torch. Any later `poetry install` may restore `+cu126` wheels — re-run `poetry run install-rocm` on AMD machines before inference.

Verify:

```bash
poetry run python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
poetry run python -c "from videotuna.utils.device_utils import describe_compute_environment; print(describe_compute_environment())"
```

## Training vs inference scope

| Workflow | ROCm support | Notes |
|----------|--------------|-------|
| Flux T2I LoRA training | Supported | Accelerate only; no DeepSpeed |
| Wan 2.1 T2V/I2V LoRA training | **Not supported** | Requires DeepSpeed ZeRO-3; `poetry run install-deepspeed` aborts on ROCm |
| Wan 2.2 / Flux inference & validation | Supported | Use `VIDEOTUNA_ATTN_BACKEND=sdpa` + CPU offload presets |

Wan training requires NVIDIA CUDA:

```bash
poetry install -E cuda --with training
poetry run install-deepspeed
```

For Flux training on ROCm, add `--with training` to the install command above (no `install-deepspeed`).

## Environment variables

| Variable | Purpose |
|----------|---------|
| `VIDEOTUNA_COMPUTE_BACKEND` | `auto`, `cuda`, `rocm`, or `cpu` |
| `VIDEOTUNA_ATTN_BACKEND` | Use `sdpa` or `eager` on ROCm (`flash` is not supported) |
| `HIP_VISIBLE_DEVICES` | GPU selection (like `CUDA_VISIBLE_DEVICES`) |

## Smoke tests

```bash
export VIDEOTUNA_ATTN_BACKEND=sdpa
poetry run benchmark-attn-backends --num-inference-steps 2
poetry run inference-wan2.2-t2v-720p \
  --config configs/inference/presets/wan2_2_cpu_smoke.yaml \
  --num_inference_steps 2 --enable_model_cpu_offload
```

Per-model presets: [MODEL_VERSIONS.md](MODEL_VERSIONS.md).

## Model tiers on ROCm

| Tier | Models | Status |
|------|--------|--------|
| **Supported** | Flux T2I, Wan 2.2 Diffusers | Use `sdpa` + CPU offload |
| **Not supported** | Wan 2.1 native train/infer | CUDA + DeepSpeed required; see [Training vs inference scope](#training-vs-inference-scope) |

## NVIDIA install (default)

```bash
poetry install -E cuda
poetry run install-flash-attn   # optional
```

Training (NVIDIA only): `poetry install -E cuda --with training` then `poetry run install-deepspeed` if needed.

## CPU-only dev

```bash
poetry install -E cpu
poetry run install-cpu-torch
```

See [install-cpu.md](install-cpu.md) for smoke tests and tier matrix.

## Troubleshooting

**`torchvision::nms` / import errors after `install-rocm`**

torch and torchvision must come from the same ROCm index. If torchvision still shows `+cu126`, re-run:

```bash
poetry run install-rocm
```

**`torch.cuda.is_available()` is False**

- Confirm ROCm driver: `rocminfo`
- Re-run `poetry run install-rocm`
- Check `HIP_VISIBLE_DEVICES` is not masking all GPUs

**Out of memory**

- Use `--enable_sequential_cpu_offload`, `--enable_vae_tiling`, `--dtype bf16`
- Prefer Wan 2.2 Diffusers presets with `--memory-preset low_vram`

**flash-attn / xformers errors**

- ROCm does not use these packages. Set `export VIDEOTUNA_ATTN_BACKEND=sdpa`

**`install-deepspeed` exits with "not supported on AMD ROCm"**

Expected. Wan LoRA training needs NVIDIA CUDA + DeepSpeed ZeRO-3. Use ROCm for inference/validation and Flux T2I training only, or train Wan on a CUDA machine / cloud GPU (see [domain-adult-finetune.md](runbooks/domain-adult-finetune.md)).

## Lockfile note

The committed `poetry.lock` targets the `cuda` extra. ROCm users rely on `install-rocm` for PyTorch wheels. To regenerate a ROCm lock locally: `poetry lock` after editing extras (advanced).
