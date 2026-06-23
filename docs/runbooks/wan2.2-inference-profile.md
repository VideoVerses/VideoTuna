# Wan 2.2-T2V-720p inference profile

Optimized inference presets for **Wan-AI/Wan2.2-T2V-A14B-Diffusers** (Diffusers path via `inference-wan2.2-t2v-720p`). Device and attention routing go through `videotuna/utils/device_utils.py` and `videotuna/utils/attention.py`.

## Hardware tiers

| Environment | Typical hardware | Wan 2.2 720p feasible? |
|-------------|------------------|------------------------|
| Home dev | RX 550 / CPU only | **No** for production 720p. RX 550 is not a supported ROCm target and has far too little VRAM (~2–4 GB). Use CPU smoke for pipeline validation only. |
| Rental | 24 GB (RTX 4090, A10) | Yes — `balanced` or `low_vram` |
| Rental | 40–48 GB (A6000, L40S) | Yes — `max_speed` |
| Rental | 2× A100 | Yes — `max_speed` + `--device-map auto` (Diffusers) or native xfuser USP |

## Preset YAMLs

| Preset file | Tier | Est. peak VRAM |
|-------------|------|----------------|
| [`configs/inference/presets/low_vram_wan2_2_720p.yaml`](../../configs/inference/presets/low_vram_wan2_2_720p.yaml) | Minimum | 12–16 GB |
| [`configs/inference/presets/low_vram_wan2_2_720p_int8.yaml`](../../configs/inference/presets/low_vram_wan2_2_720p_int8.yaml) | Minimum + int8 quant | 10–14 GB (CUDA) |
| [`configs/inference/presets/low_vram_wan2_2_720p_fp8.yaml`](../../configs/inference/presets/low_vram_wan2_2_720p_fp8.yaml) | Minimum + fp8 quant (Ada/Hopper) | 10–14 GB (CUDA sm ≥ 8.9) |
| [`configs/inference/presets/balanced_wan2_2_720p.yaml`](../../configs/inference/presets/balanced_wan2_2_720p.yaml) | Recommended | ~24 GB |
| [`configs/inference/presets/max_speed_wan2_2_720p.yaml`](../../configs/inference/presets/max_speed_wan2_2_720p.yaml) | Max speed | 40–48 GB |
| [`configs/inference/presets/wan2_2_cpu_smoke.yaml`](../../configs/inference/presets/wan2_2_cpu_smoke.yaml) | Home dev only | RAM (not practical) |

## Quantization by hardware tier

Requires **torchao ≥ 0.15.0** (default Poetry dependency) and NVIDIA CUDA. See [Diffusers torchao quantization](https://huggingface.co/docs/diffusers/main/en/quantization/torchao).

| Tier / GPU examples | `int8_wo` | `fp8_wo` | Preset / CLI |
|---------------------|-----------|----------|--------------|
| Home CPU / ROCm | Not supported | Not supported | `transformer_quant: none` |
| 12–16 GB (A10, RTX 3090 — sm 8.6) | **Recommended** | Not supported (sm < 8.9) | [`low_vram_wan2_2_720p_int8.yaml`](../../configs/inference/presets/low_vram_wan2_2_720p_int8.yaml) or `--transformer-quant int8_wo` |
| 24 GB RTX 4090 (sm 8.9) | Supported | **Preferred** (speed + VRAM) | [`low_vram_wan2_2_720p_fp8.yaml`](../../configs/inference/presets/low_vram_wan2_2_720p_fp8.yaml) or `--transformer-quant fp8_wo` |
| 40–48 GB A6000 (sm 8.6) | Supported | Not supported | int8 preset or no quant + `max_speed` |
| 40–48 GB L40S / Hopper (sm ≥ 8.9) | Supported | **Preferred** | fp8 preset or CLI |
| 2× A100 (sm 8.0) | Supported if VRAM-tight | Not supported | int8 or offload without quant |

Measure peak VRAM on rental hardware with `tools/spike_wan_quant_compare.py` (records `torchao` version and per-scheme metrics).

## Three-tier command matrix (rental GPU)

### Minimum VRAM (~12–16 GB)

```bash
export VIDEOTUNA_ATTN_BACKEND=auto   # flash→sdpa on NVIDIA; sdpa on ROCm
poetry run inference-wan2.2-t2v-720p \
  --config configs/inference/presets/low_vram_wan2_2_720p.yaml \
  --min-vram-gb 10
```

Settings: sequential CPU offload, fp16, VAE tiling.

Optional **transformer weight-only quantization** (CUDA only, torchao):

```bash
poetry run inference-wan2.2-t2v-720p \
  --config configs/inference/presets/low_vram_wan2_2_720p_int8.yaml \
  --min-vram-gb 10
```

Ada/Hopper (sm ≥ 8.9) — fp8 weight-only:

```bash
poetry run inference-wan2.2-t2v-720p \
  --config configs/inference/presets/low_vram_wan2_2_720p_fp8.yaml \
  --min-vram-gb 10
```

Or add to any preset / CLI:

```bash
--transformer-quant int8_wo --quant-backend torchao
```

| Scheme | VRAM impact | GPU requirement | LoRA |
|--------|-------------|-----------------|------|
| `int8_wo` (default quant) | Lower transformer weight memory | NVIDIA CUDA | Attempted; use `none` if PEFT bridge fails |
| `int4_wo` | Further weight savings | NVIDIA CUDA | Same as int8 |
| `fp8_wo` | Best speed/memory on Ada+ | sm ≥ 8.9 (RTX 4090, Hopper) | Same as int8 |

**FP8 on Wan 2.2 Diffusers:** use `--transformer-quant fp8_wo` (torchao weight-only; Ada/Hopper sm ≥ 8.9). Legacy native checkpoint FP8 is not supported in PrivTune.

**optimum-quanto:** evaluated via `tools/spike_wan_quant_compare.py` on rental GPU (`--include-quanto`); not added as a default dependency. Use `--quant-backend quanto` only after installing `optimum-quanto>=0.2.6` manually if torchao is insufficient.

When `transformer_quant` is enabled, sequential CPU offload is upgraded to **model CPU offload** automatically for Diffusers quant compatibility.

### Recommended (~24 GB)

```bash
export VIDEOTUNA_ATTN_BACKEND=auto
poetry run inference-wan2.2-t2v-720p \
  --config configs/inference/presets/balanced_wan2_2_720p.yaml \
  --min-vram-gb 20
```

Settings: model CPU offload, bf16, VAE tiling.

### Max speed (~40–48 GB+)

```bash
poetry run install-flash-attn   # NVIDIA only, optional
export VIDEOTUNA_ATTN_BACKEND=flash
export VIDEOTUNA_TORCH_COMPILE=0
poetry run inference-wan2.2-t2v-720p \
  --config configs/inference/presets/max_speed_wan2_2_720p.yaml \
  --min-vram-gb 38
# Optional after a warm-up run (discard first compile iteration when timing):
# poetry run inference-wan2.2-t2v-720p ... --compile
```

Settings: full GPU, bf16, no offload. `--compile` sets `VIDEOTUNA_TORCH_COMPILE=1` and compiles the transformer when offload is disabled.

### Home — CPU smoke (not production)

```bash
poetry install -E cpu --with dev
poetry run install-cpu-torch
export VIDEOTUNA_ATTN_BACKEND=eager
poetry run inference-wan2.2-t2v-720p \
  --config configs/inference/presets/wan2_2_cpu_smoke.yaml \
  --cpu-smoke
```

Also validate configs without weights:

```bash
poetry run test tests/test_wan_inference_presets.py -q
poetry run test tests/test_import_smoke.py -q
```

## VRAM / speed / quality tradeoffs

| Tier | Est. peak VRAM | Speed | Quality tradeoffs |
|------|----------------|-------|-------------------|
| `low_vram` | 12–16 GB | Slowest (sequential PCIe offload) | fp16 vs bf16 — minor; full 720p / 81 frames |
| `balanced` | ~24 GB | Moderate | bf16; model offload latency between steps |
| `max_speed` | 40–48 GB | Fastest single-GPU | Full bf16 on GPU; optional compile after warm-up |
| 2× GPU `device-map auto` | ~22 GB/GPU | Moderate–fast | Same quality as max_speed |
| CPU smoke | RAM only | Impractical | 256p / 2 frames — pipeline validation only |

Quantitative throughput: check `metrics.json` beside outputs after a rental run. Use the benchmark script (below) for frames/sec at 480p.

## Attention backend

| Backend | NVIDIA | ROCm | CPU |
|---------|--------|------|-----|
| `auto` | flash → sdpa → eager | sdpa → eager | eager |
| `flash` | Yes (after `install-flash-attn`) | **Not supported** — use `sdpa` | No |
| `sdpa` | Yes | **Recommended** | Yes |
| `eager` | Yes | Yes | **Required** for `--cpu-smoke` |

```bash
export VIDEOTUNA_ATTN_BACKEND=sdpa   # ROCm rental
export VIDEOTUNA_ATTN_BACKEND=flash    # NVIDIA max_speed
```

## Benchmark methodology

The benchmark script runs a **warm-up** at `num_inference_steps=1` before resetting peak VRAM and starting the timer. The first `torch.compile` iteration is therefore excluded from timed results.

### NVIDIA rental

```bash
poetry run install-flash-attn   # optional
export VIDEOTUNA_ATTN_BACKEND=auto
poetry run benchmark-attn-backends \
  --pipeline wan \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --resolutions 480 \
  --num-inference-steps 4 \
  --json-out results/bench_wan22_attn.json
```

### ROCm rental

```bash
export VIDEOTUNA_ATTN_BACKEND=sdpa
poetry run benchmark-attn-backends \
  --pipeline wan \
  --backends eager sdpa \
  --resolutions 480
```

### 24 GB realistic offload benchmark

```bash
poetry run benchmark-attn-backends \
  --pipeline wan \
  --resolutions 480 \
  --enable-offload
```

**Interpretation:** on NVIDIA, expect `flash` ≈ `sdpa` > `eager`. On ROCm, use `sdpa`. When using `--compile` in production, run twice and discard the first timed iteration.

## Domain LoRA validation (Wan 2.1 → 2.2 bridge)

After Phase 2 training, validate the native Lightning LoRA on Wan 2.2 Diffusers:

```bash
poetry run validate-domain-t2v \
  --trained_ckpt results/train/train_wan_domain_t2v_lora_<ts>/checkpoints/only_trained_model/denoiser-000-000000025.ckpt \
  --prompt_file inputs/t2v/domain_prompt.txt \
  --enable_model_cpu_offload
```

Low VRAM (12–16 GB):

```bash
poetry run validate-domain-t2v \
  --config configs/inference/presets/wan_domain_lora_smoke_22_low_vram.yaml \
  --trained_ckpt <denoiser.ckpt>
```

The bridge loads adapters onto both `transformer` (high-noise) and `transformer_2` (low-noise). Run `poetry run test tests/test_wan_lora_bridge.py -q` on CPU; full visual QA requires a rental GPU.

## Domain I2V LoRA validation (Wan 2.1 → 2.2 I2V bridge)

After optional Phase 2.5 I2V training:

```bash
poetry run validate-domain-i2v \
  --trained_ckpt results/train/train_wan_domain_i2v_lora_<ts>/checkpoints/only_trained_model/denoiser-000-000000025.ckpt \
  --prompt_dir inputs/i2v/domain_smoke \
  --enable_model_cpu_offload
```

Preset: `configs/inference/presets/wan_domain_i2v_smoke_22.yaml` (720×1280, 4 steps, ~24 GB with offload).

`--prompt_dir` must contain paired `.txt` prompts and reference images (same contract as native Wan I2V inference).

Export LoRA for reuse: `poetry run python tools/convert_wan_lora_21_to_22.py --input <ckpt> --output-dir results/lora/wan22-i2v-export/ --mode i2v`

## Multi-GPU (2× A100)

Wan 2.2 via `inference-wan2.2-t2v-720p` uses **Diffusers** (`DiffusersVideoFlow`).

| Path | Command | Pros | Cons |
|------|---------|------|------|
| **device-map auto** (recommended) | `CUDA_VISIBLE_DEVICES=0,1 poetry run inference-wan2.2-t2v-720p --config configs/inference/presets/max_speed_wan2_2_720p.yaml --device-map auto` | Single process; spreads transformer across GPUs | Slower than xfuser USP; experimental |
| **xfuser USP** (native) | `torchrun --nproc_per_node=2 scripts/inference_new.py --config configs/inference/presets/wan2_2_native_t2v_14b.yaml --ulysses_degree 2 --ring_degree 1` | Faster sequence-parallel attention | CUDA-only; no CPU offload; needs `checkpoints/wan/` layout |

See [multi-gpu.md](../multi-gpu.md) for xfuser requirements (`ulysses_degree × ring_degree == WORLD_SIZE`).

## Clear errors when VRAM is insufficient

- **`min_vram_gb` in preset YAML** or **`--min-vram-gb` CLI** — `require_min_vram()` fails before model load with next-step hints (`low_vram`, lower resolution, pick another GPU).
- **720p without GPU** — `require_accelerator_for_flow()` returns `tier=gpu_required` for Wan Diffusers at 720×1280 unless `--cpu-smoke` is set.

## Environment variables (summary)

| Variable | Rental NVIDIA | Rental ROCm | Home CPU |
|----------|---------------|-------------|----------|
| `VIDEOTUNA_ATTN_BACKEND` | `auto` or `flash` | `sdpa` | `eager` |
| `VIDEOTUNA_TORCH_COMPILE` | `0` (or `1` / `--compile` after warm-up) | `0` | `0` |
| `VIDEOTUNA_COMPUTE_BACKEND` | `auto` | `rocm` | `cpu` |
| `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` | GPU selection | GPU selection | N/A |

## Related docs

- [README.md](../../README.md) — performance tuning section
- [install-cpu.md](../install-cpu.md) — CPU smoke tiers
- [install-rocm.md](../install-rocm.md) — AMD ROCm setup
- [multi-gpu.md](../multi-gpu.md) — device-map and xfuser
- [domain-adult-finetune.md](domain-adult-finetune.md) — Wan 2.1 LoRA training (separate from 2.2 Diffusers inference)
