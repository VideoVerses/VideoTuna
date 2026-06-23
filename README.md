# PrivTune

**PrivTune** is a private-domain LoRA training platform for still-image and short-video generation.

Canonical runbook: [`docs/runbooks/domain-adult-finetune.md`](docs/runbooks/domain-adult-finetune.md)

The Python package directory remains `videotuna/` for compatibility; Poetry project name is `privtune`.

## What PrivTune does

| Phase | Model | Role |
|-------|-------|------|
| 1 — T2I | FLUX.1-dev LoRA | Train domain still-image style |
| 2 — T2V | Wan 2.1 T2V LoRA | Train domain short-video motion |
| 3 — Validate | Wan 2.2 Diffusers | Production domain LoRA validation via `validate-domain-t2v` (see [domain-adult-finetune.md](docs/runbooks/domain-adult-finetune.md)); general 720p inference profile in [wan2.2-inference-profile.md](docs/runbooks/wan2.2-inference-profile.md) |

QA uses **training ImageLogger callbacks** and **LoRA smoke inference** on held-out prompts — not generic T2V benchmarking (VBench removed).

## Legal and data requirements

- Use only **rights-cleared, consented** training data.
- Never commit `data/`, checkpoints, `results/`, or `outputs/` to git.

## Install (default = training)

PrivTune supports **Poetry** (default) and **[uv](https://docs.astral.sh/uv/)**.

```shell
conda create -n privtune python=3.11 -y
conda activate privtune
pip install poetry
poetry install -E cuda --with training
poetry run install-deepspeed   # required for Wan LoRA (DeepSpeed ZeRO-3)
```

| Use case | Poetry | uv |
|----------|--------|-----|
| **Default (CUDA + training)** | `poetry install -E cuda --with training` | `uv sync --group training` |
| Inference AMD ROCm | `poetry install -E rocm --with training` then `poetry run install-rocm` | Wan training requires CUDA; ROCm is inference + Flux training only — see [install-rocm.md](docs/install-rocm.md) |
| CPU dev / CI | `poetry install -E cpu --with dev --with training` then `poetry run install-cpu-torch` | see [install-cpu.md](docs/install-cpu.md) |
| + Dev (pytest, ruff) | add `--with dev` | `uv sync --group dev` |

### Docker (optional)

For containerized dev (e.g. Mac arm64), use the Compose image **`privtune`** (`privtune:latest` or `privtune:${TAG}`):

```shell
export HOST_UID=$(id -u) HOST_GID=$(id -g)
docker compose build privtune
docker compose run --rm privtune bash
# inside container: poetry install -E cuda --with training
```

The legacy **`videotune`** Compose service and `videotune:latest` image tag remain as a backward-compatible alias (deprecated). A separate legacy all-in-one RunPod image lives under [`docker/Dockerfile`](docker/Dockerfile) and is not the recommended training path.

See [`docs/vendor-policy.md`](docs/vendor-policy.md) for vendored upstream policy.

## Data preparation

**Phase 1 — still images** (`data/t2i/domain/`):

```
data/t2i/domain/
  0001.jpg
  0001.txt          # e.g. "sks_style, portrait, studio lighting"
```

Use a consistent trigger token (default: `sks_style`) in every `.txt` caption file.

**Phase 2 — short video** (`data/t2v/domain/`):

```
data/t2v/domain/
  metadata.csv
  videos/
    clip001.mp4
```

See the runbook for CSV format and ffmpeg re-encode notes.

## Train

```bash
# Phase 1 — Flux T2I domain LoRA
poetry run train-domain-t2i

# Phase 2 — Wan 2.1 T2V domain LoRA
poetry run train-domain-t2v
```

Configs: [`configs/domain/flux_t2i.json`](configs/domain/flux_t2i.json), [`configs/domain/flux_t2i_data.json`](configs/domain/flux_t2i_data.json), [`configs/domain/wan_t2v_lora.yaml`](configs/domain/wan_t2v_lora.yaml).

Legacy aliases `train-flux-lora` and `train-wan2-1-t2v-lora` remain available.

Outputs:

- Phase 1: `results/train/flux-domain-adult/checkpoint-*/`
- Phase 2: `results/train/train_wan_domain_t2v_lora_*/`

## Validate

**Phase 1 (this milestone):**

```bash
poetry run inference-domain-t2i \
  --lorackpt results/train/flux-domain-adult/checkpoint-2000 \
  --prompt "sks_style, portrait, soft lighting"
```

**Phase 2 interim (Wan 2.1 native smoke):**

```bash
poetry run python scripts/inference_new.py \
  --config configs/inference/presets/wan_domain_lora_smoke.yaml \
  --ckpt_path checkpoints/wan/Wan2.1-T2V-14B \
  --trained_ckpt results/train/.../denoiser-000-000000025.ckpt \
  --prompt "sks_style, slow camera push-in"
```

**Phase 2 production (Wan 2.2 domain LoRA validation):**

```bash
poetry run validate-domain-t2v \
  --trained_ckpt results/train/.../denoiser-000-000000025.ckpt \
  --prompt_file inputs/t2v/domain_prompt.txt
```

See [`docs/runbooks/domain-adult-finetune.md`](docs/runbooks/domain-adult-finetune.md) for VRAM presets and known bridge limitations. For general Wan 2.2 720p inference (non-domain), see [`docs/runbooks/wan2.2-inference-profile.md`](docs/runbooks/wan2.2-inference-profile.md).

## VRAM and hardware

| Phase | Model | Peak VRAM | Notes |
|-------|-------|-----------|-------|
| 1 — T2I | Flux LoRA @ 512px | ~24–40 GB | 1 GPU |
| 2 — T2V | Wan 2.1 LoRA @ 480×832×81 | ~38 GB | 1 GPU + DeepSpeed ZeRO-3 offload |

## CPU dev / CI gates

```bash
poetry install -E cpu --with dev --with training
poetry run install-cpu-torch
poetry run lint
poetry run format-check
poetry run test tests/test_import_smoke.py -q
poetry run test tests/test_domain_finetune_configs.py -q
poetry run test tests/test_flux_lora_train_smoke.py -q
poetry run test tests/test_poetry_scripts.py -q
```

## Environment variables

`VIDEOTUNA_*` env vars are retained for compatibility (see [`.env.example`](.env.example)).

| Variable | Purpose |
|----------|---------|
| `VIDEOTUNA_ATTN_BACKEND` | `auto`, `flash`, `sdpa`, `eager` — use `sdpa` on ROCm |
| `VIDEOTUNA_COMPUTE_BACKEND` | `auto`, `cuda`, `rocm`, `cpu` |
| `HF_TOKEN` | Gated models (FLUX.1-dev) |

## Project layout

```
videotuna/
  training/flux_lora/   # Phase 1 trainer
  models/wan/           # Phase 2 native stack
  flow/                 # wanvideo (train), diffusers_video (infer)
configs/domain/         # flux_t2i*.json, wan_t2v_lora.yaml, cloud smoke variants
configs/inference/presets/  # smoke + Wan 2.2 inference presets
cloud/vast/             # rented GPU provisioning
docs/runbooks/          # domain-adult-finetune, wan2.2-inference-profile
```

## Cloud GPU training

Rented GPU provisioning (Vast.ai): [`docs/runbooks/cloud-gpu-training.md`](docs/runbooks/cloud-gpu-training.md)

## Related docs

| Doc | Topic |
|-----|-------|
| [domain-adult-finetune.md](docs/runbooks/domain-adult-finetune.md) | Full domain training runbook |
| [checkpoints.md](docs/checkpoints.md) | Weight download layout |
| [MODEL_VERSIONS.md](docs/MODEL_VERSIONS.md) | FLUX.1 + Wan 2.1/2.2 pins |
| [cloud-gpu-training.md](docs/runbooks/cloud-gpu-training.md) | Vast.ai provisioning |
| [vendor-policy.md](docs/vendor-policy.md) | Vendored upstream policy |

## License

See [LICENSE](./LICENSE).
