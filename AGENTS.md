# PrivTune — Agent Instructions

## Project overview

**PrivTune** (`privtune` in Poetry; Python import path `videotuna/`) is a **training platform ONLY** for private-domain LoRA:

- **Phase 1:** Flux T2I LoRA (`videotuna/training/flux_lora/`, `train-domain-t2i`)
- **Phase 2:** Wan 2.1 T2V LoRA (`videotuna/flow/wanvideo.py`, `train-domain-t2v`)
- **Phase 2.5 (optional):** Wan 2.1 I2V LoRA (`train-domain-i2v`) — LoRA-only; full Wan fine-tune is out of scope
- **Phase 3:** Wan 2.2 Diffusers domain LoRA validation via `validate-domain-t2v` (production-ready; see runbook). General Wan 2.2 720p inference profile remains optional — [`wan2.2-inference-profile.md`](docs/runbooks/wan2.2-inference-profile.md).

Training stacks differ by design — see [ADR-001](docs/decisions/0001-dual-training-stacks.md).

Canonical runbook: [`docs/runbooks/domain-adult-finetune.md`](docs/runbooks/domain-adult-finetune.md)

Python 3.11+; Poetry default (`poetry run …`); optional uv.

## Role

Optimize for:

1. **Correct behavior** — training and smoke inference on CUDA, ROCm, and CPU config validation.
2. **Scoped diffs** — change only what the task requires.
3. **Portable device handling** — respect `videotuna/utils/device_utils.py` and `.env.example`.
4. **Safe boundaries** — never commit weights, datasets, `outputs/`, or secrets.

Cursor rules: [`.cursor/rules/privtune.mdc`](.cursor/rules/privtune.mdc)

## Agent workflow

1. `cd` into the repo root before running commands.
2. Prefer **Poetry** (`poetry run …`) unless the user explicitly uses uv.
3. Read [`docs/vendor-policy.md`](docs/vendor-policy.md) before touching vendored code.

## Install profiles

| Use case | Poetry | uv |
|----------|--------|-----|
| **Default (CUDA + training)** | `poetry install -E cuda --with training` | `uv sync --group training` |
| Inference AMD ROCm | `poetry install -E rocm --with training` then `poetry run install-rocm` | Wan training requires CUDA; ROCm is inference + Flux training only — see [install-rocm.md](docs/install-rocm.md) |
| CPU dev / CI | `poetry install -E cpu --with dev --with training` then `poetry run install-cpu-torch` | see [install-cpu.md](docs/install-cpu.md) |
| + Dev | add `--with dev` | `uv sync --group dev` |

After install for Wan LoRA: `poetry run install-deepspeed`

## Verification (required before finishing)

```bash
poetry run lint
poetry run format-check
poetry run coverage-gate
```

`coverage-gate` runs the CI smoke test list and enforces a **33%** line-coverage floor on `videotuna/training/` + `videotuna/utils/`. For local exploratory reporting without a gate, use `poetry run coverage-report`.

| Change area | Additional tests |
|-------------|------------------|
| Wan 2.2 presets / bridge | `test_wan_inference_presets.py` |
| diffusers_video | `test_diffusers_video_flow.py` |
| device/attention | `test_device_utils.py`, `test_attention_backend.py` |
| inference CLI / memory | `test_inference_optimization.py` |

## Commands

### Training (canonical)

```bash
poetry run train-domain-t2i
poetry run train-domain-t2v
poetry run install-deepspeed   # Wan LoRA
```

Configs: `configs/domain/flux_t2i.json`, `configs/domain/flux_t2i_data.json`, `configs/domain/wan_t2v_lora.yaml`

Legacy aliases: `train-flux-lora`, `train-wan2-1-t2v-lora`

### Smoke inference

```bash
poetry run inference-domain-t2i --lorackpt results/train/flux-domain-adult/checkpoint-2000
poetry run validate-domain-t2v --trained_ckpt results/train/.../denoiser.ckpt
poetry run inference-wan2.2-t2v-720p   # general Wan 2.2 720p — optional
```

### Dev tooling

```bash
poetry run test -q
poetry run lint
poetry run format-check
poetry run type-check   # mypy on typed allowlist only (see pyproject.toml)
```

## Environment variables

`VIDEOTUNA_*` prefix is retained for compatibility (no `PRIVTUNE_*` aliases in v0.2).

| Variable | Default | Purpose |
|----------|---------|---------|
| `VIDEOTUNA_COMPUTE_BACKEND` | `auto` | cuda / rocm / cpu override |
| `VIDEOTUNA_ATTN_BACKEND` | `auto` | flash / sdpa / eager |
| `VIDEOTUNA_TORCH_COMPILE` | `0` | denoiser compile |
| `HF_TOKEN` | — | Gated HF models |

## Project layout

```
videotuna/
  training/flux_lora/   # Phase 1
  models/wan/           # Phase 2
  flow/                 # wanvideo, diffusers_video
configs/domain/         # flux_t2i*.json, wan_t2v_lora.yaml
configs/inference/presets/  # smoke + Wan 2.2 presets
cloud/vast/
docs/runbooks/
```

## Safety

- Never commit `.env`, checkpoints, `outputs/`, `results/`, or training data.
- ROCm: `VIDEOTUNA_ATTN_BACKEND=sdpa`; do not run `install-flash-attn`.
- **QA = training callbacks + smoke inference.** VBench was removed: domain QA does not need generic T2V benchmarking; ImageLogger previews and LoRA smoke inference cover the supported workflows.

## Related docs

| Doc | Topic |
|-----|-------|
| [0001-dual-training-stacks.md](docs/decisions/0001-dual-training-stacks.md) | Why Flux uses Accelerate and Wan uses Lightning+DeepSpeed |
| [domain-adult-finetune.md](docs/runbooks/domain-adult-finetune.md) | Domain training runbook |
| [wan2.2-inference-profile.md](docs/runbooks/wan2.2-inference-profile.md) | Wan 2.2 rental GPU presets (Phase 3) |
| [checkpoints.md](docs/checkpoints.md) | Weight layout |

## Cursor Cloud specific instructions

The Cloud VM is **CPU-only (no GPU/CUDA driver)** and runs Python 3.12 (satisfies `^3.11`). It uses the documented "CPU dev / CI" profile (see [docs/install-cpu.md](docs/install-cpu.md)). The startup update script installs deps and swaps to CPU torch; the notes below are durable caveats, not setup steps.

- **CPU-torch swap is mandatory after every `poetry install`.** The lockfile pins CUDA `torch==2.6.0+cu126`, so any `poetry install` re-installs the CUDA wheel; you must re-run `poetry run install-cpu-torch` afterward or imports break with CUDA errors. Verify with `poetry run verify-cpu-torch`.
- **Use `VIDEOTUNA_ATTN_BACKEND=eager` on CPU** (flash/xformers/bitsandbytes are CUDA-only and absent here). Keep `VIDEOTUNA_TORCH_COMPILE=0`.
- **The `poetry run test <path>` script appends args to `pytest tests`**, so it always collects the whole suite regardless of the path you pass. To run a single file, call `poetry run pytest <path>` directly.
- **Full suite + the `test_import_smoke.py` gate need the `training` group** (`pytorch_lightning`, `pandas`); without it ~5 modules fail to import. Install with `--with dev --with training`.
- **Known pre-existing baseline failures (not environment issues):** `poetry run lint` reports ~1000 ruff errors; `poetry run pytest tests` shows ~6 failures (`tests/datasets/test_dataset_from_csv.py` hits a `PosixPath` bug in `videotuna/data/datasets.py`, and `test_wan_checkpoint.py::test_wan_from_pretrained_missing_dir` depends on diffusers/network behavior). The rest (~129) pass.
- **GPU training/inference and real FLUX/Wan weights are not runnable here.** `inference-wan2.2-t2v-720p` and the CPU smoke preset download a 14B Wan 2.2 model. Validate core behavior via the CPU test gates in [capability-matrix.md](docs/capability-matrix.md) and small LoRA training-step smokes instead.
=======
| [MODEL_VERSIONS.md](docs/MODEL_VERSIONS.md) | Model pins |

Removed: `capability-matrix.md`, `finetune_flux.md`, `finetune_wan.md`, `evaluation.md` (superseded by runbook + smoke inference).
