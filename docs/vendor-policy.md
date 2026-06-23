# Vendor policy

PrivTune vendors upstream model code when Diffusers/Hugging Face pipelines are insufficient for training. This document defines **where** vendored code lives, **how** to update it, and **what** attribution is required.

## Directory convention

| Location | Purpose | Example |
|----------|---------|---------|
| `videotuna/models/<family>/` | Native model implementations for training and/or smoke inference | `wan/` |
| `videotuna/training/<task>/` | First-party training loops (Diffusers + PEFT + Accelerate) | `flux_lora/` |
| `videotuna/vendor/<upstream>/` | Third-party snapshots (git submodule preferred) | *(none today)* |

**Rule:** New upstream code goes under `videotuna/vendor/<upstream>/` with a `VENDOR.md` at the tree root. Prefer first-party trainers under `videotuna/training/` when Diffusers covers the model.

## Required provenance (`VENDOR.md`)

Every vendored tree must include `VENDOR.md` (or `LICENSE` + `VENDOR.md`) with:

1. **Upstream repository URL**
2. **License** (SPDX identifier + link to upstream `LICENSE`)
3. **Pinned commit** (full SHA) at last sync
4. **Import date / PR** that introduced or last updated the snapshot
5. **PrivTune entrypoints** that depend on the tree
6. **Update procedure** (submodule bump, manual diff, or replacement plan)

## Update process

1. Identify upstream release or commit to pin.
2. Record the SHA in `VENDOR.md` before merging.
3. Run import smoke tests (`poetry run test tests/test_import_smoke.py -q`).
4. Run the relevant Poetry script on a smoke config.
5. Note breaking config changes in `README.md` and `docs/MODEL_VERSIONS.md` if applicable.

Prefer **git submodule** or **pip/git dependency** over copying large trees.

## Dependency groups

| Group | Install command | Consumers |
|-------|-----------------|-----------|
| *(default / main)* | `poetry install -E cuda` or `uv sync` | Diffusers inference (Flux + Wan 2.2), Wan native smoke |
| `training` | `--with training` | Flux LoRA, Wan 2.1 Lightning LoRA, DeepSpeed |
| `dev` | `--with dev` | pytest, ruff, mypy |

## Poetry scripts → dependency groups

| Scripts | Groups required |
|---------|-----------------|
| `inference-flux-lora`, `inference-wan2.2-t2v-720p` | default (inference) |
| `train-flux-lora`, `train-wan2-1-t2v-lora` | `training` |
| `install-deepspeed`, `install-flash-attn` | `training` / optional runtime |
| `test`, `lint`, `format*` | `dev` |

## Inventory

| Path | Upstream | License | Entrypoints | Vendor deps | Status |
|------|----------|---------|-------------|-------------|--------|
| `videotuna/training/flux_lora/` | PrivTune first-party | N/A | `train-flux-lora` | — | **Keep** — Accelerate stack ([ADR-001](decisions/0001-dual-training-stacks.md)) |
| `videotuna/models/wan/` | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) (Wan 2.1 hub weights for domain training — see `VENDOR.md`) | Apache-2.0 | `train-domain-t2v`, `train-domain-i2v`, `train-wan2-1-*-lora`, `scripts/train_new.py`, `scripts/inference_new.py` | `easydict` (configs only) | **Keep** — Lightning+DeepSpeed stack ([ADR-001](decisions/0001-dual-training-stacks.md)); T2V/I2V only; s2v/animate/ti2v pruned (see `VENDOR.md`) |

Provenance details: [`videotuna/models/wan/VENDOR.md`](../videotuna/models/wan/VENDOR.md).

`easydict` is pinned in `pyproject.toml` for upstream Wan config modules under `videotuna/models/wan/wan/configs/`. First-party code must not import it; `tests/test_vendor_import_boundary.py` enforces that boundary.

## Flux LoRA training

First-party trainer at `videotuna/training/flux_lora/` (Diffusers + PEFT + Accelerate). Configs under `configs/domain/`. Inference via `DiffusersVideoFlow` / `inference-flux-lora`. Training stack rationale: [ADR-001](decisions/0001-dual-training-stacks.md).

## Removing vendored code

Before deleting any file:

1. Confirm zero references outside the vendor tree.
2. Confirm no Poetry script imports it.
3. Archive provenance in `docs/vendor/` and update this inventory.
