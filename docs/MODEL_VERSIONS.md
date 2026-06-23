# Model versions

PrivTune supports three model families: two for **training** and one for **validation**.

| Model | Hub ID | Role |
|-------|--------|------|
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | **Train** — Phase 1 T2I LoRA |
| Wan 2.1 T2V 14B | `Wan-AI/Wan2.1-T2V-14B` | **Train** — Phase 2 T2V LoRA |
| Wan 2.1 I2V 14B 480P | `Wan-AI/Wan2.1-I2V-14B-480P` | **Train** — Phase 2.5 I2V LoRA (optional) |
| Wan 2.2 T2V A14B Diffusers | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | **Validate** — Phase 2 production inference |
| Wan 2.2 I2V A14B Diffusers | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | **Validate** — Phase 2.5 I2V production inference |

## Training configs

| Phase | Config |
|-------|--------|
| Flux T2I | `configs/domain/flux_t2i.json` + `configs/domain/flux_t2i_data.json` |
| Wan T2V | `configs/domain/wan_t2v_lora.yaml` |
| Wan I2V (optional) | `configs/domain/wan_i2v_lora.yaml` |
| Flux T2I cloud smoke | `configs/domain/flux_t2i_cloud_smoke.json` + `configs/domain/flux_t2i_data.json` |
| Wan T2V cloud smoke | `configs/domain/wan_t2v_lora_cloud_smoke.yaml` |

Commands: `poetry run train-domain-t2i` / `poetry run train-domain-t2v` / `poetry run train-domain-i2v`

Wan training requires DeepSpeed ZeRO-3: `poetry run install-deepspeed`

## Smoke inference presets

| Phase | Preset | Command |
|-------|--------|---------|
| Flux LoRA | `configs/inference/presets/flux_domain_lora_smoke.yaml` | `poetry run inference-domain-t2i` |
| Wan 2.2 domain LoRA | `configs/inference/presets/wan_domain_lora_smoke_22.yaml` | `poetry run validate-domain-t2v --trained_ckpt <ckpt>` |
| Wan 2.2 domain I2V LoRA | `configs/inference/presets/wan_domain_i2v_smoke_22.yaml` | `poetry run validate-domain-i2v --trained_ckpt <ckpt> --prompt_dir <dir>` |
| Wan 2.2 (general) | `configs/inference/presets/balanced_wan2_2_720p.yaml` | `poetry run inference-wan2.2-t2v-720p` |
| Wan 2.1 LoRA (optional) | `configs/inference/presets/wan_domain_lora_smoke.yaml` | `inference_new` + `--trained_ckpt` |
| Wan 2.1 I2V LoRA (optional) | `configs/inference/presets/wan_domain_i2v_smoke.yaml` | `inference_new` + `--prompt_dir` |

LoRA bridge (Wan 2.1 native → 2.2 Diffusers): `videotuna/utils/wan_lora_bridge.py`

Offline export: `tools/convert_wan_lora_21_to_22.py`

## ML stack pins (Wan 2.2 LoRA bridge audit)

Audited at commit `a17b6a0` (2026-06-23). Validated against `tests/test_wan_lora_bridge.py` (remap coverage ≥ 90% on production-style fixture keys; 11 CPU tests + optional GPU smoke).

| Package | Constraint | Locked | Notes |
|---------|------------|--------|-------|
| diffusers | `^0.38.0` | 0.38.0 | Wan `WanTransformer3DModel` + `WanLoraLoaderMixin`; pipeline `set_adapters(adapter_weights=…)` |
| peft | `^0.17.0` | 0.17.1 | Runtime bridge uses `get_peft_model` + `set_peft_model_state_dict`; cap at 0.17.x (see matrix) |
| transformers | `^4.48.0` | 4.57.6 | Shared with Flux training; lock resolves above floor |
| accelerate | `^1.14.0` | 1.14.0 | peft requires `>=0.21.0`; diffusers 0.38 dev extra recommends `>=0.31.0` |
| safetensors | `^0.8.0` | 0.8.0 | Required by diffusers 0.38 (`>=0.8.0rc0`) |

### Upstream API alignment (diffusers 0.38 + peft 0.17)

Wan 2.2 T2V/I2V pipelines expose two denoisers: `transformer` (high-noise) and `transformer_2` (low-noise). Diffusers community practice ([PR #12074](https://github.com/huggingface/diffusers/pull/12074)):

- `pipeline.load_lora_weights(..., adapter_name=…)` for high-noise expert
- `pipeline.load_lora_weights(..., load_into_transformer_2=True)` for low-noise expert
- `pipeline.set_adapters(["a1", "a2"], adapter_weights=[w1, w2])` to activate both

PrivTune bridge (`videotuna/utils/wan_lora_bridge.py`) mirrors this at runtime without pre-exported safetensors:

| Upstream | PrivTune bridge |
|----------|-----------------|
| `load_lora_weights` per expert | `get_peft_model()` + `set_peft_model_state_dict()` on each `WanTransformer3DModel` |
| Named adapters per expert | `domain_high` / `domain_low` |
| `set_adapters` with weights | `pipeline.set_adapters(adapters, adapter_weights=scales)` |

Offline export (`tools/convert_wan_lora_21_to_22.py`) writes `high_noise.safetensors` / `low_noise.safetensors` for native `load_lora_weights`; validated in `test_exported_lora_loads_via_diffusers_adapter`.

Production inference (`videotuna/flow/diffusers_video.py`) detects native Wan 2.1 ckpts and calls the bridge; Diffusers-format LoRA dirs fall through to `pipeline.load_lora_weights()`.

### Version matrix

Procedure: swap only `diffusers` + `peft` in the Poetry env (`poetry run pip install --no-deps diffusers==X peft==Y`), run `poetry run test tests/test_wan_lora_bridge.py -q -k "not gpu"` and `poetry run python tools/spike_wan_lora_bridge.py --synthetic /tmp/synthetic-matrix.ckpt`, restore `diffusers==0.38.0 peft==0.17.1`.

| Row | diffusers | peft | Result |
|-----|-----------|------|--------|
| A (baseline) | 0.36.0 | 0.17.1 | pass — remap 100% |
| B | 0.37.1 | 0.17.1 | pass — remap 100% |
| C | 0.38.0 | 0.17.1 | pass — **chosen combo** |
| F | 0.38.0 | 0.18.1 | pass — remap 100% |
| D | 0.38.0 | 0.19.1 | fail — `peft` 0.19.1 requires `torchao>=0.16.0`; project pins `torchao ^0.9.0` |
| E | 0.36.0 | 0.19.1 | fail — same `torchao` incompatibility in `get_peft_model()` |

**Decision:** keep `diffusers ^0.38.0` + `peft ^0.17.0`. peft 0.18.1 passes bridge tests but offers no benefit over 0.17.1 for this workflow; peft 0.19.x is blocked by the `torchao` pin (used elsewhere for Wan 2.2 quant inference).

Debug inventory: `poetry run python tools/spike_wan_lora_bridge.py --synthetic /tmp/synthetic.ckpt`

## Wan training stack pins (DeepSpeed + PyTorch Lightning audit)

See [ADR-001](decisions/0001-dual-training-stacks.md) for rationale; pins below.

| Phase | Stack | Entry point |
|-------|-------|-------------|
| Flux T2I LoRA | Hugging Face **Accelerate** | `poetry run train-domain-t2i` |
| Wan 2.1 T2V / I2V LoRA | **PyTorch Lightning** + **DeepSpeed ZeRO-3** | `poetry run train-domain-t2v` / `train-domain-i2v` |

Wan domain YAMLs set `train.lightning.strategy: deepspeed_stage_3_offload` with `precision: bf16-mixed`. Training runs through `scripts/train_new.py` (Lightning `Trainer` + DeepSpeed autocast wrapper). Checkpoint export for LoRA uses `deepspeed.utils.zero_to_fp32` in `videotuna/utils/callbacks.py`.

Pinned in `[tool.poetry.group.training]` / `uv` `training` group (`poetry install -E cuda --with training`):

| Package | Pin | Rationale |
|---------|-----|-----------|
| `deepspeed` | **0.19.2** | ZeRO-3 CPU offload for 14B LoRA on ~40 GB GPUs; `poetry run install-deepspeed` rebuilds against the active torch/CUDA build. |
| `pytorch-lightning` | **2.4.0** | Native Wan training path (callbacks, `VideoTunaModelCheckpoint`, DeepSpeed strategy registry). Flux stays on Accelerate — no Lightning upgrade required for Phase 1. |
| `torch` | **^2.6** (cu126) | Shared base; DeepSpeed ops JIT-built against installed torch. |

### Breaking-change notes (0.19.2 + 2.4.0)

**DeepSpeed 0.19.2 — mixed-dtype ZeRO-3 + PEFT (critical for Wan LoRA)**

PR [#8066](https://github.com/deepspeedai/DeepSpeed/pull/8066) stopped blanket-casting all ZeRO-Init parameters to bf16 (correct for fp32 buffers such as RoPE `inv_freq`). That exposed a latent bug when PEFT’s default `autocast_adapter_dtype=True` keeps LoRA adapters in **fp32** while the frozen base stays **bf16**: the first optimizer step can fail in `_allgather_params_coalesced` ([DeepSpeed #8072](https://github.com/deepspeedai/DeepSpeed/issues/8072)). Upstream fix: [DeepSpeed #8073](https://github.com/deepspeedai/DeepSpeed/pull/8073) (open; no 0.19.3 release yet).

PrivTune mitigates by passing `autocast_adapter_dtype=False` to `get_peft_model()` in `videotuna/base/generation_base.py` and `videotuna/utils/wan_training.py` (same pattern as [TRL #6091](https://github.com/huggingface/trl/pull/6091)).

**PyTorch Lightning 2.4.0 — DeepSpeed integration**

- String strategy `deepspeed_stage_3_offload` maps to `DeepSpeedStrategy` (stage 3, CPU optimizer + param offload).
- `DeepSpeedOptimizer` import path fix landed in PL 2.3.x; compatible with DeepSpeed ≥ 0.14.1.
- PL 2.5.x adds `exclude_frozen_parameters` on `DeepSpeedStrategy` (useful for LoRA) but is **not required** for current configs. Defer upgrade; avoid PL 2.6.2+ (upstream supply-chain advisory). Torch 2.6 is ahead of PL 2.4’s original test matrix but works in practice with these pins.

**ZeRO-3 gradients:** use `deepspeed.utils.safe_get_full_grad(param)` if reading partitioned grads in custom code (not used in default Wan loss path).

### GPU training smoke (manual)

Requires NVIDIA GPU (~40 GB for Wan cloud smoke), Wan 14B weights under `checkpoints/wan/Wan2.1-T2V-14B`, and `data/t2v/domain/metadata.csv` + videos.

```bash
poetry install -E cuda --with training
poetry run install-deepspeed

# Short cloud smoke preset (1 epoch, checkpoint every 5 steps)
poetry run train-domain-t2v \
  --base configs/domain/wan_t2v_lora_cloud_smoke.yaml \
  --devices 0,
```

On Vast.ai after provisioning: `TRAIN_PROFILE=wan-t2v-lora ./cloud/vast/run-smoke-train.sh` (see [cloud-gpu-training.md](runbooks/cloud-gpu-training.md)).

Success: trainer reports `DeepSpeedStrategy`, completes ≥5 steps, writes `results/train/.../checkpoints/only_trained_model/denoiser-*.ckpt`.

Local dev without GPU: CPU CI covers config YAML and flow-matching helpers only (`tests/test_domain_finetune_configs.py`, `tests/test_wan_training_step.py`); ZeRO-3 behavior is not exercised in CI.

## CI smoke (CPU config validation)

```bash
poetry run lint
poetry run format-check
poetry run test tests/test_import_smoke.py -q
poetry run test tests/test_domain_finetune_configs.py -q
poetry run test tests/test_flux_lora_train_smoke.py -q
poetry run test tests/test_wan_lora_bridge.py -q
poetry run test tests/test_wan_i2v_lora_bridge.py -q
poetry run test tests/test_wan_domain_lora_smoke_22_config.py -q
poetry run test tests/test_wan_domain_i2v_smoke_22_config.py -q
poetry run test tests/test_wan_i2v_dataset.py -q
poetry run test tests/test_wan_training_step.py -q
poetry run test tests/test_poetry_scripts.py -q
```

GPU inference smoke (optional, manual — skipped in CI without GPU):

```bash
# Base pipeline only (no LoRA)
poetry run inference-wan2.2-t2v-720p \
  --config configs/inference/presets/wan2_2_cpu_smoke.yaml \
  --num_inference_steps 4 --enable_model_cpu_offload

# Domain LoRA validation (requires trained denoiser ckpt + GPU)
poetry run validate-domain-t2v \
  --trained_ckpt results/train/.../denoiser-000-000000025.ckpt \
  --num_inference_steps 4 \
  --config configs/inference/presets/wan_domain_lora_smoke_22_low_vram.yaml
```
