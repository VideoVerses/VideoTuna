# Vendor: Wan native stack

| Field | Value |
|-------|-------|
| **Path** | `videotuna/models/wan/` (in-tree snapshot; submodule migration planned) |
| **Upstream** | https://github.com/Wan-Video/Wan2.2 |
| **License** | Apache-2.0 — [upstream LICENSE.txt](https://github.com/Wan-Video/Wan2.2/blob/main/LICENSE.txt) |
| **Pinned commit** | `42bf4cfaa384bc21833865abc2f9e6c0e67233dc` (2026-03-17; basis of sync in `1100b6a`) |
| **Import / last sync** | Initial Wan2.1 import `7b1513f` (2025-04-12); Wan2.2 resync `1100b6a` (2026-06-22); PyAV patch `1ac39d9` (2026-06-23) |
| **Weight lineage** | Domain training loads **Wan 2.1** hub checkpoints (`Wan-AI/Wan2.1-T2V-14B`, `Wan-AI/Wan2.1-I2V-14B-480P`); code tree is Wan2.2-native |
| **PrivTune entrypoints** | See below |
| **Vendor deps** | `easydict` (configs only; enforced by `tests/test_vendor_import_boundary.py`) |

## PrivTune entrypoints

**Poetry scripts**

- `train-domain-t2v`, `train-domain-i2v`
- `train-wan2-1-t2v-lora`, `train-wan2-1-i2v-lora`

**Runners and flow**

- `scripts/train_new.py`, `scripts/inference_new.py`
- `videotuna/flow/wanvideo.py`
- `shscripts/inference_wanvideo_t2v_lora.sh`

**Configs**

- `configs/domain/wan_*_lora*.yaml`
- `configs/inference/presets/wan_domain_lora_smoke*.yaml`

**Not this tree** — Wan 2.2 Diffusers validation/inference uses `validate-domain-*`, `inference-wan2.2-*`, and `videotuna/flow/diffusers_video.py`.

## Local patches

Re-apply after every upstream bump:

| File | Patch |
|------|-------|
| `wan/configs/__init__.py` | Legacy Wan2.1 task aliases (`t2v-14B`, `i2v-14B`, …) for domain YAMLs |
| `wan/modules/vae.py` | Shim re-exporting `WanVAE_` from `vae2_1` for YAML targets |
| `wan/modules/t5.py` | `videotuna.utils.device_utils.resolve_inference_device()` instead of hard-coded CUDA |
## Pruned variants (not in PrivTune scope)

The following upstream Wan 2.2 variants were **removed** from this snapshot. PrivTune domain training uses **T2V/I2V LoRA only** (`train-domain-t2v`, `train-domain-i2v`).

| Removed path | Upstream variant |
|--------------|------------------|
| `wan/speech2video.py`, `wan/modules/s2v/`, `wan/configs/wan_s2v_14B.py` | Speech-to-video (s2v) |
| `wan/animate.py`, `wan/modules/animate/`, `wan/configs/wan_animate_14B.py` | Character animation |
| `wan/textimage2video.py`, `wan/configs/wan_ti2v_5B.py`, `wan/modules/vae2_2.py` | Text-image-to-video (ti2v) |

Do **not** restore these paths during upstream resync unless product scope expands. Re-copy from upstream Wan2.2 if needed in the future.

## Update procedure (in-tree snapshot)

1. Identify upstream commit to pin on [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2).
2. Diff and sync `videotuna/models/wan/wan/` from upstream `wan/` — **skip pruned variant paths** (table above).
3. Re-apply local patches (table above).
4. Update the pinned SHA and sync dates in this file.
5. Run smoke tests:

```bash
poetry run test tests/test_import_smoke.py -q
poetry run test tests/test_vendor_import_boundary.py -q
poetry run test tests/test_domain_finetune_configs.py -q
poetry run test tests/test_wan_lora_bridge.py -q
poetry run test tests/test_wan_training_step.py -q
```

## Submodule migration plan (future)

The tree predates the `videotuna/vendor/` layout. Migration is deferred until a fork or patch queue exists.

### Step A — Submodule + fork

```ini
# .gitmodules (future)
[submodule "videotuna/vendor/wan22"]
    path = videotuna/vendor/wan22
    url = https://github.com/Wan-Video/Wan2.2.git
```

PrivTune patches upstream files; use a **fork** or `git format-patch` queue before the submodule can replace the in-tree copy.

### Step B — Import-path bridge

Keep YAML targets stable (`videotuna.models.wan.wan.modules.*`) via a thin wrapper at `videotuna/models/wan/` that re-exports from the submodule or patched fork checkout. Alternative (higher churn): move to `videotuna/vendor/wan22/wan/` and update all YAML configs plus `videotuna/flow/wanvideo.py` imports.

### Step C — Submodule bump

```bash
git submodule update --init videotuna/vendor/wan22
cd videotuna/vendor/wan22
git fetch origin
git checkout <new-sha>
cd ../../..
# Re-apply patch queue or merge fork, sync into videotuna/models/wan/wan/
# Update this file with the new SHA, then run smoke tests (see above)
```
