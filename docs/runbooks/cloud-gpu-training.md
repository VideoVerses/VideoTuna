# Cloud GPU training runbook (Vast.ai / linux-desktop template)

Headless training on rented NVIDIA GPUs using the PrivTune `cloud/vast/` provisioning bundle. Primary workflow: **Flux LoRA T2I → Wan 2.1 T2V LoRA** (see [domain-adult-finetune.md](domain-adult-finetune.md)).

Never commit datasets, weights, API keys, or `results/` to git.

## A. Instance selection

| Workload | Peak VRAM | GPU examples | Notes |
|----------|-----------|--------------|-------|
| Flux LoRA @ 512px | ~24–40 GB | RTX 4090 24GB, A100 40GB | No DeepSpeed required |
| Wan 2.1 T2V LoRA @ 480×832×81 | ~38 GB | A100 40GB, H100 | Requires DeepSpeed ZeRO-3 offload |

**CUDA:** Template should ship NVIDIA driver + CUDA 12.x compatible with PrivTune's cu126 PyTorch wheels (`poetry install -E cuda --with training`). For faster Wan video dataloading, add `-E video-fast` (optional `torchcodec` extra; PyAV remains the fallback).

**Disk:** Base weights are large (Wan 14B ≈ tens of GB; full train + validate bundle ≈ 200 GB+). Use **≥200 GB** volume. Pre-download via manifest when `HF_TOKEN` is set:

| Model | When downloaded | Purpose |
|-------|-----------------|---------|
| `black-forest-labs/FLUX.1-dev` | Manifest + bootstrap | Flux T2I train |
| `Wan-AI/Wan2.1-T2V-14B` | Manifest + bootstrap | Wan T2V train |
| `Wan-AI/Wan2.1-I2V-14B-480P` | Manifest + bootstrap | Wan I2V train (optional) |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | Bootstrap (`hf-download-cache`) | `validate-domain-t2v` |
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | Bootstrap (`hf-download-cache`) | `validate-domain-i2v` |

Wan 2.2 Diffusers weights populate the Hugging Face hub cache so validation presets (hub IDs) resolve without config changes.

## B. Launch checklist

1. Rent a **linux-desktop** (Selkies/VNC + SSH + Jupyter) template with a compatible CUDA tag.
2. Set template environment variables at rent time:

| Variable | Value |
|----------|-------|
| `WORKSPACE` | `/workspace` |
| `PROVISIONING_MANIFEST` | `https://raw.githubusercontent.com/miguelenes/VideoTuna/main/cloud/vast/provisioning.yaml` |
| `HF_TOKEN` | `hf_...` (required for FLUX.1-dev; accept license on Hugging Face first) |
| `VIDEOTUNA_ATTN_BACKEND` | `auto` or `sdpa` if flash-attn not installed |

### Fast model downloads (opt-in)

Multi-GB first-boot pulls (FLUX.1-dev, Wan 2.1 train weights, Wan 2.2 validate weights) can dominate rental cost on datacenter GPUs with fast network and NVMe storage. PrivTune exposes an **opt-in** cloud knob:

| Variable | Value |
|----------|-------|
| `VIDEOTUNA_FAST_HF_DOWNLOAD` | `1` |

When set at **rent time**, bootstrap exports `HF_XET_HIGH_PERFORMANCE=1` (modern `hf-xet` high-bandwidth mode; **not** deprecated `hf_transfer`) and persists it into `.env` for training-time hub pulls.

**Important:** `conditional_downloads` in the manifest run **before** `bootstrap.sh`. Set `VIDEOTUNA_FAST_HF_DOWNLOAD=1` when launching the instance so both manifest-phase and bootstrap-phase pulls benefit.

**When to use:** datacenter GPU + NVMe, multi-GB weight pre-downloads.

**Caveats:** higher CPU/RAM use; best on SSD/NVMe. On spinning disks, consider `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1`. Leave unset for local dev (default adaptive `hf-xet` only).

See [HF Xet env vars](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfxethighperformance).

Fallback imperative provisioner:

```text
PROVISIONING_SCRIPT=https://raw.githubusercontent.com/miguelenes/VideoTuna/main/cloud/vast/bootstrap.sh
```

### Provisioning retries

Two independent retry layers apply during first boot:

**Manifest-level (Vast provisioner)** — configured in [`cloud/vast/provisioning.yaml`](../../cloud/vast/provisioning.yaml):

| Key | Behavior |
|-----|----------|
| `settings.retry` | Exponential backoff on manifest-phase downloads (`conditional_downloads`, wget, apt). Default: 5 attempts, delays 2s → 4s → 8s → 16s. |
| `post_commands` | **No per-command retry.** `bootstrap.sh` is fail-fast; a non-zero exit aborts the phase. |
| `on_failure` | Whole-pipeline retry: sleep **60s**, re-run from the failed phase (idempotent hash skip), up to **3** times; then `action: continue` (log + exit 1, instance stays up). |

Override manifest failure behavior at rent time: `PROVISIONER_RETRY_MAX`, `PROVISIONER_RETRY_DELAY`, `PROVISIONER_FAILURE_ACTION`.

**Bootstrap-level (tenacity)** — [`cloud/vast/provision_retry.py`](../../cloud/vast/provision_retry.py) mirrors `settings.retry` for network-heavy steps inside `bootstrap.sh`: Poetry install, DeepSpeed install, and bootstrap-phase `hf download`. This catches transient flakes without waiting 60s for a full manifest retry.

Manual recovery (idempotent): `bash /workspace/VideoTuna/cloud/vast/bootstrap.sh`

3. **SSH in** (preferred) or use the Jupyter terminal on port 8080.
4. Wait for provisioning to finish:
   - Template marker: `/.provisioning_complete`
   - PrivTune marker: `/workspace/.videotuna_provisioned`
   - Logs: `/var/log/portal/provisioning.log` (if present)
5. Confirm smoke tests passed during bootstrap (`poetry run test tests/test_import_smoke.py`).
6. Sync datasets via Syncthing (port **8384**) — see [C. Data sync](#c-data-sync).
7. Start training:

```bash
cd /workspace/VideoTuna
source .env
export TRAIN_PROFILE=flux-lora
./cloud/vast/run-smoke-train.sh    # short GPU validation
./cloud/vast/run-train.sh          # full run
```

**Supervisor (survives SSH disconnect):**

```bash
cp cloud/vast/supervisor/videotuna-train.conf /etc/supervisor/conf.d/
supervisorctl reread && supervisorctl update
# Set TRAIN_PROFILE in .env first
supervisorctl start videotuna-train
supervisorctl status
```

## C. Data sync

Provisioner creates Syncthing-friendly paths and symlinks into the repo:

```
/workspace/data/t2i/domain/     ↔  local data/t2i/domain/
/workspace/data/t2v/domain/     ↔  local data/t2v/domain/
/workspace/results/             ↔  local results/  (pull back after training)
/workspace/checkpoints/         ↔  local checkpoints/  (optional, large)
```

Repo-relative configs (`data/t2i/domain`, `checkpoints/wan/...`) resolve via symlinks under `/workspace/VideoTuna/`.

**Flux layout:** paired `0001.jpg` + `0001.txt` with trigger token `sks_style`.

**Wan layout:** `metadata.csv` + `videos/*.mp4` at 480×832, 81 frames.

## D. Monitoring

```bash
tail -f /workspace/results/train.log
tail -f /workspace/results/train.err
supervisorctl status videotuna-train
nvidia-smi -l 5
```

**TensorBoard (training metrics):** event files live under each run directory (Flux: `{output_dir}/tensorboard/`; Wan: `{workdir}/tensorboard/`). View locally with SSH port-forward:

```bash
tensorboard --logdir /workspace/PrivTune/results/train --bind_all --port 6006
# then open http://<instance-ip>:6006 from your laptop
```

**Trackio (Flux, optional):** when `VIDEOTUNA_METRICS_BACKEND=trackio` and the `trackio` extra is installed, metrics sync to a local SQLite database and optional private Hugging Face Space (`VIDEOTUNA_TRACKIO_SPACE_ID`). View with `trackio show` or open the Space URL — no port-forward required. See [domain-adult-finetune.md](domain-adult-finetune.md#optional-trackio-flux-phase-1).

Smoke run logs: `/workspace/results/smoke-train.log`.

## E. Checkpoint recovery

| Phase | Checkpoint path |
|-------|-----------------|
| Flux LoRA | `results/train/flux-domain-adult/checkpoint-<step>/` |
| Flux smoke | `results/train/flux-cloud-smoke/checkpoint-<step>/` |
| Wan LoRA | `results/train/train_wan_domain_t2v_lora_<timestamp>/checkpoints/only_trained_model/denoiser-*.ckpt` |

**Before terminating the instance:** Syncthing `results/` (and optionally `checkpoints/`) back to your machine.

Resume Wan training:

```bash
export TRAIN_PROFILE=wan-t2v-lora
export RESUME_CKPT=/workspace/results/train/.../checkpoints/...
./cloud/vast/run-train.sh
```

## F. Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA OOM (Flux) | Lower `--resolution` in JSON; keep `gradient_checkpointing: true` |
| CUDA OOM (Wan) | Confirm `poetry run install-deepspeed` succeeded; reduce frames/resolution in YAML |
| HF gated model | Set `HF_TOKEN`; `huggingface-cli login`; accept FLUX.1-dev license |
| flash-attn build fail | `export VIDEOTUNA_ATTN_BACKEND=sdpa` in `.env`; do not run `install-flash-attn` |
| DeepSpeed build fail | Check CUDA toolkit / nvcc; re-run `poetry run install-deepspeed` |
| Wan grey preview | Use `unconditional_guidance_scale: 12.0` in training YAML `image_logger` |
| Provisioning retry | Manifest `on_failure` retries the whole pipeline (60s delay); bootstrap uses `provision_retry.py` for finer-grained retries on poetry/HF steps. Re-run `bash /workspace/VideoTuna/cloud/vast/bootstrap.sh` manually. |
| Slow HF weight download | Set `VIDEOTUNA_FAST_HF_DOWNLOAD=1` at rent time (see [Fast model downloads](#fast-model-downloads-opt-in)) |

## G. Cost control

1. Run `./cloud/vast/run-smoke-train.sh` before any multi-hour job.
2. Stop the instance when finished — only persist `results/` and `checkpoints/` via Syncthing.
3. Use smoke configs: `configs/domain/flux_t2i_cloud_smoke.json`, `configs/domain/wan_t2v_lora_cloud_smoke.yaml`.

## Training profiles (`TRAIN_PROFILE`)

| Profile | Default config |
|---------|----------------|
| `flux-lora` | `configs/domain/flux_t2i.json` |
| `wan-t2v-lora` | `configs/domain/wan_t2v_lora.yaml` |
| `flux-lora` (smoke) | `configs/domain/flux_t2i_cloud_smoke.json` |
| `wan-t2v-lora` (smoke) | `configs/domain/wan_t2v_lora_cloud_smoke.yaml` |

Override with `CONFIG_PATH` and `DATA_CONFIG_PATH` (Flux only) in `.env`.

## Related docs

- [domain-adult-finetune.md](domain-adult-finetune.md) — dataset layout, hyperparameters, inference smoke
- [../checkpoints.md](../checkpoints.md) — weight download layout
- [`cloud/vast/provisioning.yaml`](../../cloud/vast/provisioning.yaml) — manifest source
- [`AGENTS.md`](../../AGENTS.md) — local dev verification gates
