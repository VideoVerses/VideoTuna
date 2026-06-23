# Prepare checkpoints

PrivTune domain training and validation use Hugging Face hub weights (downloaded on first run) or local clones under `checkpoints/`.

## Supported models

| Phase | Model | Hub ID | Local path (optional) |
|-------|-------|--------|----------------------|
| T2I LoRA | FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | — (HF cache) |
| T2V LoRA train | Wan 2.1 T2V 14B | `Wan-AI/Wan2.1-T2V-14B` | `checkpoints/wan/Wan2.1-T2V-14B` |
| I2V LoRA train | Wan 2.1 I2V 14B 480P | `Wan-AI/Wan2.1-I2V-14B-480P` | `checkpoints/wan/Wan2.1-I2V-14B-480P` |
| T2V validate | Wan 2.2 Diffusers | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | — (HF cache) |
| I2V validate | Wan 2.2 I2V Diffusers | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | — (HF cache) |

## Compute compatibility

| Backend | Flux T2I | Wan 2.1 train | Wan 2.2 infer |
|---------|----------|---------------|---------------|
| NVIDIA CUDA | Yes | Yes (+ DeepSpeed) | Yes |
| AMD ROCm | Yes (`sdpa`) | Experimental | Yes (`sdpa` + offload) |
| CPU | Config/smoke only | Config validation only | Tiny smoke preset only |

Install: NVIDIA `poetry install -E cuda --with training` · AMD [`docs/install-rocm.md`](install-rocm.md) · CPU [`docs/install-cpu.md`](install-cpu.md)

See [`docs/MODEL_VERSIONS.md`](MODEL_VERSIONS.md) and [`docs/runbooks/domain-adult-finetune.md`](runbooks/domain-adult-finetune.md) for commands and presets.

## Download (offline / air-gapped)

```bash
mkdir -p checkpoints/wan
cd checkpoints/wan
hf download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
```

Cloud renters on Vast.ai can opt into faster multi-GB hub pulls with `VIDEOTUNA_FAST_HF_DOWNLOAD=1` at instance launch — see [`docs/runbooks/cloud-gpu-training.md`](runbooks/cloud-gpu-training.md#fast-model-downloads-opt-in). Local dev is unchanged.

Flux and Wan 2.2 Diffusers weights are pulled from the hub on first `train-flux-lora` or `inference-wan2.2-t2v-720p` run unless you set `HF_HOME` or pass `--ckpt_path` / config overrides.

## Commands

| Use case | Command |
|----------|---------|
| Flux LoRA train | `poetry run train-domain-t2i` |
| Flux LoRA smoke | `poetry run inference-domain-t2i` |
| Wan LoRA train | `poetry run train-domain-t2v` |
| Wan I2V LoRA train | `poetry run train-domain-i2v` |
| Wan native smoke | `poetry run python scripts/inference_new.py --config configs/inference/presets/wan_domain_lora_smoke.yaml` |
| Wan 2.2 T2V validation | `poetry run inference-wan2.2-t2v-720p` |
| Wan 2.2 I2V validation | `poetry run validate-domain-i2v` |

## Checkpoint layout

```
VideoTuna/
    └── checkpoints/
        └── wan/
            └── Wan2.1-T2V-14B/    # required for Wan 2.1 LoRA training
```

Training outputs go under `results/train/` (not committed).
