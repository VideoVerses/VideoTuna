# Multi-GPU inference on PrivTune

PrivTune supports multi-GPU paths for Wan 2.2 Diffusers validation and optional native Wan distributed inference.

## Single-process multi-GPU (Diffusers)

For Wan 2.2 on a single host:

```shell
CUDA_VISIBLE_DEVICES=0,1 poetry run inference-wan2.2-t2v-720p --device-map auto
```

- Uses `accelerate` `infer_auto_device_map` to spread the transformer across GPUs.
- Requires `poetry install -E cuda` (accelerate is a core dependency).

## Distributed sequence parallel (xfuser)

Native Wan flows support Ulysses + Ring attention via [xfuser](https://github.com/xdit-project/xDiT).

### Requirements

- NVIDIA CUDA only (blocked on ROCm).
- `ulysses_degree × ring_degree == WORLD_SIZE` (number of processes).
- No CPU offload when USP is enabled.
- NCCL-compatible driver and peers on the same node.

### Wan native

```shell
torchrun --nproc_per_node=4 scripts/inference_new.py \
  --config configs/inference/presets/wan_domain_lora_smoke.yaml \
  --ulysses_degree 2 --ring_degree 2
```

Wan re-enables `dist.init_process_group` when `WORLD_SIZE > 1`.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `NCCL_DEBUG=INFO` | Debug collective hangs |
| `CUDA_DEVICE_MAX_CONNECTIONS=1` | Sometimes stabilizes NCCL + flash attention |
| `CUDA_VISIBLE_DEVICES` | Restrict visible GPUs before `--device cuda:0` remapping |

## Failure modes

| Symptom | Likely cause |
|---------|----------------|
| Hang at init | `ulysses × ring ≠ nproc` or missing `torchrun` |
| OOM on rank > 0 | Model loaded on all ranks without broadcast (check flow logs) |
| xfuser import error | Install CUDA extra: `poetry install -E cuda` |
| xfuser on ROCm | Use single-GPU Wan 2.2 Diffusers with `VIDEOTUNA_ATTN_BACKEND=sdpa` |

## Training multi-GPU

- **Wan Lightning:** `--devices N` in `train-wan2-1-t2v-lora` / `scripts/train_new.py`
- **DeepSpeed:** `poetry run install-deepspeed` for ZeRO stage configs in domain YAMLs

## Device selection with `CUDA_VISIBLE_DEVICES`

When GPUs are remapped, always use logical indices after remapping:

```shell
CUDA_VISIBLE_DEVICES=1 poetry run inference-wan2.2-t2v-720p --device cuda:0
```

`--device cuda:1` selects the second *visible* GPU.
