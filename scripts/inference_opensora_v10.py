import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import colossalai
import torch.distributed as dist
from mmengine.runner import set_random_seed

from src.opensora.datasets import save_sample
from src.opensora.registry import MODELS, SCHEDULERS, build_module
from src.opensora.utils.config_utils import parse_configs
from src.opensora.utils.misc import to_torch_dtype, is_distributed
from src.opensora.acceleration.parallel_states import set_sequence_parallel_group
from colossalai.cluster import DistCoordinator


def load_prompts(prompt_path):

    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts


def main():
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False)
    print(cfg)

    # == init distributed env ==
    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    else:
        coordinator = None
        enable_sequence_parallelism = False

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)
    prompts = load_prompts(cfg.prompt_path)

    # ======================================================
    # 3. build model & load weights
    # ======================================================
    # 3.1. build model
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size(input_size)
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)  # T5 must be fp32
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        dtype=dtype,
        enable_sequence_parallelism=enable_sequence_parallelism,
    )
    text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    # 3.2. move to device & eval
    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()

    # 3.3. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ======================================================
    # 4. inference
    # ======================================================
    sample_idx = 0
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    for i in range(0, len(prompts), cfg.batch_size):
        if sample_idx >= cfg.infer_num:
            break
        batch_prompts = prompts[i : i + cfg.batch_size]
        samples = scheduler.sample(
            model,
            text_encoder,
            z_size=(vae.out_channels, *latent_size),
            prompts=batch_prompts,
            device=device,

        )
        samples = vae.decode(samples.to(dtype))

        for idx, sample in enumerate(samples):
            prompt = batch_prompts[idx]
            print(f"Prompt: {prompt}")
            save_path = os.path.join(save_dir, f"{prompt.replace(' ', '_')}")
            save_sample(sample, fps=cfg.fps, save_path=save_path)
            sample_idx += 1


if __name__ == "__main__":
    main()
