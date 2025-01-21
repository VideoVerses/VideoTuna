import argparse
import json
import os
import sys
import time
from functools import partial

import numpy as np
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm, trange

sys.path.insert(0, os.getcwd())
sys.path.insert(1, f"{os.getcwd()}/src")
from videotuna.base.ddim import DDIMSampler
from videotuna.base.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from videotuna.utils.common_utils import instantiate_from_config
from videotuna.utils.inference_utils import (
    load_image_batch,
    load_inputs_i2v,
    load_model_checkpoint,
    load_prompts_from_txt,
    sample_batch_i2v,
    sample_batch_t2v,
    save_videos,
    save_videos_vbench,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="t2v",
        type=str,
        help="inference mode: t2v/i2v",
        choices=["t2v", "i2v"],
    )
    #
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="model config (yaml) path")
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="a text file containing many prompts for text-to-video",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=None,
        help="a input dir containing images and prompts for image-to-video/interpolation",
    )
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument(
        "--standard_vbench",
        action="store_true",
        default=False,
        help="inference standard vbench prompts",
    )
    #
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    #
    parser.add_argument(
        "--height", type=int, default=320, help="video height, in pixel space"
    )
    parser.add_argument(
        "--width", type=int, default=512, help="video width, in pixel space"
    )
    parser.add_argument(
        "--frames", type=int, default=None, help="video frame number, in pixel space"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="video motion speed. 512 or 1024 model: large value -> slow motion; 256 model: large value -> large motion;",
    )
    parser.add_argument(
        "--n_samples_prompt",
        type=int,
        default=1,
        help="num of samples per prompt",
    )
    #
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="steps of ddim if positive, otherwise use DDPM",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
    )
    parser.add_argument(
        "--uncond_prompt",
        type=str,
        default="",
        help="unconditional prompts, or negative prompts",
    )
    parser.add_argument(
        "--unconditional_guidance_scale",
        type=float,
        default=12.0,
        help="prompt classifier-free guidance",
    )
    parser.add_argument(
        "--unconditional_guidance_scale_temporal",
        type=float,
        default=None,
        help="temporal consistency guidance",
    )
    # dc args
    parser.add_argument(
        "--multiple_cond_cfg",
        action="store_true",
        default=False,
        help="i2v: use multi-condition cfg or not",
    )
    parser.add_argument(
        "--cfg_img",
        type=float,
        default=None,
        help="guidance scale for image conditioning",
    )
    parser.add_argument(
        "--timestep_spacing",
        type=str,
        default="uniform",
        help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.",
    )
    parser.add_argument(
        "--guidance_rescale",
        type=float,
        default=0.0,
        help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=False,
        help="generate looping videos or not",
    )
    parser.add_argument(
        "--gfi",
        action="store_true",
        default=False,
        help="generate generative frame interpolation (gfi) or not",
    )
    # lora args
    parser.add_argument(
        "--lorackpt",
        type=str,
        default=None,
        help="[Optional] checkpoint path for lora model. ",
    )
    #
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    return parser


def load_model(args, cuda_idx=0):
    """
    Create model and load weight.
    """
    # build model
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    if args.lorackpt is not None:
        model_config["params"]["lora_args"] = {"lora_ckpt": args.lorackpt}
    model = instantiate_from_config(model_config)
    model = model.cuda(cuda_idx)
    # load weights
    assert os.path.exists(
        args.ckpt_path
    ), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    # load lora weights
    if hasattr(model, "lora_args") and len(model.lora_args) != 0:
        model.inject_lora()

    model.eval()
    return model


def load_inputs(args):
    """
    load inputs:
        t2v: prompts
        i2v: prompts + images
    """
    assert (
        args.prompt_file is not None or args.prompt_dir is not None
    ), "Error: input file/dir NOT Found!"

    if args.prompt_file is not None:
        assert os.path.exists(args.prompt_file)
        # load inputs for t2v
        prompt_list = load_prompts_from_txt(args.prompt_file)
        num_prompts = len(prompt_list)
        filename_list = [f"prompt-{idx+1:04d}" for idx in range(num_prompts)]
        image_list = None
    elif args.prompt_dir is not None:
        assert os.path.exists(args.prompt_dir)
        # load inputs for i2v
        filename_list, image_list, prompt_list = load_inputs_i2v(
            args.prompt_dir,
            video_size=(args.height, args.width),
            video_frames=args.frames,
        )
    return prompt_list, image_list, filename_list


def run_inference(args, gpu_num=1, rank=0, **kwargs):
    """
    Inference t2v/i2v models
    """
    assert (args.height % 16 == 0) and (
        args.width % 16 == 0
    ), "Error: image size [h,w] should be multiples of 16!"

    seed_everything(args.seed)
    os.makedirs(args.savedir, exist_ok=True)

    # load model, sampler, inputs
    model = load_model(args)
    if args.mode == "i2v" and args.multiple_cond_cfg:
        ddim_sampler = DDIMSampler_multicond(model)
    else:
        ddim_sampler = DDIMSampler(model)
    args.frames = model.temporal_length if args.frames is None else args.frames
    prompt_list, image_list, filename_list = load_inputs(args)

    # split across multiple gpus
    num_samples = len(prompt_list)
    num_samples_rank = num_samples // gpu_num
    remainder = num_samples % gpu_num
    indices_rank = list(range(num_samples_rank * rank, num_samples_rank * (rank + 1)))
    if rank == 0 and remainder != 0:
        indices_rank = indices_rank + list(range(num_samples - remainder, num_samples))
    #
    prompt_list_rank = [prompt_list[i] for i in indices_rank]
    filename_list_rank = [filename_list[i] for i in indices_rank]
    if args.mode == "i2v":
        image_list_rank = [image_list[i] for i in indices_rank]

    # noise shape
    h, w, frames, channels = (
        args.height // 8,
        args.width // 8,
        args.frames,
        model.channels,
    )

    # -----------------------------------------------------------------
    # inference
    format_file = {}
    start = time.time()
    n_iters = len(prompt_list_rank) // args.bs + (
        1 if len(prompt_list_rank) % args.bs else 0
    )
    with torch.no_grad():
        for idx in trange(0, n_iters, desc="Sample Iters"):
            # print(f'[rank:{rank}] batch {idx}: prompt bs {args.bs}) x nsamples_per_prompt {args.n_samples_prompt} ...')

            prompts = prompt_list_rank[idx * args.bs : (idx + 1) * args.bs]
            filenames = filename_list_rank[idx * args.bs : (idx + 1) * args.bs]

            if args.mode == "i2v":
                images = image_list_rank[idx * args.bs : (idx + 1) * args.bs]
                if isinstance(images, list):
                    images = torch.stack(images, dim=0).to("cuda")
                else:
                    images = images.unsqueeze(0).to("cuda")
            # idx_s = idx*args.bs
            # idx_e = min(idx_s+args.bs, len(prompt_list_rank))
            # batch_size = idx_e - idx_s
            # filenames = filename_list_rank[idx_s:idx_e]

            # prompts = prompt_list_rank[idx_s:idx_e]
            # if isinstance(prompts, str):
            #     prompts = [prompts]
            # prompts = batch_size * [""]

            # if args.mode == 't2v':
            #     cond = {"c_crossattn": [text_emb], "fps": fps}

            # TODO
            # elif args.mode == 'i2v':
            #     cond_images = load_image_batch(image_list_rank[idx_s:idx_e], (args.height, args.width))
            #     cond_images = cond_images.to(model.device)
            #     img_emb = model.get_image_embeds(cond_images)
            #     imtext_cond = torch.cat([text_emb, img_emb], dim=1)
            #     cond = {"c_crossattn": [imtext_cond], "fps": fps}
            # else:
            #     raise NotImplementedError

            ## inference
            bs = args.bs if args.bs == len(prompts) else len(prompts)
            noise_shape = [bs, channels, frames, h, w]
            if args.mode == "t2v":
                batch_samples = sample_batch_t2v(
                    model,
                    ddim_sampler,
                    prompts,
                    noise_shape,
                    args.fps,
                    args.n_samples_prompt,
                    args.ddim_steps,
                    args.ddim_eta,
                    args.unconditional_guidance_scale,
                    args.unconditional_guidance_scale_temporal,
                    args.uncond_prompt,
                )
            elif args.mode == "i2v":
                batch_samples = sample_batch_i2v(
                    model,
                    ddim_sampler,
                    prompts,
                    images,
                    noise_shape,
                    args.n_samples_prompt,
                    args.ddim_steps,
                    args.ddim_eta,
                    args.unconditional_guidance_scale,
                    args.cfg_img,
                    args.fps,
                    args.uncond_prompt,
                    args.multiple_cond_cfg,
                    args.loop,
                    args.gfi,
                    args.timestep_spacing,
                    args.guidance_rescale,
                )
            else:
                raise ValueError

            if args.standard_vbench:
                save_videos_vbench(
                    batch_samples, args.savedir, prompts, format_file, fps=args.savefps
                )
                print("test")
            else:
                save_videos(batch_samples, args.savedir, filenames, fps=args.savefps)

    if args.standard_vbench:
        with open(os.path.join(args.savedir, "info.json"), "w") as f:
            json.dump(format_file, f)

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


if __name__ == "__main__":

    args = get_parser().parse_args()
    run_inference(args)
