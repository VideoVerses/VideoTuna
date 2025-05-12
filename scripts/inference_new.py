import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm, trange

sys.path.insert(0, os.getcwd())
sys.path.insert(1, f"{os.getcwd()}/src")

from videotuna.utils.args_utils import prepare_inference_args
from videotuna.utils.common_utils import instantiate_from_config
from videotuna.base.generation_base import GenerationBase
from videotuna.utils.common_utils import monitor_resources

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default=None,
        type=str,
        help="inference mode: t2v/i2v",
    )
    #
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument(
        "--lorackpt",
        type=str,
        default=None,
        help="[Optional] checkpoint path for lora model. ",
    )
    parser.add_argument(
        "--trained_ckpt", type=str, default=None, help="denoiser full checkpoint"
    )
    parser.add_argument("--config", type=str, default=None, help="model config (yaml) path")
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
        default=None,
        help="inference standard vbench prompts",
    )
    #
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    #
    parser.add_argument(
        "--height", type=int, default=None, help="video height, in pixel space"
    )
    parser.add_argument(
        "--width", type=int, default=None, help="video width, in pixel space"
    )
    parser.add_argument(
        "--frames", type=int, default=None, help="video frame number, in pixel space"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="video motion speed. 512 or 1024 model: large value -> slow motion; 256 model: large value -> large motion;",
    )
    parser.add_argument(
        "--n_samples_prompt",
        type=int,
        default=None,
        help="num of samples per prompt",
    )
    #
    parser.add_argument("--bs", type=int, default=None, help="batch size for inference")
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=None,
        help="steps of ddim if positive, otherwise use DDPM",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=None,
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
    )
    parser.add_argument(
        "--uncond_prompt",
        type=str,
        default=None,
        help="unconditional prompts, or negative prompts",
    )
    parser.add_argument(
        "--unconditional_guidance_scale",
        type=float,
        default=None,
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
        default=None,
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
        default=None,
        help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.",
    )
    parser.add_argument(
        "--guidance_rescale",
        type=float,
        default=None,
        help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=None,
        help="generate looping videos or not",
    )
    parser.add_argument(
        "--gfi",
        action="store_true",
        default=None,
        help="generate generative frame interpolation (gfi) or not",
    )
    parser.add_argument("--savefps", type=str, default=None, help="video fps to generate")
    parser.add_argument(
        "--time_shift", 
        type=float, 
        default=None, 
        help="time shift",
    )
    parser.add_argument(
        "--num_inference_steps", 
        type=int, 
        default=None, 
        help="sampling steps",
    )
    parser.add_argument(
        "--dit_weight", 
        type=str, 
        default=None, 
        help="hunyuan dit weight",
    )
    parser.add_argument(
        "--i2v_resolution", 
        type=str, 
        default=None, 
        help="target resolution",
    )
    parser.add_argument(
        "--enable_model_cpu_offload", 
        action="store_true",
        help="model cpu offload",
    )
    parser.add_argument(
        "--enable_sequential_cpu_offload", 
        action="store_true",
        help="seqeuential cpu offload",
    )
    parser.add_argument(
        "--enable_vae_tiling", 
        action="store_true",
        help="vae tiling",
    )
    parser.add_argument(
        "--enable_vae_slicing", 
        action="store_true",
        help="vae slicing",
    )
    return parser


def run_inference(args, gpu_num=1, rank=0, **kwargs):
    """
    Inference t2v/i2v models
    """
    # load and replace inference args with user agrgument
    assert Path(args.config).exists(), f"Error: config file {args.config} NOT Found!"
    config = OmegaConf.load(args.config)
    config = prepare_inference_args(args, config)
    
    inference_config = config.pop("inference", OmegaConf.create(flags={"allow_objects": True}))
    seed_everything(inference_config.seed)

    # 1. create flow
    # 1.1 init class on meta
    # 1.2 load weight to cpu
    # 1.3 vram management (default to cuda)
    flow_config = config.pop("flow", OmegaConf.create(flags={"allow_objects": True}))
    flow : GenerationBase = instantiate_from_config(flow_config, resolve=True)
    flow.from_pretrained(inference_config.ckpt_path, inference_config.trained_ckpt, inference_config.lorackpt)
    flow.enable_vram_management()
    flow.eval()

    # 2. flow inference
    decorated_inference = monitor_resources(return_metrics=True)(flow.inference)
    metrics = decorated_inference(inference_config) 


if __name__ == "__main__":
    args = get_parser().parse_args()
    run_inference(args)
