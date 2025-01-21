import argparse
import os

import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

# create arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default="genmo/mochi-1-preview")
parser.add_argument("--prompt_file", type=str, default="inputs/t2v/prompts.txt")
parser.add_argument("--savedir", type=str, default="results/t2v/")
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--width", type=int, default=848)
parser.add_argument("--bs", type=int, default=1)
parser.add_argument("--fps", type=int, default=28)
parser.add_argument("--seed", type=int, default=123)

args = parser.parse_args()

os.makedirs(args.savedir, exist_ok=True)

pipe = MochiPipeline.from_pretrained(
    "genmo/mochi-1-preview", variant="bf16", torch_dtype=torch.bfloat16
)
# Enable memory savings
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

# there are many prompts in the prompt_file, we need to read them all
with open(args.prompt_file, "r") as file:
    prompts = file.readlines()

# set seed
torch.manual_seed(args.seed)

for index, prompt in enumerate(prompts):

    with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
        frames = pipe(prompt, num_frames=84).frames[0]

    export_to_video(frames, f"{args.savedir}/mochi_{index}.mp4", fps=30)
