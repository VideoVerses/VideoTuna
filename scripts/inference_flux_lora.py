import argparse
import os

import torch
from diffusers import FluxPipeline
from videotuna.utils.inference_utils import load_prompts_from_txt


def inference(args):
    if args.model_type == "dev":
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
        )
    else:
        raise ValueError("model_type must be either 'dev'.")

    # load lora weights
    if args.lora_path is not None:
        pipe.load_lora_weights(args.lora_path)
        print("Load lora weights.")
    else:
        print("No lora weights.")

    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.to(torch.float16)

    # prompt preprocessing
    if args.prompt.endswith(".txt"):
        # model_input is a file for t2i
        prompts = load_prompts_from_txt(prompt_file=args.prompt)
        os.makedirs(args.out_path, exist_ok=True)
        out_paths = [
            os.path.join(args.out_path, f"{i:05d}_{prompts[i]}.jpg")
            for i in range(len(prompts))
        ]
    else:
        prompts = [args.prompt]
        out_paths = [args.out_path]

    for prompt, out_path in zip(prompts, out_paths):
        out = pipe(
            prompt=prompt,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            max_sequence_length=256,
            generator=torch.Generator().manual_seed(args.seed)
        ).images[0]
        out.save(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="dev", choices=["dev", "schnell"]
    )
    parser.add_argument("--prompt", type=str, default="A photo of a cat", help="Inference prompt, string or path to a .txt file")
    parser.add_argument("--out_path", type=str, default="./results/t2i/image.png")
    parser.add_argument("--lora_path", type=str, default=None, help="Full path to lora weights")
    parser.add_argument("--width", type=int, default=1360)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    inference(args)
