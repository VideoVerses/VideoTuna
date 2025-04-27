import argparse
import os

import torch
from diffusers import SanaPipeline

from videotuna.utils.inference_utils import load_prompts_from_txt

def inference(args):
    pipe = SanaPipeline.from_pretrained(
        args.model_path,
        variant="bf16",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.vae.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)
    
    if args.lora_path:
        pipe.load_lora_weights(args.lora_path)

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
        "--model_path", 
        type=str, 
        default="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
    )
    parser.add_argument('--lora_path', type=str, default=None, help='Path to the LoRA weights')
    parser.add_argument("--prompt", type=str, default="A cat holding a sign that says hello world")
    parser.add_argument("--out_path", type=str, default="results/sana/output.png")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument('--seed', type=int, default=42, help='The random seed for the generator.')
    args = parser.parse_args()
    inference(args)