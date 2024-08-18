import torch
from diffusers import FluxPipeline
import argparse

def inference(args):
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    # pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.to(torch.float16) 
    out = pipe(
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=256,
    ).images[0]
    out.save(args.out_path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prompt',type=str,default="A cat holding a sign that says hello world")
    parser.add_argument('--out_path',type=str,default='./image.png')
    parser.add_argument('--width',type=int,default=1360)
    parser.add_argument('--height',type=int,default=768)
    parser.add_argument('--num_inference_steps',type=int,default=4)
    parser.add_argument('--guidance_scale',type=float,default=0.)
    args=parser.parse_args()
    inference(args)