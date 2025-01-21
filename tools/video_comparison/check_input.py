import argparse
import os

parser = argparse.ArgumentParser(description="Check the input directory")
parser.add_argument(
    "--input_dir", type=str, help="The input should be a directory", required=True
)
parser.add_argument(
    "--seed", type=int, help="The seed for the random number generator", default=42
)
args = parser.parse_args()

# check if there are images in the input directory, jpg/png...
image_files = [
    f for f in os.listdir(args.input_dir) if f.endswith(".jpg") or f.endswith(".png")
]
num_of_prompts = len(open(f"{args.input_dir}/prompts.txt", "r").readlines())
num_of_images = len(image_files)
print(f"number of prompts: {num_of_prompts}")
print(f"number of images: {num_of_images}")

if num_of_images != 0:
    assert (
        num_of_prompts == num_of_images
    ), "The number of prompts should be equal to the number of images"
else:
    # create images using flux
    import torch
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    lines = open(f"{args.input_dir}/prompts.txt", "r").readlines()
    for index, line in enumerate(lines):
        prompt = line.strip()
        print(f"creating image {index} using prompt: {prompt}")

        out = pipe(
            prompt=prompt,
            guidance_scale=0.0,
            height=576,
            width=1024,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator("cuda").manual_seed(args.seed),
        ).images[0]
        index_str = str(index).zfill(5)
        out.save(f"{args.input_dir}/prompt_{index_str}.png")
