#!/bin/bash
mkdir results/flux

# inference a single prompt
# python scripts/inference_flux_schnell.py \
#     --prompt "A cat holding a sign that says hello world" \
#     --out_path 'results/flux/image.png' \
#     --width 1360 \
#     --height 768 \
#     --num_inference_steps 4 \
#     --guidance_scale 0.

# inference with a file of prompts
python scripts/inference_flux_schnell.py \
    --prompt inputs/t2v/prompts.txt \
    --out_path 'results/flux/' \
    --width 1360 \
    --height 768 \
    --num_inference_steps 4 \
    --guidance_scale 0.
