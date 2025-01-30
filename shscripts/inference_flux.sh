#!/bin/bash
# inference with a file of prompts or a single prompt
# default inference with dev model
python scripts/inference_flux.py \
    --model_type dev \
    --prompt inputs/t2v/prompts.txt \
    --out_path results/flux-dev/ \
    --width 1360 \
    --height 768 \
    --num_inference_steps 50 \
    --guidance_scale 0.

# default inference with schell model
python scripts/inference_flux.py \
    --model_type schnell \
    --prompt inputs/t2v/prompts.txt \
    --out_path results/flux-schnell/ \
    --width 1360 \
    --height 768 \
    --num_inference_steps 4 \
    --guidance_scale 0.