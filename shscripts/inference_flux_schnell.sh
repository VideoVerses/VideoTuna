#!/bin/bash
mkdir results/flux
python scripts/inference_flux_schnell.py \
--prompt "A cat holding a sign that says hello world" \
--out_path 'results/flux/image.png' \
--width 1360 \
--height 768 \
--num_inference_steps 4 \
--guidance_scale 0.