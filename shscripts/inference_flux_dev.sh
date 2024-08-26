#!/bin/bash
python inference_flux_dev.py \
--prompt "A cat holding a sign that says hello world" \
--out_path './image.png' \
--width 1360 \
--height 768 \
--num_inference_steps 50 \
--guidance_scale 3.5