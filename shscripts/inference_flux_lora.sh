#!/bin/bash
export lora_ckpt="{YOUR_CORA_CKPT_PATH}"

python scripts/inference_flux_lora.py \
    --model_type dev \
    --prompt inputs/t2v/prompts.txt \
    --out_path results/t2i/flux-lora/ \
    --lora_path $lora_ckpt \
    --width 1360 \
    --height 768 \
    --num_inference_steps 50 \
    --guidance_scale 3.5 \
    --seed 42
