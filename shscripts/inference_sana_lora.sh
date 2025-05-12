model_path=Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers
lora_path=YOUR_LORA_CKPT
out_path=results/t2i/sana_lora
prompt=inputs/t2i/prompts_sana_lora.txt

python scripts/inference_sana.py \
  --model_path $model_path \
  --lora_path $lora_path \
  --out_path $out_path \
  --height 1024 \
  --width 1024 \
  --num_inference_steps 20 \
  --guidance_scale 6.5 \
  --prompt $prompt \
  --seed 42