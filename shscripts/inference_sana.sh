model_path=Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers
prompt=inputs/t2i/prompts.txt
out_path=results/t2i/sana

python scripts/inference_sana.py \
    --model_path $model_path \
    --out_path $out_path \
    --width 1024 \
    --height 1024 \
    --num_inference_steps 20 \
    --guidance_scale 4.5 \
    --prompt $prompt \
    --seed 42