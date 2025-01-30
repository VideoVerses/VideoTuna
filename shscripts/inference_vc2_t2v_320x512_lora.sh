ckpt=checkpoints/videocrafter/t2v_v2_512/model.ckpt
config=configs/001_videocrafter2/vc2_t2v_lora.yaml
LORACKPT=YOUR_LORA_CKPT
prompt_file=inputs/t2v/prompts.txt
res_dir=results/train/003_vc2_lora_ft

python3 scripts/inference.py \
    --seed 123 \
    --mode 't2v' \
    --ckpt_path $ckpt \
    --lorackpt $LORACKPT \
    --config $config \
    --savedir $res_dir \
    --n_samples 1 \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 12.0 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --prompt_file $prompt_file \
    --fps 28