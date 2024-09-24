

ckpt=checkpoints/videocrafter/base_512_v2/model.ckpt
config=configs/inference/vc2_t2v_512_lora.yaml
LORACKPT="results/train/20240904012736_train_t2v_512_lora/checkpoints/trainstep_checkpoints/epoch=000109-step=000005350.ckpt"
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
