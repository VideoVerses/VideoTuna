current_time=$(date +%Y%m%d%H%M%S)

ckpt="checkpoints/open-sora/t2v_v10/OpenSora-v1-HQ-16x256x256.pth"
config='configs/003_opensora/opensorav10_256x256.yaml'

prompt_file="inputs/t2v/prompts.txt"
res_dir="results/t2v/$current_time-opensorav10-HQ-16x256x256"

python3 scripts/inference.py \
    --seed 123 \
    --mode 't2v' \
    --ckpt_path $ckpt \
    --config $config \
    --savedir $res_dir \
    --n_samples 3 \
    --bs 2 --height 256 --width 256 \
    --unconditional_guidance_scale 7.0 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --prompt_file $prompt_file \
    --fps 8 \
    --frames 16