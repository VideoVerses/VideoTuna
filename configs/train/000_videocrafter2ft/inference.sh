current_time=$(date +%Y%m%d%H%M%S)

ckpt="checkpoints/videocrafter/base_512_v2/model.ckpt"
config="configs/train/000_videocrafter2ft/config.yaml"

prompt_file="inputs/t2v/prompts.txt"
res_dir="results/000_videocrafter2ft/inference_$current_time"

python3 scripts/inference.py \
--seed 123 \
--mode 't2v' \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir \
--n_samples 1 \
--bs 2 --height 320 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_file $prompt_file \
--fps 28
