current_time=$(date +%Y%m%d%H%M%S)

ckpt="/project/llmsvgen/share/videotuna_ckpt/opensorav10/model_ckpt.pt"
config='configs/train/001_opensorav10/config_opensorav10.yaml'

prompt_file="inputs/t2v/prompts.txt"
res_dir="results/001_opensorav10/inference_for_eval_$current_time"

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
--frames 16 \
--standard_vbench