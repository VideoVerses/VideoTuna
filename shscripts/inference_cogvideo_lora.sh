# ----------------------diffusers based pl inference ----------------------
# ‘configs/004_cogvideox/cogvideo2b.yaml’ or 'configs/004_cogvideox/cogvideo5b.yaml'
config='configs/004_cogvideox/cogvideo2b.yaml'
prompt_file="inputs/t2v/prompts.txt"
current_time=$(date +%Y%m%d%H%M%S)
savedir="results/t2v/$current_time-cogvideo"
ckpt="{YOUR_CKPT_PATH}"

python3 scripts/inference_cogvideo.py \
--ckpt_path $ckpt \
--config $config \
--prompt_file $prompt_file \
--savedir $savedir \
--bs 1 --height 480 --width 720 \
--fps 16 \
--seed 6666 \