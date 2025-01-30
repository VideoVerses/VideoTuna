config=configs/004_cogvideox/cogvideo5b-i2v.yaml
ckpt=results/train/cogvideox_i2v_5b/$YOUR_CKPT_PATH.ckpt
prompt_dir=$YOUR_PROMPT_DIR

current_time=$(date +%Y%m%d%H%M%S)
savedir="results/inference/i2v/cogvideox-i2v-lora-$current_time"

python3 scripts/inference_cogvideo.py \
    --config $config \
    --ckpt_path $ckpt \
    --prompt_dir $prompt_dir \
    --savedir $savedir \
    --bs 1 --height 480 --width 720 \
    --fps 16 \
    --seed 6666 \
    --mode i2v