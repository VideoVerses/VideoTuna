ckpt='checkpoints/wan/Wan2.1-I2V-14B-480P/'
config='configs/008_wanvideo/wan2_1_i2v_14B_480P_fullft.yaml'
prompt_dir="inputs/i2v/576x1024"
savedir="results/i2v/wanvideo/480P"

#replace your trained checkpoint
trained_ckpt="results/train/train_wanvideo_i2v_fullft_20250427220943/checkpoints/only_trained_model/denoiser-000-000000002.ckpt"


python3 scripts/inference_new.py \
    --ckpt_path "$ckpt" \
    --trained_ckpt "$trained_ckpt" \
    --config "$config" \
    --prompt_dir "$prompt_dir" \
    --savedir "$savedir" \
    --height 480 \
    --width 832 \
    --frames 81 \
    --seed 44 \
    --num_inference_steps 40 \
    --time_shift 3.0 \
    --enable_model_cpu_offload