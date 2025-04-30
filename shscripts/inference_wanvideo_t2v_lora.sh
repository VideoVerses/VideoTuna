ckpt='checkpoints/wan/Wan2.1-T2V-14B/'
config='configs/008_wanvideo/wan2_1_t2v_14B_lora.yaml'
prompt_file="inputs/t2v/prompts.txt"
savedir="results/t2v/wanvideo/480P"

#replace your trained checkpoint
trained_ckpt="results/train/train_wanvideo_t2v_lora_20250429045205/checkpoints/only_trained_model/denoiser-000-000000050.ckpt"

python3 scripts/inference_new.py \
    --ckpt_path "$ckpt" \
    --trained_ckpt "$trained_ckpt" \
    --config "$config" \
    --prompt_file "$prompt_file" \
    --savedir "$savedir" \
    --height 480 \
    --width 832 \
    --frames 81 \
    --seed 44 \
    --time_shift 3.0 \
    --num_inference_steps 50 \
    --enable_model_cpu_offload