resolution="720P"

if [ "$resolution" = "480P" ]; then
    ckpt='checkpoints/wan/Wan2.1-I2V-14B-480P/'
    config='configs/008_wanvideo/wan2_1_i2v_14B_480P.yaml'
    prompt_dir="inputs/i2v/576x1024"
    savedir="results/i2v/wanvideo/480P"

    python3 scripts/inference_new.py \
        --ckpt_path "$ckpt" \
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
        
elif [ "$resolution" = "720P" ]; then
    #720P
    ckpt='checkpoints/wan/Wan2.1-I2V-14B-720P/'
    config='configs/008_wanvideo/wan2_1_i2v_14B_720P.yaml'
    prompt_dir="inputs/i2v/576x1024"
    savedir="results/i2v/wanvideo/720P"

    python3 scripts/inference_new.py \
        --ckpt_path "$ckpt" \
        --config "$config" \
        --prompt_dir "$prompt_dir" \
        --savedir "$savedir" \
        --height 720 \
        --width 1280 \
        --frames 81 \
        --seed 44 \
        --num_inference_steps 40 \
        --time_shift 5.0 \
        --enable_model_cpu_offload
else
    echo "Unsupported resolution: $resolution"
    exit 1
fi