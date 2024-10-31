# ----------------------diffusers based pl inference ----------------------
ckpt='checkpoints/cogvideo/CogVideoX-2b/transformer/diffusion_pytorch_model.safetensors'
config='configs/train/005_cogvideoxft/config.yaml'
prompt_file="inputs/t2v/prompts.txt"
current_time=$(date +%Y%m%d%H%M%S)
savedir="results/t2v/$current_time-cogvideo"

python3 scripts/inference_cogvideo.py \
    --ckpt_path $ckpt \
    --config $config \
    --prompt_file $prompt_file \
    --savedir $savedir \
    --bs 1 --height 480 --width 720 \
    --fps 16 \
    --seed 6666