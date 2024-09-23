# ----------------------diffusers based pl inference ----------------------
ckpt='checkpoints/cogvideo/transformer/1000/mp_rank_00_model_states.pt'
config='configs/train/005_cogvideoxft/cogvideo_diffusers.yaml'
prompt_file="inputs/t2v/prompts.txt"
current_time=$(date +%Y%m%d%H%M%S)
savedir="results/t2v/$current_time-cogvideo"

python3 scripts/inference_cogVideo_diffusers.py \
    --ckpt_path $ckpt \
    --config $config \
    --prompt_file $prompt_file \
    --savedir $savedir \
    --bs 1 --height 480 --width 720 \
    --fps 16 \
    --seed 6666




# ----------------------sat based pl inference ----------------------

# export CUDA_VISIBLE_DEVICES=0
# savedir="results/cogvideo/t2v_pl"
# prompt_file="inputs/t2v/cogvideo/elon_musk_video/labels/1.txt"
# environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

# export environs

# ckpt_path="cogVideo/CogVideoX-2b-sat/transformer/1000/mp_rank_00_model_states.pt"
# # ckpt_path="results/cogvideo_train/20240913160358_correctness/checkpoints/epoch=0005-step=000600.ckpt"
# python scripts/inference_cogvideo.py \
#     --config "configs/train/005_cogvideoxft/config.yaml" \
#     --savedir $savedir \
#     --prompt_file $prompt_file \
#     --ckpt_path $ckpt_path  \
#     --bs 1 --height 480 --width 720 \
#     --frames 13 \
#     --savefps 8 \
#     --seed 125

