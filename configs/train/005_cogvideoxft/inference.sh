export CUDA_VISIBLE_DEVICES=0
savedir="results/cogvideo/t2v_pl"
prompt_file="inputs/t2v/cogvideo/elon_musk_video/labels/1.txt"
environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

export environs

ckpt_path="cogVideo/CogVideoX-2b-sat/transformer/1000/mp_rank_00_model_states.pt"
# ckpt_path="results/cogvideo_train/20240913160358_correctness/checkpoints/epoch=0005-step=000600.ckpt"
python scripts/inference_cogvideo.py \
    --config "configs/train/005_cogvideoxft/config.yaml" \
    --savedir $savedir \
    --prompt_file $prompt_file \
    --ckpt_path $ckpt_path  \
    --bs 1 --height 480 --width 720 \
    --frames 13 \
    --savefps 8 \
    --seed 125

