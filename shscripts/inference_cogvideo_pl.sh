environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"
export environs


savedir="results/cogvideo/t2v_pl/125"
prompt_file="src/sat/configs/test.txt"
ckpt_path="cogVideo/CogVideoX-2b-sat/transformer/1000/mp_rank_00_model_states.pt"

# orig arg format
python scripts/inference_cogvideo.py \
    --base configs/inference/cogvideo_t2v_pl.yaml  \
    --input-file $prompt_file \
    --load $ckpt_path \
    --sampling-num-frames 13 \
    --sampling-fps 8 \
    --force-inference \
    --seed 123

# vc2 args format
# python scripts/inference_cogvideo.py \
#     --base configs/inference/cogvideo_t2v_pl.yaml  \
#     --savedir $savedir \
#     --prompt_file $prompt_file \
#     --load "cogVideo/CogVideoX-2b-sat/transformer/1000/mp_rank_00_model_states.pt" \
#     --bs 1 --height 480 --width 720 \
#     --frames 13 \
#     --savefps 8 \
#     --seed 125

