

export CUDA_VISIBLE_DEVICES=1
savedir="results/cogvideo/t2v_pl/125"
prompt_file="/home/liurt/liurt_data/haoyu/VideoTuna/src/sat/configs/test.txt"
environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

export environs

# run_cmd="$environs python sample_video.py \
# --base configs/cogvideox_2b.yaml configs/inference.yaml --seed 123"

python scripts/inference_cogvideo.py \
    --base /home/liurt/liurt_data/haoyu/VideoTuna/configs/inference/cogvideo_t2v_pl.yaml  \
    --input-file $prompt_file \
    --load "/home/liurt/liurt_data/haoyu/VideoTuna/cogVideo/CogVideoX-2b-sat/transformer/1000/mp_rank_00_model_states.pt" \
    --sampling-num-frames 13 \
    --sampling-fps 8 \
    --force-inference \
    --seed 123
# python scripts/inference_cogvideo.py \
#     --base /home/liurt/liurt_data/haoyu/VideoTuna/configs/inference/cogvideo_t2v_pl.yaml  \
#     --savedir $savedir \
#     --prompt_file $prompt_file \
#     --load "/home/liurt/liurt_data/haoyu/VideoTuna/cogVideo/CogVideoX-2b-sat/transformer/1000/mp_rank_00_model_states.pt" \
#     --bs 1 --height 480 --width 720 \
#     --frames 13 \
#     --savefps 8 \
#     --seed 125

