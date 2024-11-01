ckpt='checkpoints/videocrafter/t2v_v2_512/model.ckpt'
config='configs/001_videocrafter2/vc2_t2v_320x512.yaml'
prompt_file="inputs/t2v/prompts.txt"
current_time=$(date +%Y%m%d%H%M%S)
savedir="results/t2v/$current_time-videocrafter2"

python3 scripts/inference.py \
    --ckpt_path $ckpt \
    --config $config \
    --prompt_file $prompt_file \
    --savedir $savedir \
    --bs 1 --height 320 --width 512 \
    --fps 28 \
    --seed 123

