ckpt='checkpoints/mochi-1-preview'
prompt_file="inputs/t2v/prompts.txt"
savedir="results/t2v/mochi2"
height=480
width=848

python3 scripts/inference_mochi.py \
    --ckpt_path $ckpt \
    --prompt_file $prompt_file \
    --savedir $savedir \
    --bs 1 --height $height --width $width \
    --fps 28 \
    --seed 124

