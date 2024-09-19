ckpt='checkpoints/videocrafter/t2v_v2_512/model.ckpt'
# config='configs/inference/inference_t2v_512_v2.0.yaml'
config='configs/inference/vc2_t2v_512.yaml'
prompt_file="inputs/t2v/prompts.txt"
savedir="results/t2v/videocrafter2"

python3 scripts/inference.py \
--ckpt_path $ckpt \
--config $config \
--prompt_file $prompt_file \
--savedir $savedir \
--bs 1 --height 320 --width 512 \
--fps 28 \
--seed 123

