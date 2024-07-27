ckpt='checkpoints/videocrafter/base_512_v2/model.ckpt'
config='configs/inference/inference_t2v_512_v2.0.yaml'
prompt_file="inputs/t2v/prompts.txt"
res_dir="results/t2v/videocrafter2"

python3 scripts/inference.py \
--ckpt_path $ckpt \
--config $config \
--prompt_file $prompt_file \
--savedir $res_dir \
--n_samples 1 \
--bs 1 --height 320 --width 512 \
--fps 28 \
--seed 123

