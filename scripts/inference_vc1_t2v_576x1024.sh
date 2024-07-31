ckpt=checkpoints/dynamicrafter/i2v_576x1024/model.ckpt
config=configs/inference/dc_i2v_1024.yaml
prompt_file=inputs/i2v/576x1024
res_dir="results/t2v/videocrafter1-576x1024"

python3 scripts/inference.py \
--ckpt_path $ckpt \
--config $config \
--prompt_file $prompt_file \
--savedir $res_dir \
--bs 1 --height 576 --width 1024 \
--fps 28 \
--seed 123

