ckpt='checkpoints/videocrafter/i2v_v1_512/model.ckpt'
config='configs/000_videocrafter/vc1_i2v_512.yaml'
prompt_dir="inputs/i2v/576x1024"
savedir="results/i2v/vc1-i2v-320x512"

python3 scripts/inference.py \
--mode 'i2v' \
--ckpt_path $ckpt \
--config $config \
--prompt_dir $prompt_dir \
--savedir $savedir \
--bs 1 --height 320 --width 512 \
--fps 8 \
--seed 123
