#! /bin/bash

input_dir='/path/to/your/input_dir'
save_dir='/path/to/your/save_dir'

#### check input ####
# Check if the directory exists
if [ ! -d "$input_dir" ]; then
  echo "The input should be a directory, exiting..."
  exit 1  # Exit the script with an error code
fi

# Check if the prompts.txt file exists within the directory
if [ ! -f "${input_dir}/prompts.txt" ]; then
  echo "The file prompts.txt does not exist in the directory, exiting..."
  exit 1  # Exit the script with an error code
fi
python check_input.py --input_dir=$input_dir



#### run videocrafter ####
ckpt='checkpoints/videocrafter/base_512_v2/model.ckpt'
config='configs/inference/vc2_t2v_512.yaml'
prompt_file="${input_dir}/prompts.txt"
height=320
width=512
fps=28

python3 scripts/inference.py \
--ckpt_path $ckpt \
--config $config \
--prompt_file $prompt_file \
--savedir ${save_dir}/t2v/videocrafter2-${height}x${width}-${fps}fps \
--bs 1 --height 320 --width 512 \
--fps ${fps} \
--seed 123



#### run dynamicrafter ####
ckpt=checkpoints/dynamicrafter/i2v_576x1024/model.ckpt
config=configs/inference/dc_i2v_1024.yaml
prompt_dir="${input_dir}"
height=576
width=1024
fps=10

python3 scripts/inference.py \
--mode 'i2v' \
--ckpt_path $ckpt \
--config $config \
--prompt_dir $prompt_dir \
--savedir ${save_dir}/i2v/dynamicrafter-${height}x${width}-${fps}fps \
--bs 1 --height ${height} --width ${width} \
--fps ${fps} \
--seed 123



#### run cogvideo
prompt_file="${input_dir}/prompts.txt"
environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"
height=480
width=720
fps=8
savedir=${save_dir}/t2v/cogvideo-${height}x${width}-${fps}fps
export environs

ckpt_path="cogVideo/CogVideoX-2b-sat/transformer/1000/mp_rank_00_model_states.pt"
# ckpt_path="results/cogvideo_train/20240913160358_correctness/checkpoints/epoch=0005-step=000600.ckpt"
python scripts/inference_cogvideo.py \
    --config "configs/train/005_cogvideoxft/config.yaml" \
    --savedir $savedir \
    --prompt_file $prompt_file \
    --ckpt_path $ckpt_path  \
    --bs 1 --height ${height} --width ${width} \
    --frames 13 \
    --savefps ${fps} \
    --seed 123


#### combine video
python3 combine.py --save_dir=$save_dir --input_dir=$input_dir
