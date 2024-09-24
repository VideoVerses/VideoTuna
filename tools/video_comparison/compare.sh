#! /bin/bash

input_dir='inputs/t2v'
save_dir='results/compare2/'

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

# run check_input.py, will create images using t2i if there is no images in the input directory
python tools/video_comparison/check_input.py --input_dir=$input_dir



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
python scripts/inference_cogVideo_diffusers.py \
  --model_input $prompt_file \
  --model_path checkpoints/cogvideo/CogVideoX-2b \
  --output_path ${save_dir}/t2v/cogvideo \
  --num_inference_steps 50 \
  --guidance_scale 3.5 \
  --num_videos_per_prompt 1 \
  --dtype float16 --seed 123



#### combine video
python3 tools/video_comparison/combine.py --save_dir=$save_dir --input_dir=$input_dir
