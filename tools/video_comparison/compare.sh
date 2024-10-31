#! /bin/bash

input_dir='inputs/t2v'
save_dir='results/compare14/'

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



#### run videocrafter2 ####
ckpt='checkpoints/videocrafter/t2v_v2_512/model.ckpt'
config='configs/train/000_videocrafter2ft/config.yaml'
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
config=configs/train/002_dynamicrafterft_1024/config.yaml
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



#### run cogvideo—t2v
python scripts/inference_cogVideo_diffusers.py \
  --model_input $prompt_file \
  --model_path checkpoints/cogvideo/CogVideoX-2b \
  --output_path ${save_dir}/t2v/cogvideo-t2v-720x480-8fps \
  --num_inference_steps 50 \
  --guidance_scale 3.5 \
  --num_videos_per_prompt 1 \
  --dtype float16 --seed 123


#### run cogvideo—i2v
python scripts/inference_cogVideo_diffusers.py \
    --generate_type i2v \
    --model_input "inputs/t2v/" \
    --model_path checkpoints/cogvideo/CogVideoX-5b-I2V \
    --output_path ${save_dir}/t2v/cogvideo-i2v-720x480-8fps \
    --num_inference_steps 50 \
    --guidance_scale 3.5 \
    --num_videos_per_prompt 1 \
    --dtype float16

#### run opensora
ckpt="checkpoints/open-sora/t2v_v10/OpenSora-v1-HQ-16x256x256.pth"
config='configs/train/001_opensorav10/config_opensorav10.yaml'
height=256
width=256
fps=8
res_dir="${save_dir}/t2v/opensora-${height}x${width}-${fps}fps"

python3 scripts/inference.py \
    --seed 123 \
    --mode 't2v' \
    --ckpt_path $ckpt \
    --config $config \
    --savedir $res_dir \
    --n_samples 1 \
    --bs 1 --height ${height} --width ${width} \
    --unconditional_guidance_scale 7.0 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --prompt_file $prompt_file \
    --fps ${fps} \
    --frames 16



#### combine video
python3 tools/video_comparison/combine.py --save_dir=$save_dir --input_dir=$input_dir
