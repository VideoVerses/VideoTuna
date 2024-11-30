#! /bin/bash

input_dir='inputs/t2v'
save_dir='results/compare1/'
seed=42
unified_visualization_height=320
inference_methods="videocrafter2;dynamicrafter;cogvideo—t2v;cogvideo—i2v;opensora;mochi"

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
python tools/video_comparison/check_input.py --input_dir=$input_dir --seed=$seed



################################ videocrafter2 ################################
ckpt='checkpoints/videocrafter/t2v_v2_512/model.ckpt'
config='configs/001_videocrafter2/vc2_t2v_320x512.yaml'
prompt_file="${input_dir}/prompts.txt"
height=320
width=512
fps=28

if [[ $inference_methods == *"videocrafter2"* ]]; then
  python3 scripts/inference.py \
    --ckpt_path $ckpt \
    --config $config \
    --prompt_file $prompt_file \
    --savedir ${save_dir}/t2v/videocrafter2-${width}x${height}-${fps}fps \
    --bs 1 --height 320 --width 512 \
    --fps ${fps} \
    --seed ${seed}
fi


################################ dynamicrafter ################################
ckpt=checkpoints/dynamicrafter/i2v_576x1024/model.ckpt
config=configs/002_dynamicrafter/dc_i2v_1024.yaml
prompt_dir="${input_dir}"
height=576
width=1024
fps=10

if [[ $inference_methods == *"dynamicrafter"* ]]; then
  python3 scripts/inference.py \
    --mode 'i2v' \
    --ckpt_path $ckpt \
    --config $config \
    --prompt_dir $prompt_dir \
    --savedir ${save_dir}/i2v/dynamicrafter-${width}x${height}-${fps}fps \
    --bs 1 --height ${height} --width ${width} \
    --fps ${fps} \
    --seed ${seed}
fi


################################ cogvideo—t2v ################################
if [[ $inference_methods == *"cogvideo—t2v"* ]]; then
  python scripts/inference_cogVideo_diffusers.py \
    --model_input $prompt_file \
    --model_path checkpoints/cogvideo/CogVideoX-2b \
    --output_path ${save_dir}/t2v/cogvideo-t2v-720x480-8fps \
    --num_inference_steps 50 \
    --guidance_scale 3.5 \
    --num_videos_per_prompt 1 \
    --dtype float16 --seed ${seed}
fi


################################ cogvideo—i2v ################################
if [[ $inference_methods == *"cogvideo—i2v"* ]]; then
  python scripts/inference_cogVideo_diffusers.py \
      --generate_type i2v \
      --model_input ${input_dir} \
      --model_path checkpoints/cogvideo/CogVideoX-5b-I2V \
      --output_path ${save_dir}/i2v/cogvideo-i2v-720x480-8fps \
      --num_inference_steps 50 \
      --guidance_scale 3.5 \
      --num_videos_per_prompt 1 \
      --dtype float16 --seed ${seed}
fi

################################ opensora ################################
ckpt="checkpoints/open-sora/t2v_v10/OpenSora-v1-HQ-16x256x256.pth"
config='configs/003_opensora/opensorav10_256x256.yaml'
height=256
width=256
fps=8
res_dir="${save_dir}/t2v/opensora-${width}x${height}-${fps}fps"

if [[ $inference_methods == *"opensora"* ]]; then
  python3 scripts/inference.py \
      --seed ${seed} \
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
fi

################################ mochi ################################
if [[ $inference_methods == *"mochi"* ]]; then
  ckpt='genmo/mochi-1-preview'
  prompt_file="${input_dir}/prompts.txt"
  height=480
  width=848
  savedir="${save_dir}/t2v/mochi-${width}x${height}-28fps"

  python3 scripts/inference_mochi.py \
      --ckpt_path $ckpt \
      --prompt_file $prompt_file \
      --savedir $savedir \
      --bs 1 --height $height --width $width \
      --fps 28 \
      --seed ${seed}
fi



#### combine video
python3 tools/video_comparison/combine.py --save_dir=$save_dir --input_dir=$input_dir --unified_height=$unified_visualization_height
