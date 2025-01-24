#!/bin/bash

export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export LOG_LEVEL=DEBUG

GPU_IDS="0"

DATA_ROOT="Dataset/video-dataset-disney"
CAPTION_COLUMN="prompt.txt"
VIDEO_COLUMN="videos.txt"
OUTPUT_DIR="results/hunyuan/hunyuan-video-loras/your-experiment-name"

ID_TOKEN="yqyz"

# Model arguments
model_cmd="--model_name hunyuan_video \
  --pretrained_model_name_or_path checkpoints/hunyuan/HunyuanVideo"

# Dataset arguments
dataset_cmd="--data_root $DATA_ROOT \
  --video_column $VIDEO_COLUMN \
  --caption_column $CAPTION_COLUMN \
  --id_token $ID_TOKEN \
  --video_resolution_buckets 17x512x768 49x512x768 61x512x768 \
  --caption_dropout_p 0.05"

# Dataloader arguments
dataloader_cmd="--dataloader_num_workers 0"

# Diffusion arguments
diffusion_cmd=""

# Training arguments
training_cmd="--training_type lora \
  --seed 42 \
  --mixed_precision bf16 \
  --batch_size 1 \
  --train_steps 500 \
  --rank 128 \
  --lora_alpha 128 \
  --target_modules to_q to_k to_v to_out.0 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --checkpointing_steps 100 \
  --checkpointing_limit 5 \
  --enable_slicing \
  --enable_tiling"

# Optimizer arguments
optimizer_cmd="--optimizer adamw \
  --lr 2e-5 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 100 \
  --lr_num_cycles 1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight_decay 1e-4 \
  --epsilon 1e-8 \
  --max_grad_norm 1.0"

# Validation arguments
validation_cmd="--validation_prompts \"$ID_TOKEN A baker carefully cuts a green bell pepper cake on a white plate against a bright yellow background, followed by a strawberry cake with a similar slice of cake being cut before the interior of the bell pepper cake is revealed with the surrounding cake-to-object sequence.@@@49x512x768:::$ID_TOKEN A cake shaped like a Nutella container is carefully sliced, revealing a light interior, amidst a Nutella-themed setup, showcasing deliberate cutting and preserved details for an appetizing dessert presentation on a white base with accompanying jello and cutlery, highlighting culinary skills and creative cake designs.@@@49x512x768:::$ID_TOKEN A cake shaped like a Nutella container is carefully sliced, revealing a light interior, amidst a Nutella-themed setup, showcasing deliberate cutting and preserved details for an appetizing dessert presentation on a white base with accompanying jello and cutlery, highlighting culinary skills and creative cake designs.@@@61x512x768:::$ID_TOKEN A vibrant orange cake disguised as a Nike packaging box sits on a dark surface, meticulous in its detail and design, complete with a white swoosh and 'NIKE' logo. A person's hands, holding a knife, hover over the cake, ready to make a precise cut, amidst a simple and clean background.@@@61x512x768:::$ID_TOKEN A vibrant orange cake disguised as a Nike packaging box sits on a dark surface, meticulous in its detail and design, complete with a white swoosh and 'NIKE' logo. A person's hands, holding a knife, hover over the cake, ready to make a precise cut, amidst a simple and clean background.@@@97x512x768:::$ID_TOKEN A vibrant orange cake disguised as a Nike packaging box sits on a dark surface, meticulous in its detail and design, complete with a white swoosh and 'NIKE' logo. A person's hands, holding a knife, hover over the cake, ready to make a precise cut, amidst a simple and clean background.@@@129x512x768:::$ID_TOKEN A person with gloved hands carefully cuts a cake shaped like a Skittles bottle, beginning with a precise incision at the lid, followed by careful sequential cuts around the neck, eventually detaching the lid from the body, revealing the chocolate interior of the cake while showcasing the layered design's detail.@@@61x512x768:::$ID_TOKEN A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage@@@61x512x768\" \
  --num_validation_videos 1 \
  --validation_steps 100"

# Miscellaneous arguments
miscellaneous_cmd="--tracker_name finetrainers-hunyuan-video \
  --output_dir $OUTPUT_DIR \
  --nccl_timeout 1800 \
  --report_to wandb"

cmd="accelerate launch --config_file configs/006_hunyuanVideo/hunyuan_t2v_lora.yaml --gpu_ids $GPU_IDS ./scripts/train_hunyuan_lora.py \
  $model_cmd \
  $dataset_cmd \
  $dataloader_cmd \
  $diffusion_cmd \
  $training_cmd \
  $optimizer_cmd \
  $validation_cmd \
  $miscellaneous_cmd"

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"