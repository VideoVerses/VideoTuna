
# sample a single video
python scripts/inference_cogVideo_diffusers.py \
    --model_input "A cat playing with a ball" \
    --model_path checkpoints/cogvideo/CogVideoX-2b \
    --output_path results/output.mp4 \
    --num_inference_steps 50 \
    --guidance_scale 3.5 \
    --num_videos_per_prompt 1 \
    --dtype float16

# sample multiple videos
# python scripts/inference_cogVideo_diffusers.py \
    # --model_input "inputs/t2v/prompts.txt" \
    # --model_path checkpoints/cogvideo/CogVideoX-2b \
    # --output_path results/cogvideo-test \
    # --num_inference_steps 50 \
    # --guidance_scale 3.5 \
    # --num_videos_per_prompt 1 \
    # --dtype float16

