python scripts/inference_cogVideo_diffusers.py \
--prompt "A cat playing with a ball" \
--model_path checkpoints/cogvideo/CogVideoX-2b \
--output_path output.mp4 \
--num_inference_steps 50 \
--guidance_scale 3.5 \
--num_videos_per_prompt 1 \
--dtype float16
