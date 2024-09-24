python scripts/inference_cogVideo_diffusers.py \
    --generate_type i2v \
    --model_input "inputs/i2v/576x1024" \
    --model_path checkpoints/cogvideo/CogVideoX-5b-I2V \
    --output_path results/cogvideo-test-i2v \
    --num_inference_steps 50 \
    --guidance_scale 3.5 \
    --num_videos_per_prompt 1 \
    --dtype float16

