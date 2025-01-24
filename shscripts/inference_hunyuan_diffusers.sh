export TOKENIZERS_PARALLELISM=false

HunyuanCKPTPath="checkpoints/hunyuan/HunyuanVideo"
LoRAPath="results/hunyuan/hunyuan-video-loras/your-experiment-name/checkpoint-500/pytorch_lora_weights.safetensors"
LoRAweight=0.6
Prompt="A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions."
OutputPath="results/hunyuan/output_with_lora.mp4"

python scripts/inference_hunyuan_diffusers.py \
    --ckpt-path $HunyuanCKPTPath \
    --lora-path $LoRAPath \
    --lora-weight $LoRAweight \
    --prompt "$Prompt" \
    --video-size 320 512 \
    --video-frame-length 61 \
    --video-fps 15 \
    --infer-steps 50 \
    --output-path $OutputPath
