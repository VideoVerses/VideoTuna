export FLUX_SCHNELL='checkpoints/FLUX.1-schnell/flux1-schnell.safetensors'
export FLUX_DEV='checkpoints/FLUX.1-dev/flux1-dev.safetensors'
export AE='checkpoints/FLUX.1-schnell/ae.safetensors'
model=flux-schnell #flux-schnell/flux-dev
python -m src/flux/cli.py \
  --name $model \
  --height 768 --width 1360 \
  --prompt "a girl is dancing" \
  --device cuda \
  --output_dir output
