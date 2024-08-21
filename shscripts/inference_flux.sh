export FLUX_SCHNELL='/disk2/pengjun/flux/FLUX.1-schnell/flux1-schnell.safetensors'
export FLUX_DEV='/disk2/pengjun/flux/FLUX.1-dev/flux1-dev.safetensors'
export AE='/disk2/pengjun/flux/FLUX.1-schnell/ae.safetensors'
model=flux-schnell #flux-schnell/flux-dev
python -m flux/cli.py \
  --name $model \
  --height 768 --width 1360 \
  --prompt "a girl is dancing" \
  --device cuda \
  --output_dir output
