config=configs/010_sana/sana.yaml

accelerate launch \
  --mixed_precision=bf16 \
  --num_processes=1 \
  --num_machines=1 \
  scripts/train_sana_lora.py \
  --base $config