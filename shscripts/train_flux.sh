export TOKENIZERS_PARALLELISM=false
export CONFIG_PATH="configs/006_flux/config"
export DATACONFIG_PATH="configs/006_flux/multidatabackend"
export CONFIG_BACKEND="json"

accelerate launch \
--mixed_precision="bf16" \
--num_processes="1" \
--num_machines="1" \
scripts/train_flux.py \
--config_path="$CONFIG_PATH.$CONFIG_BACKEND" \
--data_config_path="$DATACONFIG_PATH.$CONFIG_BACKEND" \
