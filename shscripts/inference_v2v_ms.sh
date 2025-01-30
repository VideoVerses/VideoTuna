input_dir="inputs/v2v/001"
current_time=$(date +%Y%m%d%H%M%S)
output_dir="results/v2v/$current_time-v2v-modelscope-001"

python3 scripts/inference_v2v_ms.py \
    --input_dir $input_dir --output_dir $output_dir