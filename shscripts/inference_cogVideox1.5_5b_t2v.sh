load_transformer="checkpoints/cogvideo/CogVideoX1.5-5B-SAT/transformer_t2v"
input_type="txt"
input_file="inputs/t2v/prompts.txt"
output_dir="results/t2v/"
base="configs/005_cogvideox1.5/cogvideox1.5_5b.yaml"

python scripts/inference_cogVideo_sat_refactor.py \
--load_transformer $load_transformer \
--input_file $input_file \
--output_dir $output_dir \
--base $base    \
--mode_type "t2v"   \
--sampling_num_frames 22    \
