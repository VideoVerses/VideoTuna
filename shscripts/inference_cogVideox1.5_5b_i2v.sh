load_transformer="checkpoints/cogvideo/CogVideoX1.5-5B-SAT/transformer_i2v"
input_type="txt"
input_file="inputs/i2v/576x1024/test_prompts.txt"
output_dir="results/i2v/"
base="configs/005_cogvideox1.5/cogvideox1.5_5b.yaml"
image_folder="inputs/i2v/576x1024/"

python scripts/inference_cogVideo_sat_refactor.py \
--load_transformer $load_transformer \
--input_file $input_file \
--output_dir $output_dir \
--base $base    \
--mode_type "i2v"   \
--sampling_num_frames 22    \
--image_folder $image_folder \
--seed 42
