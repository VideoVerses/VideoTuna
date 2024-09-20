video_path="/project/llmsvgen/share/vbench/opensoraV12-vbench/all_dimension"
json_path='/project/llmsvgen/share/vbench/opensoraV12-vbench/info.json'
output_path="/home/yhebm/code-share/VideoTuna/results/eval/"

# - standard evaluation
# will save final_results.json
mkdir -p $output_path
python eval/scripts/evaluation.py  \
    --output_path $output_path \
    --videos_path $video_path \
    --map_json_path $json_path \

# calculate the final score after standard evaluation
# will save scaled_results.json
result_path="$output_path/final_results.json"
python eval/scripts/tabular_score.py \
    --result_path $result_path \

echo "Evaluation finished"
