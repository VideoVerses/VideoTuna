video_path="/project/llmsvgen/share/vbench/opensoraV12-vbench/all_dimension"
json_path='/project/llmsvgen/share/vbench/opensoraV12-vbench/info.json'
output_path="/home/yhebm/code-share/VideoTuna/results/eval-custom/"

# - customed evaluation

mkdir -p $output_path
python eval/scripts/evaluation.py  \
    --output_path $output_path \
    --videos_path $video_path \
    --dimension "aesthetic_quality" "dynamic_degree" \
    --mode custom_input --map_json_path $json_path 

echo "Evaluation finished"
