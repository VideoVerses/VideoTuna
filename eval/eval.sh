export video_path="/project/llmsvgen/share/vbench/opensoraV12-vbench/all_dimension"
export json_path='/project/llmsvgen/share/vbench/opensoraV12-vbench/info.json'
export output_path="/project/llmsvgen/share/vbench/quan_result/opensoraV12-vbench-256/"

# - standard evaluation
python scripts/evaluation.py  \
--output_path $output_path \
--videos_path $video_path \
--map_json_path $json_path \

# calculate the final score after standard evaluation
export result_path="/project/llmsvgen/share/vbench/quan_result/opensoraV12-vbench-256/final_results.json"
python scripts/tabular_score.py \
--result_path $result_path \


# - customed evaluation
python scripts/evaluation.py  \
--output_path $output_path \
--videos_path $path \
--dimension "aesthetic_quality" "dynamic_degree" \
--mode custom_input

echo "Evaluation finished"