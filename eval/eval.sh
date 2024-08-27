export path="/project/llmsvgen/share/vbench/stage5_vbench_all/"
export output_path="/project/llmsvgen/share/vbench/quan_result/test/"

# - standard evaluation
python scripts/evaluation.py  \
--output_path $output_path \
--videos_path $path \

# calculate the final score after standard evaluation
python scripts/tabular_score.py \
--result_path $output_path \



# - customed evaluation
python scripts/evaluation.py  \
--output_path $output_path \
--videos_path $path \
--dimension "aesthetic_quality" "dynamic_degree" \
--mode custom_input

echo "Evaluation finished"