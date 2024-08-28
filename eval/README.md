## Installation
If you have installed the environment for the model training and inference, you can simply install some extra packages for evaluation.
```shell
pip install -r requirements_vbench.txt
```
If you encounter some error during installing the [detectron2](https://github.com/facebookresearch/detectron2), you can check [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## Usage
Firstly, a json file for mapping the video name to prompt is necessary. Its format is as follows:
```json
{
    "sample1.mp4": "sample1's prompt",
    "sample2.mp4": "sample2's prompt",
    ...
}
```

**standard evaluation**

For the standard vbench evaluation, you have to do inference on `all_dimensions.txt`. Then you can run:
```shell
python scripts/evaluation.py  \
--output_path $output_path \
--videos_path $video_path \
--map_json_path $json_path \
```
After this, you can caluate the final score including *overall score*, *quality score* and *sementic score* by running:
```shell
python scripts/tabular_score.py \
--result_path $result_json_path \
```

**costomized evaluation**

If you want to evaluate the pperformance on your own prompts, you can choose the custom mode. For the custom mode, vbench only support the following dimensions:
```python
dimensions = [
    # Quality Score
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_flickering",
    # Semantic Score
    "temporal_style",
    "overall_consistency",
    "human_action",
]
```
you can run:
```shell
python scripts/evaluation.py  \
--output_path $output_path \
--videos_path $video_path \
--map_json_path $json_path \
--dimension $dim1 $dim2 ... \
--mode custom_input
```
