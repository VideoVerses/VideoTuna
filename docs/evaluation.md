## Installation
If you have installed the environment for the model training and inference, you can simply install some extra packages for evaluation.
```shell
pip install -r eval/requirements_vbench.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
If you encounter errors during installing the [detectron2](https://github.com/facebookresearch/detectron2), you can check [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) for detailed suggestions.

## Usage
1. Prepare samples and a json file.
    Firstly, if you already have video samples, please export a json file for mapping the video file name to prompt. The format is as follows:
    ```json
    {
        "sample1.mp4": "sample1's prompt",
        "sample2.mp4": "sample2's prompt",
        ...
    }
    ```

    For the standard vbench evaluation, you have to do inference on `all_dimensions.txt`.

2. Evaluation
(1) Standard evaluation

    Run the following command:
    ```shell
    python eval/scripts/evaluation.py  \
        --output_path $output_path \
        --videos_path $video_path \
        --map_json_path $json_path
    ```
    The final score of all dimensions are saved in the file `final_results.json`. If you want to submit your result to the VBench Leaderboard, you can zip the files `results_eval_results.json` and `results_full_info.json` and upload it to the [Leaderboard](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard).

    Besides, you also can caluate the ***overall score***, ***quality score*** and ***sementic score*** in the VBench Leaderboard by yourself:
    ```shell
    python eval/scripts/tabular_score.py \
        --result_path $result_json_path
    ```
    The result will be saved in the file `scaled_results.json`.

    (2) Customized evaluation

    If you want to evaluate the generation performance on your own prompts, you can choose the custom mode.
    Note that Vbench only support the following dimensions for the custom mode:
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
    You can run the following command to perform the customized evaluation:
    ```shell
    python eval/scripts/evaluation.py  \
        --output_path $output_path \
        --videos_path $video_path \
        --map_json_path $json_path \
        --dimension $dim1 $dim2 ... \
        --mode custom_input
    ```
    The final score of each dimension is saved in the file `final_results.json`.
