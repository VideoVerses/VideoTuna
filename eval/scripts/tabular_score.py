import argparse
import json
import os
import shutil
from pathlib import Path

SEMANTIC_WEIGHT = 1
QUALITY_WEIGHT = 4

QUALITY_LIST = [
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "dynamic degree",
]

SEMANTIC_LIST = [
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency",
]

NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0},
    "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364},
    "overall consistency": {"Min": 0.0, "Max": 0.364},
}

DIM_WEIGHT = {
    "subject consistency": 1,
    "background consistency": 1,
    "temporal flickering": 1,
    "motion smoothness": 1,
    "aesthetic quality": 1,
    "imaging quality": 1,
    "dynamic degree": 0.5,
    "object class": 1,
    "multiple objects": 1,
    "human action": 1,
    "color": 1,
    "spatial relationship": 1,
    "scene": 1,
    "appearance style": 1,
    "temporal style": 1,
    "overall consistency": 1,
}

ordered_scaled_res = [
    "total score",
    "quality score",
    "semantic score",
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "dynamic degree",
    "aesthetic quality",
    "imaging quality",
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency",
]


def main(args):
    ori_result_path = args.result_path
    output_dir = os.path.dirname(ori_result_path)
    with open(ori_result_path, "r") as f:
        full_results = json.load(f)

    scaled_results = {}
    dims = set()
    for key, val in full_results.items():
        dim = key.replace("_", " ") if "_" in key else key
        scaled_score = (float(val) - NORMALIZE_DIC[dim]["Min"]) / (
            NORMALIZE_DIC[dim]["Max"] - NORMALIZE_DIC[dim]["Min"]
        )
        scaled_score *= DIM_WEIGHT[dim]
        scaled_results[dim] = scaled_score
        dims.add(dim)

    quality_score = sum([scaled_results[i] for i in QUALITY_LIST]) / sum(
        [DIM_WEIGHT[i] for i in QUALITY_LIST]
    )
    semantic_score = sum([scaled_results[i] for i in SEMANTIC_LIST]) / sum(
        [DIM_WEIGHT[i] for i in SEMANTIC_LIST]
    )
    scaled_results["quality score"] = quality_score
    scaled_results["semantic score"] = semantic_score
    scaled_results["total score"] = (
        quality_score * QUALITY_WEIGHT + semantic_score * SEMANTIC_WEIGHT
    ) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)

    formated_scaled_results = {"items": []}
    for key in ordered_scaled_res:
        formated_score = format(scaled_results[key] * 100, ".2f") + "%"
        formated_scaled_results["items"].append({key: formated_score})

    # all_results.json is the same with final_results.json
    # output_file_path = os.path.join(output_dir, "all_results.json")
    # with open(output_file_path, "w") as outfile:
    #     json.dump(full_results, outfile, indent=4, sort_keys=True)
    # print(f"results saved to: {output_file_path}")

    scaled_file_path = os.path.join(output_dir, "scaled_results.json")
    with open(scaled_file_path, "w") as outfile:
        json.dump(formated_scaled_results, outfile, indent=4, sort_keys=True)
    print(f"results saved to: {scaled_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VBench", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="The path of result json file",
    )
    args = parser.parse_args()
    main(args)
