import json
import os
import os.path as osp
from pathlib import Path

import cv2
import fire
import pandas as pd
from tqdm import tqdm


def read_video_meta(video_path):
    # Video fps
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # The number of frames
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Height
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Width
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    return {
        "fps": fps,
        "frames": frames,
        "height": height,
        "width": width,
    }


def get_video_data(video_root):
    video_root = Path(video_root)
    video_dict = {}
    for sub_path in video_root.iterdir():
        for sub_sub_path in sub_path.iterdir():
            try:
                if len(os.listdir(sub_sub_path)) == 0:
                    continue

                with open(
                    sub_sub_path / f"{sub_sub_path.name}_cut_meta.json", "r"
                ) as f:
                    video_meta = json.load(f)

                for clip_meta in video_meta["clips"]:
                    video_path = sub_sub_path / clip_meta["clip_id"]
                    meta = {"path": str(video_path.relative_to(video_root))}
                    meta.update(read_video_meta(video_path))
                    video_dict[osp.splitext(clip_meta["clip_id"])[0]] = meta

            except Exception as e:
                import traceback

                traceback.print_exc()

    return video_dict


def main(input_path, output_path, video_root):
    with open(input_path, "r") as jsonl_file:
        lines = jsonl_file.readlines()

    video_dict = get_video_data(video_root)
    data_list = []

    for i, line in tqdm(enumerate(lines)):
        data = json.loads(line)  # parse json file

        clip_id = data.get("clip_id")
        video_meta = video_dict.get(clip_id, None)
        if video_meta is None:
            continue

        # concat captions
        caption_data = data["caption"]
        caption = ""
        for caption_keys in caption_data.keys():
            caption_id = caption_keys
            caption_text = caption_data[caption_keys]
            if not caption_text.endswith("."):
                caption_text += "."
            caption += caption_text + " "
        video_meta["caption"] = caption
        data_list.append(video_meta)

    df = pd.DataFrame(
        data_list, columns=["path", "caption", "fps", "frames", "height", "width"]
    )

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # python vript_anno_converter.py --input_path {ROOT}/Vript/vript_captions/vript_short_videos_captions.jsonl --output_path ./test.csv --video_root  {ROOT}/Vript/vript_short_videos_clips
    fire.Fire(main)
