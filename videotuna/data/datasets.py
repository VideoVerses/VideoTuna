import os
import sys

sys.path.append(os.getcwd())
import copy
import random
from typing import Dict, List, Union

import pandas as pd
import torch
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import Compose

from videotuna.data.datasets_utils import (
    is_image,
    is_video,
    read_image_meta,
    read_video,
    read_video_meta,
)
from videotuna.data.transforms import (
    CheckVideo,
    get_transforms_image,
    get_transforms_video,
)


class DatasetFromCSV(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        csv_path: str or list
            the path of the csv file. CSV file format:
            ```
            path, caption
            path/to/video1, caption1
            path/to/video2, caption2
            ...
            ```
            or
            ```
            path, caption, fps, frames, height, width
            path/to/video1, caption1, 30, 100, 512, 512
            path/to/video2, caption2, 30, 50, 1080, 512
            ...
            ```

        data_root : str or list
            the root path of the data item. If the path in the csv file is a relative path,
            the data_root will be added to the file path.

        transform : callable
            the transform function to process the video/image data.

        num_frames : int
            the number of frames to sample from the video.

        frame_interval : int
            the interval of the sampled frames.

        train : bool
            if True, the dataset is for training. Otherwise, the dataset is for validation.

        split_val : bool
            if True, split the dataset into training and validation dataset.

    """

    def __init__(
        self,
        csv_path: Union[str, List[str]],
        data_root: Union[str, List[str], None] = None,
        transform: Union[Dict[str, Compose], None] = None,
        height: int = 256,
        width: int = 256,
        num_frames: int = 16,
        frame_interval: int = 1,
        use_multi_res: bool = False,
        train: bool = True,
        split_val: bool = False,
        image_to_video: bool = False,
        **kwargs,
    ):
        self.csv_path = csv_path
        if isinstance(csv_path, str):
            csv_path = [csv_path]
        if data_root is None or isinstance(data_root, str):
            data_root = [data_root]

        if len(data_root) == 1:
            data_root = data_root * len(csv_path)

        assert len(csv_path) == len(
            data_root
        ), "The number of csv files and data root should be the same."

        if transform is None:
            transform = dict(
                video=get_transforms_video((height, width), num_frames, frame_interval),
                image=get_transforms_image((height, width), num_frames),
            )

        assert (
            "video" in transform or "image" in transform
        ), "The transform should contain 'video' or 'image'."
        self.transform = transform
        self.height = height
        self.width = width
        self.resolution = (height, width)
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.frame_limit = num_frames * frame_interval
        self.data_root = data_root
        self.use_multi_res = use_multi_res
        self.train = train
        self.split_val = split_val
        self.safe_data_list = set()
        self.image_to_video = image_to_video
        self.check_video = CheckVideo(self.resolution, frame_interval, num_frames)

        self.load_annotations(csv_path, data_root)

        if split_val:
            if self.train:
                self.data_list = self.data_list[
                    min(100, int(len(self.data_list) * 0.2)) :
                ]
                print(f"Training Dataset size: {len(self.data_list)}")
            else:
                self.data_list = self.data_list[
                    : min(100, int(len(self.data_list) * 0.2))
                ]
                print(f"Validation Dataset size: {len(self.data_list)}")

    def load_annotations(self, csv_path, data_root):
        self.data_list = []
        for i, path in enumerate(csv_path):
            df = pd.read_csv(path)
            self.check_df(df, path)
            for _, row in df.iterrows():
                video_path = row.get(
                    "path", row.get("video_path", row.get("image_path"))
                )
                caption = row["caption"]

                if not self._is_valid_data(row):
                    continue

                if data_root[i]:
                    video_path = os.path.join(data_root[i], video_path)
                data_dict = {"path": video_path, "caption": caption}
                data_dict["fps"] = (
                    row.get("fps") / self.frame_interval
                    if row.get("fps", None)
                    else None
                )
                if self.use_multi_res:
                    data_dict["height"] = row.get("height", None)
                    data_dict["width"] = row.get("width", None)

                self.data_list.append(data_dict)

    def getitem(self, index):
        data = copy.deepcopy(self.data_list[index])
        path = data.pop("path")
        if is_video(path):
            video = read_video(path)
            video = self.check_video(
                video, index
            )  # filter the video with unsatisfied resolution and frames
            video = self.transform["video"](video)
        elif is_image(path):
            video = pil_loader(path)
            video = self.transform["image"](video)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        data["video"] = video
        if is_video(path) and not (
            data.get("width", None)
            and data.get("height", None)
            and data.get("fps", None)
        ):
            if self.use_multi_res or not data.get("fps", None):
                file_meta = read_video_meta(path)
                if self.use_multi_res:
                    data["height"] = file_meta["height"]
                    data["width"] = file_meta["width"]

                data["fps"] = file_meta["fps"] / self.frame_interval

        if is_image(path):
            if self.use_multi_res and not (
                data.get("height", None) and data.get("width", None)
            ):
                file_meta = read_image_meta(path)
                data["height"] = file_meta["height"]
                data["width"] = file_meta["width"]
            # NOTE: for image, the fps is set to 0
            data["fps"] = 0

        if "frames" in data:
            _ = data.pop("frames")

        if self.image_to_video:
            data["image"] = data["video"][:, :1, :, :].clone()  # CTHW (3，1，H, W)
        return data

    def __getitem__(self, index):
        cnt = 100
        while cnt > 0:  # randomly get a good data, till 100 times
            try:
                index = index % len(self)
                data_item = self.getitem(index)
                self.safe_data_list.add(index)
                return data_item
            except (ValueError, AssertionError):
                import traceback

                traceback.print_exc()

                index = (
                    random.choice(list(self.safe_data_list))
                    if len(self.safe_data_list) > 0
                    else random.randint(0, len(self))
                )
                cnt -= 1

        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data_list)

    def _is_valid_data(self, row) -> bool:
        if (
            row.get("height", None) is None
            or row.get("width", None) is None
            or row.get("frames", None) is None
        ):
            # if the video meta is not provided,
            # the video will be loaded to get the meta in `transforms`.
            return True

        if (
            row["frames"] <= self.frame_limit
            or row["height"] < self.resolution[0]
            or row["width"] < self.resolution[1]
        ):
            return False

        return True

    @staticmethod
    def check_df(df, df_path):
        if (
            "path" not in df.columns
            and "video_path" not in df.columns
            and "image_path" not in df.columns
        ):
            raise ValueError(f"The csv file {df_path} must have a column named 'path'.")
        elif "caption" not in df.columns:
            raise ValueError(
                f"The csv file {df_path} must have a column named 'caption'."
            )


if __name__ == "__main__":
    csv_path = "temp/apply_lipstick.csv"
    dataset = DatasetFromCSV(
        csv_path,
        train=True,
        split_val=True,
        height=480,
        width=720,
        num_frames=49,
        image_to_video=True,
    )
    data = dataset[0]
