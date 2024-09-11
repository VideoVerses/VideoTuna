import os
import pandas as pd
import random

import numpy as np
from sympy import rad
import torch
from torchvision.datasets.folder import pil_loader

from .video_transforms import TemporalRandomCrop
from .datasets_utils import center_crop_arr, VIDEO_EXTS, IMG_EXTS, read_video, get_transforms_video, get_transforms_image


class DatasetFromCSV(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        csv_path: str
            the path of the csv file. CSV file format:
            ```
            video_path, caption
            path/to/video1, caption1
            path/to/video2, caption2
            ...
            ```

        data_root : str
            the root path of the video data. If the video path in the csv file is a relative path,
            the data_root will be added to the video path.

        transform : callable
            the transform function to process the video data.

        num_frames : int
            the number of frames to sample from the video.

        frame_interval : int
            the interval of the sampled frames.

    """

    def __init__(
        self,
        csv_path,
        train,
        resoluton=(256, 256),
        num_frames=16,
        frame_interval=1,
        data_root=None,
        **kwargs,
    ):
        self.csv_path = csv_path
        # if isinstance(csv_path, str):
        #     csv_path = [csv_path]

        self.resolution = resoluton
        self.video_transform = get_transforms_video(resoluton)
        self.image_transform = get_transforms_image(resoluton)

        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.temporal_sample = TemporalRandomCrop(
            num_frames * frame_interval
        )
        self.frame_limit = num_frames * frame_interval
        self.data_root = data_root

        # for path in csv_path:
        df = pd.read_csv(csv_path)
        self.check_df(df, csv_path)

        self.data_list = []
        for _, row in df.iterrows():
            video_path = row["video_path"]
            caption = row["caption"]
            
            # filter the bad data in advance
            if "frames" in row and "height" in row:
                if not self._is_bad_data(row):
                    data_dict = {"video_path": video_path, "caption": caption}
                else:
                    continue
            else:
                data_dict = {"video_path": video_path, "caption": caption}

            self.data_list.append(data_dict)
        
        if train:
            self.data_list = self.data_list[100:]
            print(f"Training Dataset size: {len(self.data_list)}")
        else:
            self.data_list = self.data_list[:100]
            print(f"Validation Dataset size: {len(self.data_list)}")

        self.safe_data_list = set()

    def getitem(self, index) -> dict:
        data = self.data_list[index]
        path = data["video_path"]
        text = data["caption"]
        if self.data_root is not None:
            path = os.path.join(self.data_root, path)

        if path.split(".")[-1] in VIDEO_EXTS:
            vframes, fps = read_video(path, fps=True)
            length = vframes.shape[0]
            h = vframes.shape[2]
            w = vframes.shape[3]
            if length <= self.frame_limit or h < self.resolution[0] or w < self.resolution[1]:
                raise ValueError(f"Bad video: {path}")

            total_frames = len(vframes)

            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert (
                end_frame_ind - start_frame_ind >= self.num_frames
            ), f"{path} with index {index} has not enough frames. Currnet frames: {len(vframes)}"
            frame_indice = np.linspace(
                start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int
            )

            video = vframes[frame_indice]
            video = self.video_transform(video)  # T C H W
        elif path.split(".")[-1] in IMG_EXTS:
            image = pil_loader(path)
            image = self.image_transform(image)
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
        else:
            raise ValueError(f"Unsupported file type: {path}")

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        output = {
                "video": video,
                "caption": text,
                "fps": fps/self.frame_interval
                }

        return output

    def __getitem__(self, index):
        cnt = 100
        while cnt > 0:  # randomly get a good data, till 100 times
            try:
                data_item = self.getitem(index)
                self.safe_data_list.add(index)
                return data_item
            except (ValueError, AssertionError) as e:
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
    
    def _is_bad_data(self, row) -> bool:
        if row["frames"] <= self.frame_limit or row["height"] < self.resolution[0] or row["width"] < self.resolution[1]:
            return True
    
    @staticmethod
    def check_df(df, df_path):
        if "video_path" not in df.columns:
            raise ValueError(f"The csv file {df_path} must have a column named \'video_path\'.")
        elif "caption" not in df.columns:
            raise ValueError(f"The csv file {df_path} must have a column named \'caption\'.")
