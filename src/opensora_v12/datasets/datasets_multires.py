import os
import random

import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

import decord
from decord import VideoReader, cpu
from einops import rearrange

from src.opensora_v12.registry import DATASETS

from .read_video import read_video
from .utils import VID_EXTENSIONS, get_transforms_image, get_transforms_video, read_file, temporal_random_crop
from .datasets_dir_config import DATA_DIR, SAFE_DATA_DIR, IMG_DATA_DIR, DATA_INFO_DIR

IMG_FPS = 120


def read_video_decord(video_path, get_fps=False):
    decord.bridge.set_bridge('torch')
    video = VideoReader(video_path, ctx=cpu(0))
    video_len = len(video)
    indexes = range(0, video_len)
    vframes = video.get_batch(indexes)
    vframes = rearrange(vframes, 't h w c -> t c h w')
    if get_fps:
        fps = video.get_avg_fps()
        return vframes, fps
    else:
        return vframes


@DATASETS.register_module()
class VideoTextDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        data,
        num_frames=16,
        frame_interval=1,
        image_size=(256, 256),
        transform_name="center",
        video_len_limit=None,
    ):
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }
        self.safe_data_list = []
        self.data_list = []
        for dataset in data:
            print(f'Loading {dataset} data...')
            try:
                data_csv_path = DATA_INFO_DIR[dataset]
            except:
                data_csv_path = IMG_DATA_DIR[dataset]
            df = pd.read_csv(data_csv_path)
            
            c = 0
            for _, row in df.iterrows():
                path = row['path']
                text = row['text']
                fps = int(round(row['fps']))
                num_frames = int(row['frames'])
                height = int(row['height'])
                width = int(row['width'])

                if video_len_limit is not None and num_frames < video_len_limit and num_frames != 1:
                    continue
                
                self.data_list.append([path, text, num_frames, fps, height, width])
                c += 1

            print(f"Loaded {c} data from {dataset}.")
        
        print(f"Loaded {len(self.data_list)} data.")

        self.data = pd.DataFrame(self.data_list, columns=['path', 'text', "num_frames", "fps", "height", "width"])

    def _print_data_number(self):
        num_videos = 0
        num_images = 0
        for path in self.data["path"]:
            if self.get_type(path) == "video":
                num_videos += 1
            else:
                num_images += 1
        print(f"Dataset contains {num_videos} videos and {num_images} images.")

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = sample["path"]
        text = sample["text"]
        file_type = self.get_type(path)

        if file_type == "video":
            # loading
            vframes = read_video(path)

            # Sampling video frames
            video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)

            # transform
            transform = self.transforms["video"]
            video = transform(video)  # T C H W
        else:
            # loading
            image = pil_loader(path)

            # transform
            transform = self.transforms["image"]
            image = transform(image)

            # repeat
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        return {"video": video, "text": text}

    def __getitem__(self, index):
        # for _ in range(10):
        #     try:
        #         return self.getitem(index)
        #     except Exception as e:
        #         # path = self.data.iloc[index]["path"]
        #         print(f"Index {index}: {e}")
        #         index = np.random.randint(len(self))
        # raise RuntimeError("Too many bad data.")
        return self.getitem(index)

    def __len__(self):
        return len(self.data)


@DATASETS.register_module()
class VariableVideoTextDataset(VideoTextDataset):
    def __init__(
        self,
        data,
        num_frames=None,
        frame_interval=1,
        image_size=None,
        transform_name=None,
        video_len_limit=None,
        dummy_text_feature=False,
    ):
        super().__init__(data, num_frames, frame_interval, image_size, transform_name=None, video_len_limit=video_len_limit)
        self.transform_name = transform_name
        self.data["id"] = np.arange(len(self.data))
        self.dummy_text_feature = dummy_text_feature

    def get_data_info(self, index):
        T = self.data.iloc[index]["num_frames"]
        H = self.data.iloc[index]["height"]
        W = self.data.iloc[index]["width"]
        return T, H, W

    def getitem(self, index):
        # a hack to pass in the (time, height, width) info from sampler
        if isinstance(index, str):
            index, num_frames, height, width = [int(val) for val in index.split("-")]

        sample = self.data.iloc[index]
        path = sample["path"]
        text = sample["text"]
        video_length = int(sample["num_frames"])
        fps = int(sample["fps"])
        file_type = self.get_type(path)
        ar = height / width

        if file_type == "video":
            # loading
            try:
                # vframes, vinfo = read_video(path, backend="av")
                vframes = read_video_decord(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")

            # calculate dynamic frame interval if interval is smaller than defaul interval
            if num_frames * self.frame_interval > video_length:
                frame_interval = video_length // num_frames
                fps = int((fps / frame_interval) * self.frame_interval)
            else:
                frame_interval = self.frame_interval
                
            # Sampling video frames.
            video = temporal_random_crop(vframes, num_frames, frame_interval)
            video = video.clone()
            del vframes

            # transform
            transform = get_transforms_video(self.transform_name, (height, width))
            video = transform(video)  # T C H W
        else:
            # loading
            try:
                image = pil_loader(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")

            # transform
            transform = get_transforms_image(self.transform_name, (height, width))
            image = transform(image)

            # repeat
            video = image.unsqueeze(0)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        ret = {
            "video": video,
            "text": text,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": fps,
        }
        if self.dummy_text_feature:
            text_len = 50
            ret["text"] = torch.zeros((1, text_len, 1152))
            ret["mask"] = text_len
        return ret


@DATASETS.register_module()
class BatchFeatureDataset(torch.utils.data.Dataset):
    """
    The dataset is composed of multiple .bin files.
    Each .bin file is a list of batch data (like a buffer). All .bin files have the same length.
    In each training iteration, one batch is fetched from the current buffer.
    Once a buffer is consumed, load another one.
    Avoid loading the same .bin on two difference GPUs, i.e., one .bin is assigned to one GPU only.
    """

    def __init__(self, data_path=None):
        self.path_list = sorted(glob(data_path + "/**/*.bin"))

        self._len_buffer = len(torch.load(self.path_list[0]))
        self._num_buffers = len(self.path_list)
        self.num_samples = self.len_buffer * len(self.path_list)

        self.cur_file_idx = -1
        self.cur_buffer = None

    @property
    def num_buffers(self):
        return self._num_buffers

    @property
    def len_buffer(self):
        return self._len_buffer

    def _load_buffer(self, idx):
        file_idx = idx // self.len_buffer
        if file_idx != self.cur_file_idx:
            self.cur_file_idx = file_idx
            self.cur_buffer = torch.load(self.path_list[file_idx])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        self._load_buffer(idx)

        batch = self.cur_buffer[idx % self.len_buffer]  # dict; keys are {'x', 'fps'} and text related

        ret = {
            "video": batch["x"],
            "text": batch["y"],
            "mask": batch["mask"],
            "fps": batch["fps"],
            "height": batch["height"],
            "width": batch["width"],
            "num_frames": batch["num_frames"],
        }
        return ret