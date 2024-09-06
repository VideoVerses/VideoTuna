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

from src.opensora.registry import DATASETS

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


class VideoTextDatasetNew(torch.utils.data.Dataset):
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
        transform_name="resize_crop",
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
                if height < image_size[0] or width < image_size[1]:
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
        video_length = int(sample["num_frames"])
        fps = int(sample["fps"])
        height = self.image_size[0]
        width = self.image_size[1]
        ar = height / width
        num_frames = self.num_frames

        file_type = self.get_type(path)
        
        if file_type == "video":
            # loading
            vframes = read_video_decord(path)

            # calculate dynamic frame interval if interval is smaller than defaul interval
            if num_frames * self.frame_interval > video_length:
                frame_interval = video_length // num_frames
                fps = int((fps / frame_interval) * self.frame_interval)
            else:
                frame_interval = self.frame_interval

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

        return ret

    def __getitem__(self, index):
        # print(index)
        # for _ in range(10):
        #     try:
        #         return self.getitem(index)
        #     except Exception as e:
        #         path = self.data.iloc[index]["path"]
        #         print(f"Path {path}: {e}")
        #         index = np.random.randint(len(self))
        # raise RuntimeError("Too many bad data.")
        return self.getitem(index)

    def __len__(self):
        return len(self.data)


class VideoTextDatasetLongCap(VideoTextDatasetNew):
    def __init__(
        self,
        data,
        num_frames=16,
        frame_interval=1,
        image_size=(256, 256),
        transform_name="resize_crop",
        video_len_limit=None,
        long_caption_ratio=0.8,
    ):
        self.long_caption_ratio = long_caption_ratio
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
                if 'text_long' in row:
                    if not pd.isna(row['text_long']):
                        text = row['text_long'] if random.random() < long_caption_ratio else row['text']
                    else:
                        text = row['text']
                else:
                    text = row['text']
                path = row['path']
                fps = int(round(row['fps']))
                num_frames = int(row['frames'])
                height = int(row['height'])
                width = int(row['width'])

                if video_len_limit is not None and num_frames < video_len_limit and num_frames != 1:
                    continue
                if height < image_size[0] or width < image_size[1]:
                    continue
                
                self.data_list.append([path, text, num_frames, fps, height, width])
                c += 1

            print(f"Loaded {c} data from {dataset}.")
        
        print(f"Loaded {len(self.data_list)} data.")

        self.data = pd.DataFrame(self.data_list, columns=['path', 'text', "num_frames", "fps", "height", "width"])
    

class VideoTextDatasetLongCapCont(VideoTextDatasetNew):
    def __init__(
        self,
        data,
        num_frames=16,
        frame_interval=1,
        image_size=(256, 256),
        transform_name="resize_crop",
        video_len_limit=None,
        long_caption_ratio=0.8,
    ):
        self.long_caption_ratio = long_caption_ratio
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }
        self.safe_data_list = []
        self.data_list = []
        self.pre_data_list = []
        for dataset in data:
            print(f'Loading {dataset} data...')
            try:
                data_csv_path = DATA_INFO_DIR[dataset]
            except:
                data_csv_path = IMG_DATA_DIR[dataset]
            df = pd.read_csv(data_csv_path)
            
            c = 0
            for _, row in df.iterrows():
                if 'text_long' in row:
                    if not pd.isna(row['text_long']):
                        text = row['text_long'] if random.random() < long_caption_ratio else row['text']
                    else:
                        text = row['text']
                else:
                    text = row['text']
                path = row['path']
                fps = int(round(row['fps']))
                num_frames = int(row['frames'])
                height = int(row['height'])
                width = int(row['width'])

                if video_len_limit is not None and num_frames < video_len_limit and num_frames != 1:
                    continue
                if height < image_size[0] or width < image_size[1]:
                    continue
                
                if 'eat' in dataset:
                    self.pre_data_list.append([path, text, num_frames, fps, height, width])
                else:
                    self.data_list.append([path, text, num_frames, fps, height, width])
                c += 1

            print(f"Loaded {c} data from {dataset}.")
        
        print(f"Loaded {len(self.pre_data_list)} previous finetune data.")
        
        print(f"Loaded {len(self.data_list)} data.")

        if int(len(self.data_list) * 0.3) < len(self.pre_data_list):
            sample_data = random.sample(self.pre_data_list, int(len(self.data_list) * 0.3))
        else:
            sample_data = self.pre_data_list
        
        self.data_list = self.data_list + sample_data

        self.data = pd.DataFrame(self.data_list, columns=['path', 'text', "num_frames", "fps", "height", "width"])
        

class VaeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data='/project/llmsvgen/share/data_yazhou/UHD/sample_200_original',
        num_frames=16,
        frame_interval=1,
        image_size=(256, 256),
        transform_name="resize_crop",
        video_len_limit=None,
    ):
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = get_transforms_video(transform_name, image_size)
        self.video_len_limit = video_len_limit
        self.data_list = []
        for root, _, files in os.walk(data):
            for file in files:
                if file.endswith('.mp4'):
                    path = os.path.join(root, file)
                    self.data_list.append(path)
        print(f"Loaded {len(self.data_list)} data.")
    
    def __getitem__(self, index):
        path = self.data_list[index]
        vframes = read_video_decord(path)
        if self.video_len_limit is not None and len(vframes) < self.video_len_limit:
            return {'video': torch.Tensor([1])}
        video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)
        video = self.transforms(video)
        video = video.permute(1, 0, 2, 3)
        return {'video': video}

    def __len__(self):
        return len(self.data_list)