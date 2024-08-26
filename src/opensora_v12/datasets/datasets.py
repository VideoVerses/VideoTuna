import csv
import cv2
import os
import sys
import pandas as pd
import random
import json
import warnings
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.io import write_video
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

import decord
from decord import VideoReader, cpu
from einops import rearrange
from PIL import ImageFile
from einops import rearrange
from pathlib import Path


from . import video_transforms
from .video_transforms import resize
from .utils import center_crop_arr
from .datasets_dir_config import DATA_DIR, SAFE_DATA_DIR, IMG_DATA_DIR

IMG_FPS = 120
ImageFile.LOAD_TRUNCATED_IMAGES = True
decord.bridge.set_bridge('torch')


def read_video(video_path, return_fps=False):
    video = VideoReader(video_path, ctx=cpu(0))
    video_len = len(video)
    indexes = range(0, video_len)
    vframes = video.get_batch(indexes)
    vframes = rearrange(vframes, 't h w c -> t c h w')
    if return_fps:
        return vframes, video.get_avg_fps()
    else:
        return vframes


def save_video_from_tensor(x, fps, save_path):
    """
    Args:
        x (Tensor): shape [T, C, H, W]
    """
    assert x.ndim == 4
    x = rearrange(x, 't c h w -> t h w c').clamp_(0, 255).to("cpu", torch.uint8)
    # write_video(save_path, x, fps=fps, video_codec="h264")

    # cv2
    x_np = np.array(x)
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (x_np.shape[2], x_np.shape[1]))
    for i in range(x_np.shape[0]):
        data = cv2.cvtColor(x_np[i], cv2.COLOR_BGR2RGB)
        out.write(data)
    out.release()


def video_resize(video, size=720):
    assert video.ndim == 4
    h, w = video.shape[2], video.shape[3]

    if h <= w:
        th = size
        tw = round(size * w / h)
        video = resize(video, (th, tw), "bilinear")
    else:
        tw = size
        th = round(size * h / w)
        video = resize(video, (th, tw), "bilinear")

    return video



def get_transforms_video(resolution=256):
    transform_video = transforms.Compose(
        [
            video_transforms.ToTensorVideo(),  # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(resolution),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    return transform_video


def get_transforms_image(image_size=256):
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    return transform


class DatasetFromCSV(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        csv_path,
        num_frames=16,
        frame_interval=1,
        transform=None,
        root=None,
    ):
        self.csv_path = csv_path

        df = pd.read_csv(self.csv_path)
        self.samples = df.values.tolist()
        random.shuffle(self.samples)

        self.safe_data_list = []
        self.data_list = []
        for sample in self.samples:
            if 'webvid' in self.csv_path:
                second, first, text = sample[0], sample[3], sample[4]
                video_path = f'/project/llmsvgen/share/data/webvid/videos/{first}/{second}.mp4'
                caption = sample[4]
            elif 'panda2m' in self.csv_path:
                video_path = sample[0]
                caption = sample[1]
            data_dict = {'video_path': video_path, 'caption': caption}
            self.data_list.append(data_dict)
        
        # load safe data if the dataset is panda2m
        if 'panda2m' in self.csv_path:
            safe_data_path = '/home/zraoac/Open-Sora/panda2m_safe.csv'
        elif 'webvid' in self.csv_path:
            safe_data_path = '/home/zraoac/Open-Sora/webvid_safe.csv'

        df_safe = pd.read_csv(safe_data_path)
        safe_samples = df_safe.values.tolist()

        for safe_sample in safe_samples:
            video_path = safe_sample[0]
            caption = safe_sample[1]
            data_dict = {'video_path': video_path, 'caption': caption}
            self.safe_data_list.append(data_dict)


        self.is_video = True

        self.transform = transform

        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
        self.frame_limit = num_frames * frame_interval
        self.root = root

    def getitem(self, index):
        data = self.data_list[index]
        path = data['video_path']
        text = data['caption']
        
        if self.is_video:
            if os.path.exists(path):
                try:
                    vframes = read_video(path)

                    length = vframes.shape[0]
                    h = vframes.shape[2]
                    w = vframes.shape[3]
                    if length <= self.frame_limit or h < 256 or w < 256:
                        safe_data = random.choice(self.safe_data_list)
                        path = safe_data['video_path']
                        text = safe_data['caption']
                        vframes = read_video(path)
                except:
                    safe_data = random.choice(self.safe_data_list)
                    path = safe_data['video_path']
                    text = safe_data['caption']
                    vframes = read_video(path)
            else:
                safe_data = random.choice(self.safe_data_list)
                path = safe_data['video_path']
                text = safe_data['caption']
                vframes = read_video(path)
            total_frames = len(vframes)

            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert (end_frame_ind - start_frame_ind >= self.num_frames), f"{path} with index {index} has not enough frames. Currnet frames: {len(vframes)}"
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)

            video = vframes[frame_indice]
            video = self.transform(video)  # T C H W
        else:
            image = pil_loader(path)
            image = self.transform(image)
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        return {"video": video, "text": text}

    def __getitem__(self, index):
        for _ in range(10): # randomly get a good data, till 10 times
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)


class DatasetFromMultiCSV(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        csv_path,
        num_frames=16,
        frame_interval=1,
        transform=None,
        root=None,
    ):
        self.csv_path = csv_path

        self.safe_data_list = []
        self.data_list = []
        # for dataset in ['vript', 'panda2m']:
        for dataset in ['artgrid', 'trailer', 'panda2m-hq']:
            print(f'Loading {dataset} data...')
            data_csv_path = DATA_DIR[dataset]
            df = pd.read_csv(data_csv_path)
            self.samples = df.values.tolist()
            random.shuffle(self.samples)
            for sample in self.samples:
                if dataset == 'webvid':
                    second, first, text = sample[0], sample[3], sample[4]
                    video_path = f'/project/llmsvgen/share/data/webvid/videos/{first}/{second}.mp4'
                    caption = sample[4]
                elif dataset == 'panda2m':
                    video_path = sample[0]
                    caption = sample[1]
                elif dataset == 'vript':
                    video_path = sample[4]
                    caption = sample[2]
                elif dataset == 'artgrid':
                    video_path = sample[0]
                    caption = sample[1]
                elif dataset == 'trailer':
                    video_path = sample[0]
                    caption = sample[1]
                elif dataset == 'panda2m-hq':
                    video_path = sample[0]
                    caption = sample[1]
                data_dict = {'video_path': video_path, 'caption': caption, 'dataset': dataset}
                self.data_list.append(data_dict)
            
            # load safe data if the dataset is panda2m
            safe_data_path = SAFE_DATA_DIR[dataset]

            df_safe = pd.read_csv(safe_data_path)
            safe_samples = df_safe.values.tolist()

            for safe_sample in safe_samples:
                video_path = safe_sample[0]
                caption = safe_sample[1]
                data_dict = {'video_path': video_path, 'caption': caption, 'dataset': dataset}
                self.safe_data_list.append(data_dict)


        self.is_video = True

        self.transform = transform

        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
        self.frame_limit = num_frames * frame_interval
        self.root = root
    
    def load_safe_video(self):
        safe_data = random.choice(self.safe_data_list)
        path = safe_data['video_path']
        text = safe_data['caption']
        vframes = read_video(path)
        return vframes, path, text

    def getitem(self, index):
        data = self.data_list[index]
        path = data['video_path']
        text = data['caption']
        
        if self.is_video:
            if os.path.exists(path):
                try:
                    vframes = read_video(path)

                    length = vframes.shape[0]
                    h = vframes.shape[2]
                    w = vframes.shape[3]
                    if length <= self.frame_limit or h < 256 or w < 256:
                        vframes, path, text = self.load_safe_video()
                except:
                    vframes, path, text = self.load_safe_video()
            else:
                vframes, path, text = self.load_safe_video()
            total_frames = len(vframes)

            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert (end_frame_ind - start_frame_ind >= self.num_frames), f"{path} with index {index} has not enough frames. Currnet frames: {len(vframes)}"
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)

            video = vframes[frame_indice]
            video = self.transform(video)  # T C H W
        else:
            image = pil_loader(path)
            image = self.transform(image)
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        return {"video": video, "text": text}

    def __getitem__(self, index):
        for _ in range(10): # randomly get a good data, till 10 times
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data_list)


class DatasetFromMultiCSVDebug(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path,
        num_frames=16,
        frame_interval=1,
        transform=None,
        root=None,
    ):
        self.csv_path = csv_path

        self.safe_data_list = []
        self.data_list = []

        df = pd.read_csv(self.csv_path)
        self.samples = df.values.tolist()
        random.shuffle(self.samples)

        self.art_save_path = Path('/project/llmsvgen/share/data/artgrid/videos/')
        self.art_info_save_path = Path('/project/llmsvgen/share/data/artgrid/info/')

        c = 0
        for sample in self.samples:
            # if len(self.data_list) > 100:
            #     break
            video_path = sample[0]
            caption = sample[1]
            
            sub_folder = video_path.split('/')[-2]
            video_name = video_path.split('/')[-1]

            if os.path.exists(self.art_save_path / sub_folder / video_name):
                c += 1
                continue

            data_dict = {'video_path': video_path, 'caption': caption}
            self.data_list.append(data_dict)
        print(f'Already have {c} videos')

        self.is_video = True

        self.transform = transform

        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
        self.frame_limit = num_frames * frame_interval
        self.root = root

        # self.art_save_path = Path('/project/llmsvgen/artgrid_tmp/videos/')
        # self.art_info_save_path = Path('/project/llmsvgen/artgrid_tmp/info/')

    
    def load_safe_video(self):
        safe_data = random.choice(self.safe_data_list)
        path = safe_data['video_path']
        text = safe_data['caption']
        vframes = read_video(path)
        return vframes, path, text
    
    def getitem(self, index):
        data = self.data_list[index]
        path = data['video_path']
        text = data['caption']
        
        if self.is_video:
            try:
                name_list = path.split('/')
                video_name = name_list[-1]
                subfolder_list = name_list[-2]

                new_video_path = f'/project/llmsvgen/share/data_tmp/{subfolder_list}/{video_name}'

                vframes, fps = read_video(new_video_path, return_fps=True)

                vframes = video_resize(vframes.to(torch.float32), 720)

                length, _, h, w = vframes.shape

                video_save_path = self.art_save_path / subfolder_list / video_name
                save_video_from_tensor(vframes, fps, str(video_save_path))

                info = {"video_path": str(video_save_path), "text": text, "fps": fps, "height": h, "width": w, "frames": length}
                info_save_path = self.art_info_save_path / subfolder_list / f'{video_name.replace(".mp4", ".json")}'
                with open(info_save_path, 'w') as f:
                    json.dump(info, f)
                
                print(f'Finish video: {path}')
                sys.stdout.flush()
            except:
                return {"video_path": 'NA', "text": 'NA', "fps": 24, "height": 720, "width": 1280, "frames": 100}

            # # Sampling video frames
            # start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            # assert (end_frame_ind - start_frame_ind >= self.num_frames), f"{path} with index {index} has not enough frames. Currnet frames: {len(vframes)}"
            # frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)

            # video = vframes[frame_indice]
            # video = self.transform(video)  # T C H W
        else:
            image = pil_loader(path)
            image = self.transform(image)
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # # TCHW -> CTHW
        # video = video.permute(1, 0, 2, 3)

        return {"video_path": str(video_save_path), "text": text, "fps": fps, "height": h, "width": w, "frames": length}
    
    def __getitem__(self, index):
        return self.getitem(index)
    
    def __len__(self):
        return len(self.data_list)


class DatasetMultiRes(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        csv_path,
        num_frames=16,
        frame_interval=1,
        transform=None,
        root=None,
    ):
        self.csv_path = csv_path

        self.safe_data_list = []
        self.data_list = []
        for dataset in ['vript', 'panda2m']:
            print(f'Loading {dataset} data...')
            data_csv_path = DATA_DIR[dataset]
            df = pd.read_csv(data_csv_path)
            self.samples = df.values.tolist()
            random.shuffle(self.samples)
            for sample in self.samples:
                if dataset == 'webvid':
                    second, first, text = sample[0], sample[3], sample[4]
                    video_path = f'/project/llmsvgen/share/data/webvid/videos/{first}/{second}.mp4'
                    caption = sample[4]
                elif dataset == 'panda2m':
                    video_path = sample[0]
                    caption = sample[1]
                elif dataset == 'vript':
                    video_path = sample[4]
                    caption = sample[2]
                data_dict = {'video_path': video_path, 'caption': caption, 'dataset': dataset}
                self.data_list.append(data_dict)
            
            # load safe data if the dataset is panda2m
            safe_data_path = SAFE_DATA_DIR[dataset]

            df_safe = pd.read_csv(safe_data_path)
            safe_samples = df_safe.values.tolist()

            for safe_sample in safe_samples:
                video_path = safe_sample[0]
                caption = safe_sample[1]
                data_dict = {'video_path': video_path, 'caption': caption, 'dataset': dataset}
                self.safe_data_list.append(data_dict)


        self.is_video = True

        self.transform = transform

        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
        self.frame_limit = num_frames * frame_interval
        self.root = root
    
    def load_safe_video(self):
        safe_data = random.choice(self.safe_data_list)
        path = safe_data['video_path']
        text = safe_data['caption']
        vframes = read_video(path)
        return vframes, path, text

    def getitem(self, index):
        data = self.data_list[index]
        path = data['video_path']
        text = data['caption']
        
        if self.is_video:
            if os.path.exists(path):
                try:
                    vframes = read_video(path)

                    length = vframes.shape[0]
                    h = vframes.shape[2]
                    w = vframes.shape[3]
                    if length <= self.frame_limit or h < 256 or w < 256:
                        vframes, path, text = self.load_safe_video()
                except:
                    vframes, path, text = self.load_safe_video()
            else:
                vframes, path, text = self.load_safe_video()
            total_frames = len(vframes)

            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert (end_frame_ind - start_frame_ind >= self.num_frames), f"{path} with index {index} has not enough frames. Currnet frames: {len(vframes)}"
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)

            video = vframes[frame_indice]
            video = self.transform(video)  # T C H W
        else:
            image = pil_loader(path)
            image = self.transform(image)
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        return {
            "video": video,
            "text": text,
            "num_frames": 16,
            "height": 256,
            "width": 256,
            "ar": 1,
            "fps": 24,
        }

    def __getitem__(self, index):
        for _ in range(10): # randomly get a good data, till 10 times
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data_list)


class ImageDataFromCSV(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        transform=None,
        root=None,
    ):

        self.data_list = []
        dataset = 'journeydb'
        print(f'Loading {dataset} data...')
        data_csv_path = IMG_DATA_DIR[dataset]
        df = pd.read_csv(data_csv_path)
        self.samples = df.values.tolist()
        random.shuffle(self.samples)
        for sample in self.samples:
            img_path = sample[0]
            caption = sample[1]
            data_dict = {'image_path': img_path, 'caption': caption, 'dataset': dataset}
            self.data_list.append(data_dict)

        self.transform = transform
        self.root = root

    def getitem(self, index):
        data = self.data_list[index]
        path = data['image_path']
        text = data['caption']
        
        image = pil_loader(path)
        image = self.transform(image)
        video = image.unsqueeze(0)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        return {"video": video, "text": text}

    def __getitem__(self, index):
        for _ in range(10): # randomly get a good data, till 10 times
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data_list)
