import csv
import os
import pandas as pd
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

import decord
from decord import VideoReader, cpu
from einops import rearrange

from . import video_transforms
from .utils import center_crop_arr


class DatasetFromMultiCSV_Backup(torch.utils.data.Dataset):
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
        # self.csv_path = {
        #     'webvid': '/project/llmsvgen/share/data/webvid/part3.csv',
        #     'panda2m': '/home/zraoac/Open-Sora/panda2m.csv',
        #     'vript': '/project/llmsvgen/share/data_yazhou/lzy/vript.csv'
        # }
        self.csv_path = {
            'webvid': '/project/llmsvgen/share/data/webvid/part3.csv',
            'vript': '/project/llmsvgen/share/data_yazhou/lzy/vript.csv'
        }


        self.safe_data_list = []
        self.data_list = []
        for dataset in ['webvid', 'vript']:
            print(f'Loading {dataset} data...')
            data_csv_path = self.csv_path[dataset]
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
            if dataset == 'panda2m':
                safe_data_path = '/home/zraoac/Open-Sora/panda2m_safe.csv'
            elif dataset == 'webvid':
                safe_data_path = '/home/zraoac/Open-Sora/webvid_safe.csv'

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

    def getitem(self, index):
        data = self.data_list[index]
        path = data['video_path']
        text = data['caption']
        
        if self.is_video:
            if os.path.exists(path):
                try:
                    vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")

                    length = vframes.shape[0]
                    h = vframes.shape[2]
                    w = vframes.shape[3]
                    if length <= self.frame_limit or h < 256 or w < 256:
                        safe_data = random.choice(self.safe_data_list)
                        path = safe_data['video_path']
                        text = safe_data['caption']
                        vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
                except:
                    safe_data = random.choice(self.safe_data_list)
                    path = safe_data['video_path']
                    text = safe_data['caption']
                    vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
            else:
                safe_data = random.choice(self.safe_data_list)
                path = safe_data['video_path']
                text = safe_data['caption']
                vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
            total_frames = len(vframes)

            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert (
                end_frame_ind - start_frame_ind >= self.num_frames
            ), f"{path} with index {index} has not enough frames. Currnet frames: {len(vframes)}"
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


class DatasetFromCSV_Backup(torch.utils.data.Dataset):
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
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            self.samples = list(reader)

        ext = self.samples[0][0].split(".")[-1]
        if ext.lower() in ("mp4", "avi", "mov", "mkv"): # supported videos
            self.is_video = True
        else:
            assert f".{ext.lower()}" in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            self.is_video = False

        self.transform = transform

        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
        self.root = root

    def getitem(self, index):
        sample = self.samples[index]
        path = sample[0]
        if self.root:
            path = os.path.join(self.root, path)
        text = sample[1]

        # print(text)

        if self.is_video:
            vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
            total_frames = len(vframes)

            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert (
                end_frame_ind - start_frame_ind >= self.num_frames
            ), f"{path} with index {index} has not enough frames."
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
        for _ in range(50): # randomly get a good data, till 10 times
        # for _ in range(10): # randomly get a good data, till 10 times
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)


class DatasetFromCSV_Backup2(torch.utils.data.Dataset):
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
        self.csv_path = '/project/llmsvgen/share/data/webvid/part3.csv'
        # with open(csv_path, "r") as f:
        #     reader = csv.reader(f)
        #     self.samples = list(reader)

        df = pd.read_csv(self.csv_path)
        self.samples = df.values.tolist()
        random.shuffle(self.samples)


        self.is_video = True

        # ext = self.samples[0][0].split(".")[-1]
        # if ext.lower() in ("mp4", "avi", "mov", "mkv"): # supported videos
        #     self.is_video = True
        # else:
        #     assert f".{ext.lower()}" in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
        #     self.is_video = False

        self.transform = transform

        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
        self.root = root

    def getitem(self, index):
        sample = self.samples[index]
        # print(sample)
        # exit(0)

        second, first, text = sample[0], sample[3], sample[4]
        path = f'/project/llmsvgen/share/data/webvid/videos/{first}/{second}.mp4'
        # path = sample[0]
        # if self.root:
        #     path = os.path.join(self.root, path)
        text = sample[1]

        # print(text)
        
        if self.is_video:
            vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
            total_frames = len(vframes)

            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert (
                end_frame_ind - start_frame_ind >= self.num_frames
            ), f"{path} with index {index} has not enough frames."
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


class DatasetFromCSV_Backup3(torch.utils.data.Dataset):
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
        self.root = root

    def getitem(self, index):
        data = self.data_list[index]
        path = data['video_path']
        text = data['caption']
        
        if self.is_video:
            if os.path.exists(path):
                try:
                    vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
                    if len(vframes) == 0:
                        safe_data = random.choice(self.safe_data_list)
                        path = safe_data['video_path']
                        text = safe_data['caption']
                        vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
                except:
                    safe_data = random.choice(self.safe_data_list)
                    path = safe_data['video_path']
                    text = safe_data['caption']
                    vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
            else:
                safe_data = random.choice(self.safe_data_list)
                path = safe_data['video_path']
                text = safe_data['caption']
                vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
            total_frames = len(vframes)

            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert (
                end_frame_ind - start_frame_ind >= self.num_frames
            ), f"{path} with index {index} has not enough frames. Currnet frames: {len(vframes)}"
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