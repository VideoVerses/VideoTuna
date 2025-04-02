import bisect
import os
import random

import omegaconf
import pandas as pd
import torch
import torchvision.transforms._transforms_video as transforms_video
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms


def is_list(x):
    return isinstance(x, omegaconf.listconfig.ListConfig) or isinstance(x, list)


def sample_strided_frames(vid_len, frame_stride, target_vid_len):
    frame_indices = list(range(0, vid_len, frame_stride))
    if len(frame_indices) < target_vid_len:
        frame_stride = vid_len // target_vid_len  # recal a max fs
        assert frame_stride != 0
        frame_indices = list(range(0, vid_len, frame_stride))
    return frame_indices, frame_stride


def make_spatial_transformations(resolution, type, ori_resolution=None):
    """
    resolution: target resolution, a list of int, [h, w]
    """
    if type == "random_crop":
        transformations = transforms_video.RandomCropVideo(resolution)
    elif type == "resize_center_crop":
        is_square = resolution[0] == resolution[1]
        if is_square:
            transformations = transforms.Compose(
                [
                    transforms.Resize(resolution[0]),
                    transforms_video.CenterCropVideo(resolution[0]),
                ]
            )
        else:
            if ori_resolution is not None:
                # resize while keeping original aspect ratio,
                # then centercrop to target resolution
                resize_ratio = max(
                    resolution[0] / ori_resolution[0], resolution[1] / ori_resolution[1]
                )
                resolution_after_resize = [
                    int(ori_resolution[0] * resize_ratio),
                    int(ori_resolution[1] * resize_ratio),
                ]
                transformations = transforms.Compose(
                    [
                        transforms.Resize(resolution_after_resize),
                        transforms_video.CenterCropVideo(resolution),
                    ]
                )
            else:
                # directly resize to target resolution
                transformations = transforms.Compose(
                    [
                        transforms.Resize(resolution),
                    ]
                )
    else:
        raise NotImplementedError
    return transformations


class WebVid(Dataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """

    def __init__(
        self,
        meta_path,
        data_dir,
        subsample=None,
        video_length=16,
        resolution=[256, 512],
        frame_stride=None,
        spatial_transform=None,
        fps_max=None,
        load_raw_resolution=False,
        fps_schedule=None,
        fps_list=None,
        fs_probs=None,
        bs_per_gpu=None,
        num_workers=1,
        trigger_word="",
    ):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = (
            [resolution, resolution] if isinstance(resolution, int) else resolution
        )
        self.frame_stride = frame_stride
        self.fps_max = fps_max
        self.load_raw_resolution = load_raw_resolution
        self.fs_probs = fs_probs
        assert not (is_list(frame_stride) and is_list(fps_list))
        self.spatial_transform_type = spatial_transform
        self.trigger_word = trigger_word

        self._load_metadata()
        self.num_workers = num_workers

        if not isinstance(self.resolution[0], int):
            # multiple resolutions training
            assert isinstance(self.resolution[0], list) or isinstance(
                self.resolution[0], omegaconf.listconfig.ListConfig
            )
            self.num_resolutions = len(resolution)
            self.spatial_transform = None
            self.load_raw_resolution = True
        else:
            self.num_resolutions = 1
            self.spatial_transform = (
                make_spatial_transformations(
                    self.resolution, type=self.spatial_transform_type
                )
                if self.spatial_transform_type is not None
                else None
            )

        self.fps_list = [fps_list] if isinstance(fps_list, int) else fps_list

        self.fps_schedule = fps_schedule
        self.bs_per_gpu = bs_per_gpu
        if self.bs_per_gpu is not None:
            # counter loaded video number
            self.counter = 0
        if self.fps_schedule is not None:
            # log fps stage index
            self.stage_idx = 0

    def _load_metadata(self):
        metadata = pd.read_csv(self.meta_path)
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        metadata["caption"] = metadata["name"]
        del metadata["name"]
        self.metadata = metadata
        self.metadata.dropna(inplace=True)
        # self.metadata['caption'] = self.metadata['caption'].str[:350]

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample["page_dir"], str(sample["videoid"]) + ".mp4")
        full_video_fp = os.path.join(self.data_dir, "videos", rel_video_fp)
        return full_video_fp, rel_video_fp

    def get_fs_based_on_schedule(self, frame_strides, schedule):
        assert len(frame_strides) == len(schedule) + 1  # nstage=len_fps_schedule + 1
        stage_idx = bisect.bisect(schedule, self.global_step)
        frame_stride = frame_strides[stage_idx]
        # log fps stage change
        if stage_idx != self.stage_idx:
            print(f"fps stage: {stage_idx} start ... new frame stride = {frame_stride}")
            self.stage_idx = stage_idx
        return frame_stride

    def get_item_based_on_probs(self, alist, probs):
        assert len(alist) == len(probs)
        return random.choices(alist, weights=probs)[0]

    def get_fs_randomly(self, frame_strides):
        return random.choice(frame_strides)

    def __getitem__(self, index):

        # set up dynamic resolution & spatial transformations
        if self.bs_per_gpu is not None:
            # self.global_step = self.counter * self.num_workers // self.bs_per_gpu # TODO: support resume.
            self.global_step = self.counter // self.bs_per_gpu  # TODO: support resume.
        else:
            self.global_step = None
        # print(f"self.global_step={self.global_step}")

        if isinstance(self.frame_stride, list) or isinstance(
            self.frame_stride, omegaconf.listconfig.ListConfig
        ):
            if self.fps_schedule is not None:
                frame_stride = self.get_fs_based_on_schedule(
                    self.frame_stride, self.fps_schedule
                )
            elif self.fs_probs is not None:
                frame_stride = self.get_item_based_on_probs(
                    self.frame_stride, self.fs_probs
                )
            else:
                frame_stride = self.get_fs_randomly(self.frame_stride)
        else:
            frame_stride = self.frame_stride
        assert isinstance(frame_stride, int) or frame_stride is None, type(frame_stride)

        while True:
            index = index % len(self.metadata)
            sample = self.metadata.iloc[index]
            video_path, rel_fp = self._get_video_path(sample)
            caption = sample["caption"] + self.trigger_word

            # make reader
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(
                        video_path,
                        ctx=cpu(0),
                        width=self.resolution[1],
                        height=self.resolution[0],
                    )
                vid_len = len(video_reader)
                if vid_len < self.video_length:
                    print(
                        f"video length ({vid_len}) is smaller than target length({self.video_length})"
                    )
                    index += 1
                    continue
                else:
                    pass
            except Exception:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue

            fps_ori = video_reader.get_avg_fps()

            # sample strided frames
            if frame_stride is None:
                assert self.fps_list is not None
                fps = self.get_item_based_on_probs(self.fps_list, self.fs_probs)
                frame_stride = int(fps_ori // fps)
                if frame_stride == 0:
                    frame_stride = 1
            all_frames, frame_stride = sample_strided_frames(
                vid_len, frame_stride, self.video_length
            )

            # select a random clip
            rand_idx = random.randint(0, len(all_frames) - self.video_length)
            frame_indices = all_frames[rand_idx : rand_idx + self.video_length]

            # load clip
            try:
                frames = video_reader.get_batch(frame_indices)
                break
            except Exception:
                print(f"Get frames failed! path = {video_path}")
                index += 1
                continue

        # transform
        assert (
            frames.shape[0] == self.video_length
        ), f"{len(frames)}, self.video_length={self.video_length}"
        frames = (
            torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        )  # [t,h,w,c] -> [c,t,h,w]

        if self.num_resolutions > 1:
            res_idx = self.global_step % 3
            res_curr = self.resolution[res_idx]
            self.spatial_transform = make_spatial_transformations(
                res_curr, self.spatial_transform_type, ori_resolution=frames.shape[2:]
            )
        else:
            pass

        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        if self.resolution is not None:
            if self.num_resolutions > 1:
                assert (
                    frames.shape[2] == res_curr[0] and frames.shape[3] == res_curr[1]
                ), f"frames={frames.shape}, res_curr={res_curr}"
            else:
                assert (
                    frames.shape[2] == self.resolution[0]
                    and frames.shape[3] == self.resolution[1]
                ), f"frames={frames.shape}, self.resolution={self.resolution}"
        frames = (frames / 255 - 0.5) * 2

        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max

        data = {
            "video": frames,
            "caption": caption,
            "path": video_path,
            "fps": fps_clip,
            "frame_stride": frame_stride,
        }

        if self.bs_per_gpu is not None:
            self.counter += 1
            # print(f'self.counter={self.counter}')
        return data

    def __len__(self):
        return len(self.metadata)
