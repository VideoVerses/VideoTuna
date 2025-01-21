# Copyright 2024 Vchitect/Latte

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.# Modified from Latte

# - This file is adapted from https://github.com/Vchitect/Latte/blob/main/datasets/video_transforms.py


import numbers
import random

import decord
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as torch_transforms
from decord import VideoReader, cpu
from einops import rearrange
from PIL import Image
from torchvision.datasets.folder import pil_loader
from torchvision.io import write_video

from .datasets_utils import IMG_EXTS, VIDEO_EXTS


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i : i + h, j : j + w]


def resize(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(
            f"target size should be tuple (height, width), instead got {target_size}"
        )
    return torch.nn.functional.interpolate(
        clip, size=target_size, mode=interpolation_mode, align_corners=False
    )


def resize_scale(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(
            f"target size should be tuple (height, width), instead got {target_size}"
        )
    H, W = clip.size(-2), clip.size(-1)
    scale_ = target_size[0] / min(H, W)
    return torch.nn.functional.interpolate(
        clip, scale_factor=scale_, mode=interpolation_mode, align_corners=False
    )


def resized_crop(clip, i, j, h, w, size, interpolation_mode="bilinear"):
    """
    Do spatial cropping and resizing to the video clip
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped region.
        w (int): Width of the cropped region.
        size (tuple(int, int)): height and width of resized clip
    Returns:
        clip (torch.tensor): Resized and cropped clip. Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    clip = crop(clip, i, j, h, w)
    clip = resize(clip, size, interpolation_mode)
    return clip


def center_crop(clip, crop_size):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    th, tw = crop_size
    if h < th or w < tw:
        raise ValueError("height and width must be no smaller than crop_size")

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)


def center_crop_using_short_edge(clip):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    if h < w:
        th, tw = h, h
        i = 0
        j = int(round((w - tw) / 2.0))
    else:
        th, tw = w, w
        i = int(round((h - th) / 2.0))
        j = 0
    return crop(clip, i, j, th, tw)


def random_shift_crop(clip):
    """
    Slide along the long edge, with the short edge as crop size
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)

    if h <= w:
        short_edge = h
    else:
        short_edge = w

    th, tw = short_edge, short_edge

    i = torch.randint(0, h - th + 1, size=(1,)).item()
    j = torch.randint(0, w - tw + 1, size=(1,)).item()
    return crop(clip, i, j, th, tw)


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError(
            "clip tensor should have data type uint8. Got %s" % str(clip.dtype)
        )
    # return clip.float().permute(3, 0, 1, 2) / 255.0
    return clip.float() / 255.0


def normalize(clip, mean, std, inplace=False):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (T, C, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    # print(mean)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip


def hflip(clip):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (T, C, H, W)
    Returns:
        flipped clip (torch.tensor): Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    return clip.flip(-1)


def get_transforms_video(resolution=(256, 256), num_frames=16, frame_interval=1):
    transform_video = torch_transforms.Compose(
        [
            TemporalRandomCrop(num_frames, frame_interval),
            ToTensorVideo(),  # TCHW
            RandomHorizontalFlipVideo(),
            ResizeCenterCropVideo(resolution),
            torch_transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )
    return transform_video


def get_transforms_image(resolution=256, num_frames=16):
    transform = torch_transforms.Compose(
        [
            torch_transforms.Lambda(
                lambda pil_image: center_crop_arr(pil_image, resolution)
            ),
            torch_transforms.RandomHorizontalFlip(),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
            RepeatImage2Video(num_frames),
        ]
    )
    return transform


class RandomCropVideo:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: randomly cropped video clip.
                size is (T, C, OH, OW)
        """
        i, j, h, w = self.get_params(clip)
        return crop(clip, i, j, h, w)

    def get_params(self, clip):
        h, w = clip.shape[-2:]
        th, tw = self.size

        if h < th or w < tw:
            raise ValueError(
                f"Required crop size {(th, tw)} is larger than input image size {(h, w)}"
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()

        return i, j, th, tw

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class CenterCropResizeVideo:
    """
    First use the short side for cropping length,
    center crop video, then resize to the specified size
    """

    def __init__(
        self,
        size,
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(
                    f"size should be tuple (height, width), instead got {size}"
                )
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_center_crop = center_crop_using_short_edge(clip)
        clip_center_crop_resize = resize(
            clip_center_crop,
            target_size=self.size,
            interpolation_mode=self.interpolation_mode,
        )
        return clip_center_crop_resize

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class ResizeCenterCropVideo:
    """
    First resize to the specified size in equal proportion to the short edge,
    Then center crop to the desired size
    """

    def __init__(
        self,
        size,
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(
                    f"size should be tuple (height, width), instead got {size}"
                )
            self.size = size
        else:
            self.size = (size, size)
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (Tensor): Tensor of shape (T, C, H, W) representing video frames.

        Returns:
            Tensor: Processed video of shape (T, C, target_H, target_W).
        """
        resized_clip = self.resize_with_aspect_ratio(clip)
        cropped_clip = self.center_crop(resized_clip)
        return cropped_clip

    def resize_with_aspect_ratio(self, clip):
        """
        Resize the video tensor to maintain aspect ratio with the short edge.
        """
        T, C, H, W = clip.shape
        target_h, target_w = self.size

        # Determine the scaling factor based on the edge with the max scaling factor
        scale_factor = max(target_h / H, target_w / W)
        new_h, new_w = int(H * scale_factor), int(W * scale_factor)
        # if H < W:  # Short edge is height
        #     scale_factor = target_h / H
        #     new_h, new_w = target_h, int(W * scale_factor)
        # else:  # Short edge is width
        #     scale_factor = target_w / W
        #     new_h, new_w = int(H * scale_factor), target_w

        # Resize each frame in the video clip
        resized_clip = F.interpolate(
            clip, size=(new_h, new_w), mode=self.interpolation_mode, align_corners=False
        )
        return resized_clip

    def center_crop(self, clip):
        """
        Center crop the video tensor to the desired size.
        """
        T, C, H, W = clip.shape
        target_h, target_w = self.size
        assert (
            H >= target_h and W >= target_w
        ), "Video dimensions should be larger than crop size"
        # Compute cropping indices
        top = (H - target_h) // 2
        left = (W - target_w) // 2

        # Perform cropping
        cropped_clip = clip[:, :, top : top + target_h, left : left + target_w]
        return cropped_clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class UCFCenterCropVideo:
    """
    First scale to the specified size in equal proportion to the short edge,
    then center cropping
    """

    def __init__(
        self,
        size,
        interpolation_mode="bilinear",
    ):
        if not isinstance(size, int):
            assert len(size) == 2 or len(size) == 1, "size should be int or tuple"

        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(
                    f"size should be tuple (height, width), instead got {size}"
                )
            self.size = size
        elif isinstance(size, int):
            self.size = (size, size)
        else:
            if len(size) == 2:
                self.size = size
            else:
                self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_resize = resize_scale(
            clip=clip, target_size=self.size, interpolation_mode=self.interpolation_mode
        )
        clip_center_crop = center_crop(clip_resize, self.size)
        return clip_center_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class KineticsRandomCropResizeVideo:
    """
    Slide along the long edge, with the short edge as crop size. And resie to the desired size.
    """

    def __init__(
        self,
        size,
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(
                    f"size should be tuple (height, width), instead got {size}"
                )
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        clip_random_crop = random_shift_crop(clip)
        clip_resize = resize(clip_random_crop, self.size, self.interpolation_mode)
        return clip_resize


class CenterCropVideo:
    def __init__(
        self,
        size,
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(
                    f"size should be tuple (height, width), instead got {size}"
                )
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_center_crop = center_crop(clip, self.size)
        return clip_center_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class NormalizeVideo:
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip must be normalized. Size is (C, T, H, W)
        """
        return normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, inplace={self.inplace})"


class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


class RandomHorizontalFlipVideo:
    """
    Flip the video clip along the horizontal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (T, C, H, W)
        Return:
            clip (torch.tensor): Size is (T, C, H, W)
        """
        if random.random() < self.p:
            clip = hflip(clip)
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


#  ------------------------------------------------------------
#  ---------------------  Sampling  ---------------------------
#  ------------------------------------------------------------
class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    Args:
            size (int): Desired length of frames will be seen in the model.
    """

    def __init__(self, num_frames, frame_interval):
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.sample_length = num_frames * frame_interval

    def __call__(self, frames):
        total_frames = len(frames)
        rand_end = max(0, total_frames - self.sample_length - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.sample_length, total_frames)
        assert (
            end_index - begin_index >= self.num_frames
        ), f"The video has not enough frames. Current frames: {len(vframes)}"
        frame_indice = np.linspace(
            begin_index, end_index - 1, self.num_frames, dtype=int
        )
        sample_frames = frames[frame_indice]
        return sample_frames


class LoadDummyVideo:
    def __init__(self, resolution=(256, 256), num_frames=32, probs_fail=0):
        if isinstance(resolution, int):
            resolution = (resolution, resolution)

        self.resolution = resolution
        self.num_frames = num_frames
        self.probs_fail = probs_fail

    def __call__(self, video_path):
        if self.probs_fail > 0 and random.random() < self.probs_fail:
            raise ValueError(f"Failed to load video: {video_path}")
        return torch.randint(
            0,
            256,
            size=(self.num_frames, 3, self.resolution[0], self.resolution[1]),
            dtype=torch.uint8,
        )


class LoadVideo:
    def __init__(self):
        pass

    def __call__(self, video_path):
        assert video_path.split(".")[-1] in VIDEO_EXTS
        decord.bridge.set_bridge("torch")
        video = VideoReader(video_path, ctx=cpu(0))
        video_len = len(video)
        indexes = range(0, video_len)
        vframes = video.get_batch(indexes)
        vframes = rearrange(vframes, "t h w c -> t c h w")
        return vframes


class CheckVideo:
    def __init__(self, resolution=(256, 256), frame_interval=1, num_frames=32):
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.resolution = resolution
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.frame_limit = num_frames * frame_interval

    def __call__(self, vframes, index):
        length = vframes.shape[0]  # [F, C, H, W]
        h = vframes.shape[2]
        w = vframes.shape[3]
        if length <= self.frame_limit:
            raise ValueError(
                f"The video has not enough frames. Current frames: {length}"
            )
        if h < self.resolution[0] or w < self.resolution[1]:
            raise ValueError(f"Video resolution is too low: (h, w) = {(h, w)}")
        return vframes


class LoadDummyImage:
    def __init__(self, size=(225, 225), mode="rgb", probs_fail=0.1):
        assert isinstance(size, int) or (isinstance(size, tuple) and len(size) == 2)
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        assert mode in ["rgb", "gray"]
        self.mode = mode
        self.probs_fail = probs_fail

    def __call__(self, path):
        if self.mode == "rgb":
            size = self.size + (3,)
        elif self.mode == "gray":
            size = self.size
        image_arr = np.random.randint(0, 256, size, dtype=np.uint8)
        return Image.fromarray(image_arr)


class LoadImage:
    def __init__(self):
        pass

    def __call__(self, path):
        assert path.split(".")[-1] in IMG_EXTS
        return pil_loader(path)


class RepeatImage2Video:
    def __init__(self, num_frames):
        self.num_frames = num_frames

    def __call__(self, image):
        return image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)


if __name__ == "__main__":
    import os

    import numpy as np
    import torchvision.io as io
    from torchvision import transforms
    from torchvision.utils import save_image

    vframes, aframes, info = io.read_video(
        filename="./v_Archery_g01_c03.avi", pts_unit="sec", output_format="TCHW"
    )

    trans = transforms.Compose(
        [
            ToTensorVideo(),
            RandomHorizontalFlipVideo(),
            UCFCenterCropVideo(512),
            # NormalizeVideo(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )

    target_video_len = 32
    frame_interval = 1
    total_frames = len(vframes)
    print(total_frames)

    temporal_sample = TemporalRandomCrop(target_video_len * frame_interval)

    # Sampling video frames
    start_frame_ind, end_frame_ind = temporal_sample(total_frames)
    # print(start_frame_ind)
    # print(end_frame_ind)
    assert end_frame_ind - start_frame_ind >= target_video_len
    frame_indice = np.linspace(
        start_frame_ind, end_frame_ind - 1, target_video_len, dtype=int
    )
    print(frame_indice)

    select_vframes = vframes[frame_indice]
    print(select_vframes.shape)
    print(select_vframes.dtype)

    select_vframes_trans = trans(select_vframes)
    print(select_vframes_trans.shape)
    print(select_vframes_trans.dtype)

    select_vframes_trans_int = ((select_vframes_trans * 0.5 + 0.5) * 255).to(
        dtype=torch.uint8
    )
    print(select_vframes_trans_int.dtype)
    print(select_vframes_trans_int.permute(0, 2, 3, 1).shape)

    io.write_video("./test.avi", select_vframes_trans_int.permute(0, 2, 3, 1), fps=8)

    for i in range(target_video_len):
        save_image(
            select_vframes_trans[i],
            os.path.join("./test000", "%04d.png" % i),
            normalize=True,
            value_range=(-1, 1),
        )
