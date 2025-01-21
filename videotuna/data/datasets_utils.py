import cv2
import decord
import numpy as np
import torch
import torchvision.transforms as transforms
from decord import VideoReader, cpu
from einops import rearrange
from PIL import Image
from torchvision.io import write_video
from torchvision.utils import save_image

from . import transforms

IMG_EXTS = {"jpg", "bmp", "png", "jpeg", "rgb", "tif"}
VIDEO_EXTS = {"mp4", "avi", "mov", "flv", "mkv", "webm", "wmv", "mov"}


def save_sample(x, fps=8, save_path=None, normalize=True, value_range=(-1, 1)):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
    """
    assert x.ndim == 4

    if x.shape[1] == 1:  # T = 1: save as image
        save_path += ".png"
        x = x.squeeze(1)
        save_image([x], save_path, normalize=normalize, value_range=value_range)
    else:
        save_path += ".mp4"
        if normalize:
            low, high = value_range
            x.clamp_(min=low, max=high)
            x.sub_(low).div_(max(high - low, 1e-5))

        x = (
            x.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 3, 0)
            .to("cpu", torch.uint8)
        )
        write_video(save_path, x, fps=fps, video_codec="h264")
    print(f"Saved to {save_path}")
    return save_path


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


def read_video(video_path, fps=False):
    decord.bridge.set_bridge("torch")
    video = VideoReader(video_path, ctx=cpu(0))
    video_len = len(video)
    indexes = range(0, video_len)
    vframes = video.get_batch(indexes)
    vframes = rearrange(vframes, "t h w c -> t c h w")

    if fps:
        return vframes, video.get_avg_fps()
    else:
        return vframes


def read_video_meta(video_path):
    # Video fps
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # The number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Height
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Width
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    return {
        "fps": fps,
        "frames": num_frames,
        "height": height,
        "width": width,
    }


def read_image_meta(image_path):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    return {
        "height": height,
        "width": width,
    }


def is_video(path):
    return path.split(".")[-1] in VIDEO_EXTS


def is_image(path):
    return path.split(".")[-1] in IMG_EXTS
