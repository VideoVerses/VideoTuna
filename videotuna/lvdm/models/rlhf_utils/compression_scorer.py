# Adapt from Cheng An Hsieh, et. al.: https://github.com/RewardMultiverse/reward-multiverse
import io

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

# import albumentations as A
from transformers import CLIPModel, CLIPProcessor

# import ipdb
# st = ipdb.set_trace


def jpeg_compressibility(device):
    def _fn(images):
        """
        args:
            images: shape NCHW
        """
        org_type = images.dtype
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        transform_images_tensor = torch.Tensor(np.array(images)).to(
            device, dtype=org_type
        )
        transform_images_tensor = (
            transform_images_tensor.permute(0, 3, 1, 2) / 255
        ).clamp(
            0, 1
        )  # NHWC -> NCHW
        transform_images_pil = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in transform_images_pil]

        for image, buffer in zip(transform_images_pil, buffers):
            image.save(buffer, format="JPEG", quality=95)

        sizes = [buffer.tell() / 1000 for buffer in buffers]

        return np.array(sizes), transform_images_tensor

    return _fn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, embed):
        return self.layers(embed)


def jpegcompression_loss_fn(
    target=None,
    grad_scale=0,
    device=None,
    accelerator=None,
    torch_dtype=None,
    reward_model_resume_from=None,
):
    scorer = JpegCompressionScorer(
        dtype=torch_dtype, model_path=reward_model_resume_from
    ).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    scorer.eval()

    def loss_fn(im_pix_un):
        if accelerator.mixed_precision == "fp16":
            with accelerator.autocast():
                rewards = scorer(im_pix_un)
        else:
            rewards = scorer(im_pix_un)

        if target is None:
            loss = rewards
        else:
            loss = abs(rewards - target)
        return loss * grad_scale, rewards

    return loss_fn


class JpegCompressionScorer(nn.Module):
    def __init__(self, dtype=None, model_path=None):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip.requires_grad_(False)
        self.score_generator = MLP()

        if model_path:
            state_dict = torch.load(model_path)
            self.score_generator.load_state_dict(state_dict)
        if dtype:
            self.dtype = dtype
        self.target_size = (224, 224)
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    def set_device(self, device, inference_type):
        # self.clip.to(device, dtype = inference_type)
        self.score_generator.to(device)  # , dtype = inference_type

    def __call__(self, images):
        device = next(self.parameters()).device
        im_pix = torchvision.transforms.Resize(self.target_size)(images)
        im_pix = self.normalize(im_pix).to(images.dtype)
        embed = self.clip.get_image_features(pixel_values=im_pix)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.score_generator(embed).squeeze(1)
