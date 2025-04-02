# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py
# import ipdb
# st = ipdb.set_trace
# from importlib_resources import files
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPModel

# ASSETS_PATH = files("lvdm.models.rlhf_utils.pretrained_reward_models")
ASSETS_PATH = "videotuna/lvdm/models/rlhf_utils/pretrained_reward_models"


class MLPDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed):
        return self.layers(embed)


class AestheticScorerDiff(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLPDiff()
        state_dict = torch.load(
            os.path.join(ASSETS_PATH, "sac+logos+ava1-l14-linearMSE.pth")
        )
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    def __call__(self, images):
        # device = next(self.parameters()).device
        # print("AestheticScorerDiff",device)
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)

    def eval_video(self, video_path):
        # read a video and return the aesthetic score
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frame = frame / 255.0
            frames.append(frame)
        # im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)
        # im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        # im_pix = normalize(im_pix).to(im_pix_un.dtype)
        frames = np.array(frames)
        frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)
        scores = self(frames)
        return scores.mean().item()

    def eval_video_folder(self, video_folder):
        # read a folder of videos and return the aesthetic scores

        files = os.listdir(video_folder)
        scores = []
        for file in files:
            if file.endswith(".mp4"):
                score = self.eval_video(video_folder + "/" + file)
                scores.append(score)


## the main function is a aesthetic scorer that takes in a video folder and returns the aesthetic scores
if __name__ == "__main__":
    target_size = (224, 224)
    normalize = torchvision.transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    video_folder = (
        "path to video /rlhf-visual-results/lora_aes_chatgpt_instructions-3184"
    )
    scores = scorer.eval_video_folder(video_folder)
    print(type(scores), type(scores[0]))
    print(scores, np.mean(scores))
