import json
import logging
import os

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from vbench.utils import clip_transform, load_dimension_info, load_video


def background_consistency(clip_model, preprocess, video_list, device, read_frame):
    sim = 0.0
    cnt = 0
    video_results = []
    image_transform = clip_transform(224)
    for video_path in tqdm(video_list):
        video_sim = 0.0
        if read_frame:
            video_path = video_path[:-4].replace("videos", "frames").replace(" ", "_")
            tmp_paths = [
                os.path.join(video_path, f) for f in sorted(os.listdir(video_path))
            ]
            images = []
            for tmp_path in tmp_paths:
                images.append(preprocess(Image.open(tmp_path)))
            images = torch.stack(images)
        else:
            images = load_video(video_path)
            images = image_transform(images)
        images = images.to(device)
        image_features = clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1, p=2)
        for i in range(len(image_features)):
            image_feature = image_features[i].unsqueeze(0)
            if i == 0:
                first_image_feature = image_feature
            else:
                sim_pre = max(
                    0.0, F.cosine_similarity(former_image_feature, image_feature).item()
                )
                sim_fir = max(
                    0.0, F.cosine_similarity(first_image_feature, image_feature).item()
                )
                cur_sim = (sim_pre + sim_fir) / 2
                video_sim += cur_sim
                cnt += 1
            former_image_feature = image_feature
        sim_per_image = video_sim / (len(image_features) - 1)
        sim += video_sim
        video_results.append({"video_path": video_path, "video_results": sim_per_image})
    # sim_per_video = sim / (len(video_list) - 1)
    sim_per_frame = sim / cnt
    return sim_per_frame, video_results


def compute_background_consistency(json_dir, device, submodules_list, **kwargs):
    vit_path, read_frame = submodules_list[0], submodules_list[1]
    clip_model, preprocess = clip.load(vit_path, device=device)
    video_list, _ = load_dimension_info(
        json_dir, dimension="background_consistency", lang="en"
    )
    all_results, video_results = background_consistency(
        clip_model, preprocess, video_list, device, read_frame
    )
    return all_results, video_results
