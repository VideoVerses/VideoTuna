import argparse
import copy
import glob
import json
import os
import warnings
from operator import attrgetter

import cv2
import numpy as np
import requests
import torch
import tqdm
from decord import VideoReader, cpu
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from PIL import Image


def load_video1(folder_path, num_segments):
    images = sorted(glob.glob(f"{folder_path}/*[!a-z].png"))
    num_frames = len(images)
    frame_indices = np.linspace(0, num_frames - 1, num_segments, dtype=int)

    images_group = list()
    for frame_index in frame_indices:
        img = Image.open(images[frame_index])
        images_group.append(img)
    return np.array(images_group)


def get_inf1(folder_path):
    file_path = "/project/llmsvgen/VFHQ/meta_info"
    with open(
        file_path + "/" + os.path.splitext(os.path.basename(folder_path))[0] + ".txt",
        "r",
    ) as info:
        file_lines = info.readlines()
        for line in file_lines:
            if line.startswith("FPS:"):
                fps = int(float(line.split(":")[1].strip()))
                break
    image = Image.open(f"{folder_path}/00000000.png")
    w, h = image.size
    duration = len(glob.glob(f"{folder_path}/*png")) / fps
    return duration, fps, h, w


def get_inf(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    video_length = len(vr)
    video_fps = vr.get_avg_fps()
    width = vr[0].shape[0]
    height = vr[0].shape[1]
    duration = video_length / video_fps
    return duration, video_fps, height, width


# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(
        0, total_frame_num - 1, max_frames_num, dtype=int
    )
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


def inference(args):
    warnings.filterwarnings("ignore")
    # Load the OneVision model
    pretrained = args.model_path
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa"
    )

    model.eval()
    num = args.num_process
    no = args.mp_no
    save_path = args.out_dir
    video_list = sorted(glob.glob(args.vid_dir + "/*mp4"))
    length = len(video_list)
    if no != num - 1:
        video_list = video_list[length // num * no : length // num * (no + 1)]
    else:
        video_list = video_list[length // num * no :]
    video_list = [
        data
        for data in video_list
        if not os.path.exists(
            save_path + "/" + os.path.splitext(os.path.basename(data))[0] + ".json"
        )
    ]

    for video in tqdm.tqdm(video_list):
        try:
            video_path = video
            # Load and process video
            video_frames = load_video(video_path, args.num_frame)
            # print(video_frames.shape) # (16, 1024, 576, 3)
            image_tensors = []
            frames = (
                image_processor.preprocess(video_frames, return_tensors="pt")[
                    "pixel_values"
                ]
                .half()
                .cuda()
            )
            image_tensors.append(frames)

            # Prepare conversation input
            conv_template = "qwen_1_5"
            question = f"{DEFAULT_IMAGE_TOKEN}\nPlease use no more than two sentences to generate a detailed video caption that describes the scene comprehensively and accurately. The caption should include specific elements such as the individuals, the setting, any notable objects or weather conditions, and the general atmosphere. The focus should be on providing a clear and precise description to help someone who cannot see the video understand the scene fully. Just describe the video content without making any comment or interpretation on it."
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(device)
            )
            image_sizes = [frame.size for frame in video_frames]

            # Generate response
            cont = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=2048,
                modalities=["video"],
                top_p=1,
            )
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
            duration, fps, h, w = get_inf(video_path)
            result = {
                "basic": {
                    "clip_duration": duration,
                    "clip_path": video_path,
                    "video_fps": fps,
                    "video_resolution": [h, w],
                },
                "misc": {
                    "caption": text_outputs[0],
                },
            }
            with open(
                args.out_dir
                + "/"
                + os.path.splitext(os.path.basename(video_path))[0]
                + ".json",
                "w",
            ) as f:
                json.dump(result, f, indent=4)
            # print(text_outputs[0])
        except Exception as e:
            print("An error occurred:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/project/llmsvgen/pengjun/LLaVA-NeXT/llava-onevision-qwen2-7b-ov",
    )
    parser.add_argument(
        "--vid_dir",
        type=str,
        default="/project/llmsvgen/share/data_trailer_human/videos",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/project/llmsvgen/pengjun/LLaVA-NeXT2/trailer_caption",
    )
    parser.add_argument("--num_frame", type=int, default=32)
    parser.add_argument("--num_process", type=int, default=1)
    parser.add_argument("--mp_no", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    inference(args)
