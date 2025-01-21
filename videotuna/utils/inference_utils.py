import glob
import os
import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from decord import VideoReader, cpu
from einops import rearrange, repeat
from PIL import Image

from videotuna.utils.load_weights import load_safetensors


def get_target_filelist(data_dir, ext):
    """
    Generate a sorted filepath list with target extensions.
    Args:
        data_dir (str): The directory to search for files.
        ext (str): A comma-separated string of file extensions to match.
               Examples:
                   - ext = "png,jpg,webp" (multiple extensions)
                   - ext = "png" (single extension)
    Returns: list: A sorted list of file paths matching the given extensions.
    """
    file_list = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(tuple(ext.split(",")))
    ]
    if len(file_list) == 0:
        raise ValueError(f"No file with extensions {ext} found in {data_dir}.")
    return file_list


def load_prompts_from_txt(prompt_file: str):
    """Load and return a list of prompts from a text file, stripping whitespace."""
    with open(prompt_file, "r") as f:
        lines = f.readlines()
    prompt_list = [line.strip() for line in lines if line.strip() != ""]
    return prompt_list


def load_model_checkpoint(model, ckpt):
    def load_checkpoint(model, ckpt, full_strict):
        state_dict = torch.load(ckpt, map_location="cpu")
        try:
            # deepspeed version
            new_pl_sd = OrderedDict()
            for key in state_dict["module"].keys():
                new_pl_sd[key[16:]] = state_dict["module"][key]
            model.load_state_dict(new_pl_sd, strict=full_strict)
        except:
            if "state_dict" in list(state_dict.keys()):
                state_dict = state_dict["state_dict"]
            try:
                model.model.diffusion_model.load_state_dict(
                    state_dict, strict=full_strict
                )
            except:
                model.load_state_dict(state_dict, strict=False)
        return model

    if ckpt.endswith(".safetensors"):
        state_dict = load_safetensors(ckpt)
        model.load_state_dict(state_dict, strict=False)
    else:
        load_checkpoint(model, ckpt, full_strict=True)
    print("[INFO] model checkpoint loaded.")
    return model


def load_inputs_i2v(input_dir, video_size=(256, 256), video_frames=16):
    """
    Load prompt list and conditional images for i2v from input_dir.
    """
    # load prompt files
    prompt_files = get_target_filelist(input_dir, ext="txt")
    if len(prompt_files) > 1:
        # only use the first one (sorted by name) if multiple exist
        print(
            f"Warning: multiple prompt files exist. The one {os.path.split(prompt_files[0])[1]} is used."
        )
        prompt_file = prompt_files[0]
    elif len(prompt_files) == 1:
        prompt_file = prompt_files[0]
    elif len(prompt_files) == 0:
        print(prompt_files)
        raise ValueError(f"Error: found NO prompt file in {input_dir}")
    prompt_list = load_prompts_from_txt(prompt_file)
    n_samples = len(prompt_list)

    ## load images
    img_paths = get_target_filelist(input_dir, ext="png,jpg,webp,jpeg")

    # image transforms
    transform = transforms.Compose(
        [
            transforms.Resize(min(video_size)),
            transforms.CenterCrop(video_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    image_list = []
    filename_list = []
    for idx in range(n_samples):
        # load, transform, repeat 4D T~ to 5D
        image = Image.open(img_paths[idx]).convert("RGB")
        image_tensor = transform(image).unsqueeze(1)  # [c,1,h,w]
        frame_tensor = repeat(
            image_tensor, "c t h w -> c (repeat t) h w", repeat=video_frames
        )
        image_list.append(frame_tensor)

        _, filename = os.path.split(img_paths[idx])
        filename_list.append(filename.split(".")[0])

    return filename_list, image_list, prompt_list


def load_inputs_v2v(input_dir, video_size=None, video_frames=None):
    """
    Load prompt list and input videos for v2v from an input directory.
    """
    # load prompt files
    prompt_files = get_target_filelist(input_dir, ext="txt")
    if len(prompt_files) > 1:
        # only use the first one (sorted by name) if multiple exist
        print(
            f"Warning: multiple prompt files exist. The one {os.path.split(prompt_files[0])[1]} is used."
        )
        prompt_file = prompt_files[0]
    elif len(prompt_files) == 1:
        prompt_file = prompt_files[0]
    elif len(prompt_files) == 0:
        print(prompt_files)
        raise ValueError(f"Error: found NO prompt file in {input_dir}")
    prompt_list = load_prompts_from_txt(prompt_file)
    n_samples = len(prompt_list)

    ## load videos
    video_filepaths = get_target_filelist(input_dir, ext="mp4")
    video_filenames = [
        os.path.split(video_filepath)[-1] for video_filepath in video_filepaths
    ]

    return prompt_list, video_filepaths, video_filenames


def open_video_to_tensor(filepath, video_width=None, video_height=None):
    if video_width is None and video_height is None:
        vidreader = VideoReader(
            filepath, ctx=cpu(0), width=video_width, height=video_height
        )
    else:
        vidreader = VideoReader(filepath, ctx=cpu(0))
    frame_indices = list(range(len(vidreader)))
    frames = vidreader.get_batch(frame_indices)
    frame_tensor = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
    frame_tensor = (frame_tensor / 255.0 - 0.5) * 2
    return frame_tensor.unsqueeze(0)


def load_video_batch(
    filepath_list, frame_stride, video_size=(256, 256), video_frames=16
):
    """
    Notice about some special cases:
    1. video_frames=-1 means to take all the frames (with fs=1)
    2. when the total video frames is less than required, padding strategy will be used (repreated last frame)
    """
    fps_list = []
    batch_tensor = []
    assert frame_stride > 0, "valid frame stride should be a positive interge!"
    for filepath in filepath_list:
        padding_num = 0
        vidreader = VideoReader(
            filepath, ctx=cpu(0), width=video_size[1], height=video_size[0]
        )
        fps = vidreader.get_avg_fps()
        total_frames = len(vidreader)
        max_valid_frames = (total_frames - 1) // frame_stride + 1
        if video_frames < 0:
            ## all frames are collected: fs=1 is a must
            required_frames = total_frames
            frame_stride = 1
        else:
            required_frames = video_frames
        query_frames = min(required_frames, max_valid_frames)
        frame_indices = [frame_stride * i for i in range(query_frames)]

        ## [t,h,w,c] -> [c,t,h,w]
        frames = vidreader.get_batch(frame_indices)
        frame_tensor = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        frame_tensor = (frame_tensor / 255.0 - 0.5) * 2
        if max_valid_frames < required_frames:
            padding_num = required_frames - max_valid_frames
            frame_tensor = torch.cat(
                [frame_tensor, *([frame_tensor[:, -1:, :, :]] * padding_num)], dim=1
            )
            print(
                f"{os.path.split(filepath)[1]} is not long enough: {padding_num} frames padded."
            )
        batch_tensor.append(frame_tensor)
        sample_fps = int(fps / frame_stride)
        fps_list.append(sample_fps)

    return torch.stack(batch_tensor, dim=0)


def load_image_batch(filepath_list, image_size=(256, 256)):
    batch_tensor = []
    for filepath in filepath_list:
        _, filename = os.path.split(filepath)
        _, ext = os.path.splitext(filename)
        if ext == ".mp4":
            vidreader = VideoReader(
                filepath, ctx=cpu(0), width=image_size[1], height=image_size[0]
            )
            frame = vidreader.get_batch([0])
            img_tensor = (
                torch.tensor(frame.asnumpy()).squeeze(0).permute(2, 0, 1).float()
            )
        elif ext == ".png" or ext == ".jpg":
            img = Image.open(filepath).convert("RGB")
            rgb_img = np.array(img, np.float32)
            rgb_img = cv2.resize(
                rgb_img, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR
            )
            img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
        else:
            print(
                f"ERROR: <{ext}> image loading only support format: [mp4], [png], [jpg]"
            )
            raise NotImplementedError
        img_tensor = (img_tensor / 255.0 - 0.5) * 2
        batch_tensor.append(img_tensor)
    return torch.stack(batch_tensor, dim=0)


def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, "b c t h w -> (b t) c h w")
    z = model.encode_first_stage(x)
    z = rearrange(z, "(b t) c h w -> b c t h w", b=b, t=t)
    return z


def sample_batch_t2v(
    model,
    sampler,
    prompts,
    noise_shape,
    fps,
    n_samples_prompt=1,
    ddim_steps=50,
    ddim_eta=1.0,
    cfg_scale=1.0,
    temporal_cfg_scale=None,
    uncond_prompt="",
    **kwargs,
):
    # ----------------------------------------------------------------------------------
    # make cond & uncond for t2v
    batch_size = noise_shape[0]
    text_emb = model.get_learned_conditioning(prompts)
    fps = torch.tensor([fps] * batch_size).to(model.device).long()
    cond = {"c_crossattn": [text_emb], "fps": fps}

    if cfg_scale != 1.0:  # unconditional guidance
        uc_text_emb = model.get_learned_conditioning(batch_size * [uncond_prompt])
        uncond = {k: v for k, v in cond.items()}
        uncond.update({"c_crossattn": [uc_text_emb]})
    else:
        uncond = None

    # ----------------------------------------------------------------------------------
    # sampling
    batch_samples = []
    for _ in range(n_samples_prompt):  # iter over batch of prompts
        samples, _ = sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=batch_size,
            shape=noise_shape[1:],
            verbose=False,
            unconditional_guidance_scale=cfg_scale,
            unconditional_conditioning=uncond,
            eta=ddim_eta,
            temporal_length=noise_shape[2],
            conditional_guidance_scale_temporal=temporal_cfg_scale,
            **kwargs,
        )
        res = model.decode_first_stage(samples)
        batch_samples.append(res)
    batch_samples = torch.stack(batch_samples, dim=1)
    return batch_samples


def sample_batch_i2v(
    model,
    sampler,
    prompts,
    images,
    noise_shape,
    n_samples_prompt=1,
    ddim_steps=50,
    ddim_eta=1.0,
    unconditional_guidance_scale=1.0,
    cfg_img=None,
    fs=None,
    uncond_prompt="",
    multiple_cond_cfg=False,
    loop=False,
    gfi=False,
    timestep_spacing="uniform",
    guidance_rescale=0.0,
    **kwargs,
):
    batch_size = noise_shape[0]

    # ----------------------------------------------------------------------------------
    # prepare condition for i2v
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    # cond: text embedding, image embedding
    img = images[:, :, 0]  # bchw
    img_emb = model.embedder(img)  ## blc
    img_emb = model.image_proj_model(img_emb)
    text_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([text_emb, img_emb], dim=1)]}
    # concat condition imgs
    if model.model.conditioning_key == "hybrid":
        z = get_latent_z(model, images)  # b c t h w
        if loop or gfi:
            img_cat_cond = torch.zeros_like(z)
            img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
            img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
        else:
            img_cat_cond = z[:, :, :1, :, :]
            img_cat_cond = repeat(
                img_cat_cond, "b c t h w -> b c (repeat t) h w", repeat=z.shape[2]
            )
        cond["c_concat"] = [img_cat_cond]  # b c 1 h w

    # uncond
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            uc_text_emb = model.get_learned_conditioning([uncond_prompt] * batch_size)
        elif model.uncond_type == "zero_embed":
            uc_text_emb = torch.zeros_like(text_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img))  ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uncond = {"c_crossattn": [torch.cat([uc_text_emb, uc_img_emb], dim=1)]}
        if model.model.conditioning_key == "hybrid":
            uncond["c_concat"] = [img_cat_cond]
    else:
        uncond = None

    ## uncond2: we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uncond2 = {"c_crossattn": [torch.cat([uc_text_emb, img_emb], dim=1)]}
        if model.model.conditioning_key == "hybrid":
            uncond2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uncond2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    # ----------------------------------------------------------------------------------
    # sampling
    z0 = None
    cond_mask = None

    batch_samples = []
    for _ in range(n_samples_prompt):

        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None

        samples, _ = sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=batch_size,
            shape=noise_shape[1:],
            verbose=False,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uncond,
            eta=ddim_eta,
            cfg_img=cfg_img,
            mask=cond_mask,
            x0=cond_z0,
            fs=fs,
            timestep_spacing=timestep_spacing,
            guidance_rescale=guidance_rescale,
            **kwargs,
        )
        res = model.decode_first_stage(samples)
        batch_samples.append(res)
    ## variants, batch, c, t, h, w
    batch_samples = torch.stack(batch_samples)
    return batch_samples.permute(1, 0, 2, 3, 4, 5)


def save_videos(batch_tensors, savedir, filenames, fps=10):
    # b,samples,c,t,h,w
    n_samples = batch_tensors.shape[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1.0, 1.0)
        video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w
        frame_grids = [
            torchvision.utils.make_grid(framesheet, nrow=int(n_samples))
            for framesheet in video
        ]  # [3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
        torchvision.io.write_video(
            savepath, grid, fps=fps, video_codec="h264", options={"crf": "10"}
        )


def save_videos_vbench(batch_tensors, savedir, prompts, format_file, fps=10):
    # b,samples,c,t,h,w
    b = batch_tensors.shape[0]
    n_samples = batch_tensors.shape[1]

    sub_savedir = os.path.join(savedir, "videos")
    os.makedirs(sub_savedir, exist_ok=True)

    for idx in range(b):
        prompt = prompts[idx]
        for n in range(n_samples):
            filename = f"{prompt}-{n}.mp4"
            format_file[filename] = prompt
            video = batch_tensors[idx, n].detach().cpu()
            video = torch.clamp(video.float(), -1.0, 1.0)
            video = video.permute(1, 0, 2, 3)  # t,c,h,w
            video = (video + 1.0) / 2.0
            video = (video * 255).to(torch.uint8).permute(0, 2, 3, 1)
            savepath = os.path.join(sub_savedir, filename)
            torchvision.io.write_video(
                savepath, video, fps=fps, video_codec="h264", options={"crf": "10"}
            )
