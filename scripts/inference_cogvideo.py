import os
import sys
import time
import math
import argparse
import numpy as np
from functools import partial
from tqdm import trange, tqdm
from omegaconf import OmegaConf
from omegaconf import ListConfig
from einops import rearrange, repeat
from typing import List, Union
import torch
from pytorch_lightning import seed_everything
import imageio
sys.path.insert(0, os.getcwd())
from src.lvdm.samplers.ddim import DDIMSampler
from utils.common_utils import instantiate_from_config
from src.lvdm.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from src.cogvideo.arguments import get_args
from scripts.inference_utils import (
    load_model_checkpoint, 
    load_prompts, 
    load_inputs_i2v, 
    load_image_batch, 
    sample_batch_t2v, 
    sample_batch_i2v,
    save_videos,
)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="t2v", type=str, help="inference mode: t2v/i2v", choices=["t2v", "i2v"])
    #
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="model config (yaml) path")
    parser.add_argument("--prompt_file", type=str, default=None, help="a text file containing many prompts for text-to-video")
    parser.add_argument("--prompt_dir", type=str, default=None, help="a input dir containing images and prompts for image-to-video/interpolation")
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    #
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    #
    parser.add_argument("--height", type=int, default=512, help="video height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="video width, in pixel space")
    parser.add_argument("--frames", type=int, default=None, help="video frame number, in pixel space")
    parser.add_argument("--fps", type=int, default=24, help="video motion speed. 512 or 1024 model: large value -> slow motion; 256 model: large value -> large motion;")
    parser.add_argument("--n_samples_prompt", type=int, default=1, help="num of samples per prompt",)
    #
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--uncond_prompt", type=str, default="", help="unconditional prompts, or negative prompts")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    # dc args
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False, help="i2v: use multi-condition cfg or not")
    parser.add_argument("--cfg_img", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--timestep_spacing", type=str, default="uniform", help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.")
    parser.add_argument("--guidance_rescale", type=float, default=0.0, help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--loop", action='store_true', default=False, help="generate looping videos or not")
    #
    parser.add_argument("--savefps", type=float, default=10, help="video fps to generate")
    return parser

def load_model(args, cuda_idx=0):
    """
    Create model and load weight.
    """
    # build model
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda(cuda_idx)
    # load weights
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    print(args.ckpt_path)
    # customized checkpoint loader 
    try:
        # pl chekcpoint 
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(args.ckpt_path)['state_dict'], strict=False)
    except:
        # pretrained checkpoint 
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(args.ckpt_path)['module'], strict=False)
        
    if len(unexpected_keys) > 0:
        print_rank0(
            f'Will continue but found unexpected_keys! Check whether you are loading correct checkpoints: {unexpected_keys}.')
    model.eval()
    return model

def load_inputs(args):
    """
    load inputs:
        t2v: prompts
        i2v: prompts + images
    """
    assert (args.prompt_file is not None or args.prompt_dir is not None), "Error: input file/dir NOT Found!"
    
    if args.prompt_file is not None:
        assert(os.path.exists(args.prompt_file))
        # load inputs for t2v
        prompt_list = load_prompts(args.prompt_file)
        num_prompts = len(prompt_list)
        filename_list = [f"prompt-{idx+1:04d}" for idx in range(num_prompts)]
        image_list = None
    elif args.prompt_dir is not None:
        assert(os.path.exists(args.prompt_dir))
        # load inputs for i2v
        filename_list, image_list, prompt_list = load_inputs_i2v(
                                                args.prompt_dir, 
                                                video_size=(args.height, args.width), 
                                                video_frames=args.frames, 
                                                )
    return prompt_list, image_list, filename_list
def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            # import pdb;pdb.set_trace()
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc
def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))
def save_video_as_grid_and_mp4(video_batch: torch.Tensor, 
                               save_path: str, 
                               filenames=None ,
                               fps: int = 5
                               ):
    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = os.path.join(save_path, filenames[i]+f"-{i}.mp4")
        print(now_save_path)
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)

def run_inference(args, gpu_num=1, rank=0, **kwargs):
    """
    Inference t2v/i2v models
    """
    
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    
    seed_everything(args.seed)
    os.makedirs(args.savedir, exist_ok=True)
    # import pdb;pdb.set_trace()
    # load model, sampler, inputs
    model = load_model(args)
    # cogvideo doesn't need to be wrapped by DDIM 
    
    # args.frames = model.temporal_length if args.frames is None else args.frames
    prompt_list, image_list, filename_list = load_inputs(args)

    # split across multiple gpus
    num_samples = len(prompt_list)
    num_samples_rank = num_samples // gpu_num
    remainder = num_samples % gpu_num
    indices_rank = list(range(num_samples_rank*rank, num_samples_rank*(rank+1)))
    if rank == 0 and remainder != 0:
        indices_rank = indices_rank + list(range(num_samples-remainder, num_samples))
    #
    prompt_list_rank = [prompt_list[i] for i in indices_rank]
    filename_list_rank = [filename_list[i] for i in indices_rank]

    # noise shape
    # image_size = [480, 720]
    # h, w, frames, channels = args.height // 8, args.width // 8, args.frames, model.channels
    frames, h , w, channels , F = args.frames,  args.height,  args.width , 16 , 8
    device = torch.device("cuda", rank)
    # -----------------------------------------------------------------
    # inference
    start = time.time()
    n_iters = len(prompt_list_rank) // args.bs + (1 if len(prompt_list_rank) % args.bs else 0)
    with torch.no_grad(), torch.cuda.amp.autocast():
        for idx in trange(0, n_iters, desc="Sample Iters"):
            # print(f'[rank:{rank}] batch {idx}: prompt bs {args.bs}) x nsamples_per_prompt {args.n_samples_prompt} ...')
            prompts = prompt_list_rank[idx*args.bs:(idx+1)*args.bs]
            filenames = filename_list_rank[idx*args.bs:(idx+1)*args.bs]
            model.to(device)
            print(f"{len(prompts)},{type(prompts)}",prompts[0])
            value_dict = {
                "prompt": prompts,
                "negative_prompt": ["" for p in prompts],
                "num_frames": torch.tensor(frames).unsqueeze(0),
            }
            force_uc_zero_embeddings = ["txt"]
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, [args.n_samples_prompt]
            )
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    print(key, batch[key].shape)
                elif isinstance(batch[key], list):
                    print(key, [len(l) for l in batch[key]])
                else:
                    print(key, batch[key])
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )
            # import pdb;pdb.set_trace()
            samples_z = model.sample(
                c,
                uc=uc,
                batch_size=1,
                shape=(frames, channels, h // F, w // F),
            )
            samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

            torch.cuda.empty_cache()
            first_stage_model = model.first_stage_model
            first_stage_model = first_stage_model.to(device)

            latent = 1.0 / model.scale_factor * samples_z

            # Decode latent serial to save GPU memory
            recons = []
            loop_num = (frames - 1) // 2
            for i in range(loop_num):
                if i == 0:
                    start_frame, end_frame = 0, 3
                else:
                    start_frame, end_frame = i * 2 + 1, i * 2 + 3
                if i == loop_num - 1:
                    clear_fake_cp_cache = True
                else:
                    clear_fake_cp_cache = False
                with torch.no_grad():
                    # print(latent[:, :, start_frame:end_frame].contiguous().shape)
                    # import pdb;pdb.set_trace()
                    # latent = torch.permute(latent, (0, 2, 1, 3, 4)).contiguous()
                    # print(latent.shape)
                    recon = first_stage_model.decode(
                        latent[:, :, start_frame:end_frame].contiguous(), clear_fake_cp_cache=clear_fake_cp_cache
                    )
                recons.append(recon)
                recon = torch.cat(recons, dim=2).to(torch.float32)
                # print(recon.shape)
            model.to("cpu")
            samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()
            # import pdb;pdb.set_trace()
            save_video_as_grid_and_mp4(samples, args.savedir,filenames=filenames, fps=args.savefps)
            # save_videos(batch_samples, args.savedir, filenames, fps=args.savefps)
    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


if __name__ == "__main__":
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    # py_parser = argparse.ArgumentParser(add_help=False)
    # known, args_list = py_parser.parse_known_args()
    args = get_parser().parse_args()
    # args = get_args(args_list)
    # # args = argparse.Namespace(**vars(args), **vars(known))
    # del args.deepspeed_config
    print(args)
    # args.model_config.first_stage_config.params.cp_size = 1
    # args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    # args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    # args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False
    args.height = 480
    args.width = 720
    # args.savedir = args.output_dir
    # args.savefps = args.sampling_fps
    # args.frames = args.sampling_num_frames
    # args.config = "/home/liurt/liurt_data/haoyu/VideoTuna/configs/inference/cogvideo_t2v_pl.yaml"
    # args.ckpt_path = args.load
    # args.prompt_file = args.input_file
    args.bs = 1
    args.n_samples_prompt = 1
    args.seed = 11111
    run_inference(args)
