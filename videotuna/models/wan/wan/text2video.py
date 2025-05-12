# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
from loguru import logger
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
from typing import Union
from pathlib import Path

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm
from typing import Optional

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5Encoder, T5EncoderModel
from .modules.vae import WanVAE, WanVAE_
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ....utils.common_utils import monitor_resources
from ....schedulers.flow_matching import FlowMatchScheduler

class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        first_stage_model: WanVAE_= None ,
        cond_stage_model:T5Encoder=None,
        denoiser: WanModel=None,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.t5_fsdp = t5_fsdp
        self.dit_fsdp = dit_fsdp
        self.use_usp = use_usp
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        #encoder
        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder : T5EncoderModel = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
            model=cond_stage_model)

        #vae
        self.vae: WanVAE = WanVAE(vae=first_stage_model, 
                                  vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint), 
                                  device=self.device)
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size

        #denoiser
        self.model: WanModel = denoiser
        self.shard_fn = shard_fn
        self.sample_neg_prompt = config.sample_neg_prompt

    @monitor_resources(return_metrics=True)
    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def load_weight(self):
        self.text_encoder.load_weight()
        self.vae.load_weight()
        #denoiser use from_pretrained, no need load again
        if self.use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if self.dit_fsdp:
            self.model = self.shard_fn(self.model)
        else:
            self.model = self.model.to(self.device)

    def enable_vram_management(self):
        pass

    def get_seq_len(self, frames:int=81, width:int=1280, height:int=720):
        target_shape = (self.vae.model.z_dim, (frames - 1) // self.vae_stride[0] + 1,
                        height // self.vae_stride[1],
                        width // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size
        return seq_len
    
    def training_step(self, batch, batch_idx, 
                      first_stage_key:str, 
                      cond_stage_key:str,
                      model_offload:bool = True,
                      dtype:torch.dtype = torch.bfloat16,
                      device:str = "cuda"):
        with torch.no_grad():
            if not model_offload:
                latents = torch.stack(self.vae.encode(batch[first_stage_key])).to(dtype=dtype, device=device).detach()
                text_cond_embed = self.text_encoder(batch[cond_stage_key], device)
            else:
                self.vae.model.to(device)
                latents = torch.stack(self.vae.encode(batch[first_stage_key])).to(dtype=dtype, device=device).detach()
                self.vae.model.to('cpu')
                self.text_encoder.model.to(device)
                text_cond_embed = self.text_encoder(batch[cond_stage_key], device)
                self.text_encoder.model.to('cpu')

        ## scheduler
        self.scheduler : FlowMatchScheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

        ## noise
        B = len(latents)
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=dtype, device=device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep).to(dtype=dtype, device=device)
        training_target = noise.to(device) - latents

        # compute loss
        noise_pred = self.model(x=noisy_latents, t=timestep, context=text_cond_embed, seq_len=None)
        loss = torch.nn.functional.mse_loss(torch.stack(noise_pred).float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep).to(device=device)
        return loss