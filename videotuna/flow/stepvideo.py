import torch
import logging
import os
import torch.distributed as dist
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from PIL import Image
from datetime import datetime
import sys
import asyncio
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

from videotuna.base.generation_base import GenerationBase
from videotuna.utils.common_utils import instantiate_from_config
from videotuna.schedulers.flow_matching import FlowMatchScheduler


from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from loguru import logger
import numpy as np
import pickle
import torch, copy
from transformers.models.bert.modeling_bert import BertEmbeddings
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput

from ..utils.common_utils import monitor_resources
from videotuna.utils.inference_utils import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from videotuna.models.stepvideo.stepvideo.modules.model import StepVideoModel
from videotuna.models.stepvideo.stepvideo.diffusion.scheduler import FlowMatchDiscreteScheduler
from videotuna.models.stepvideo.stepvideo.utils import VideoProcessor, with_empty_init
from videotuna.models.stepvideo.stepvideo.modules.model import RMSNorm
from videotuna.models.stepvideo.stepvideo.vae.vae import CausalConv, CausalConvAfterNorm, Upsample2D
from videotuna.models.stepvideo.stepvideo.parallel import initialize_parall_group, get_parallel_group
from xfuser.model_executor.models.customized.step_video_t2v.tp_applicator import TensorParallelApplicator
from xfuser.core.distributed.parallel_state import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank


class StepVideoModelFlow(GenerationBase):
    """
    Training and inference flow for YourModel.
    
    This model inherits from GenerationFlow, which is a base class for all generative models.
    """

    def __init__(
        self,
        first_stage_config: Dict[str, Any],
        cond_stage_config: Dict[str, Any],
        denoiser_config: Dict[str, Any],
        scheduler_config: Optional[Dict[str, Any]] = None,
        cond_stage_2_config: Optional[Dict[str, Any]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        ring_degree: int = 1,
        ulysses_degree: int = 1,
        tensor_parallel_degree: int = 1,
        scale_factor: float = 1.0,
        num_persistent_param_in_dit: int = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = torch.cuda.current_device(),
        enable_model_cpu_offload: bool = True,
        enable_sequential_cpu_offload: bool = False,
        *args, **kwargs
    ):
        logger.info("StepVideoModelFlow: init workflow")
        if tensor_parallel_degree > 1:
            logger.info("StepVideoModelFlow: init tensor parallel group")
            initialize_parall_group(ring_degree=ring_degree, ulysses_degree=ulysses_degree, tensor_parallel_degree=tensor_parallel_degree)
        super().__init__(
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            denoiser_config=denoiser_config,
            scheduler_config=scheduler_config,
            cond_stage_2_config=cond_stage_2_config,
            lora_config=lora_config,
            trainable_components=[]
        )
        
        self.ring_degree = ring_degree
        self.ulysses_degree = ulysses_degree
        self.tensor_parallel_degree = tensor_parallel_degree
        self.torch_dtype = torch_dtype
        self.device_type = device
        self.vae_scale_factor_temporal = self.vae.temporal_compression_ratio if getattr(self, "vae", None) else 8
        self.vae_scale_factor_spatial = self.vae.spatial_compression_ratio if getattr(self, "vae", None) else 16
        self.scale_factor = scale_factor
        self.num_persistent_param_in_dit = num_persistent_param_in_dit
        self.enable_sequential_cpu_offload = enable_sequential_cpu_offload
        self.enable_model_cpu_offload = enable_model_cpu_offload

    def load_lib(self, ckpt_path: str):
        logger.info(f"loading lib from {ckpt_path}")
        accepted_version = {
            '2.2': 'liboptimus_ths-torch2.2-cu121.cpython-310-x86_64-linux-gnu.so',
            '2.3': 'liboptimus_ths-torch2.3-cu121.cpython-310-x86_64-linux-gnu.so',
            '2.5': 'liboptimus_ths-torch2.5-cu124.cpython-310-x86_64-linux-gnu.so',
        }
        try:
            version = '.'.join(torch.__version__.split('.')[:2])
            if version in accepted_version:
                logger.info(f"cur dir: {os.getcwd()}")
                library = os.path.join(ckpt_path, f'lib/{accepted_version[version]}')
                logger.info(f"loading lib from {library}")
                torch.ops.load_library(library)
                logger.info(f"{library} loaded")
            else:
                raise ValueError("Not supported torch version for liboptimus")
        except Exception as err:
            print(err)

    def enable_vram_management(self):
        logger.info("StepVideoModelFlow: start enable_vram_management")
        dtype = next(iter(self.cond_stage_2_model.parameters())).dtype
        logger.info(f"cond_stage_2_model param dtype: {dtype}")
        #use enable_model_cpu_offload as default
        onload_device = self.device_type
        if self.enable_sequential_cpu_offload:
            onload_device = 'cpu'
        elif self.enable_model_cpu_offload:
            onload_device = self.device_type

        enable_vram_management(
            self.cond_stage_2_model,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                BertEmbeddings: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=onload_device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device_type,
            ),
        )
        dtype = next(iter(self.cond_stage_model.parameters())).dtype
        logger.info(f"cond_stage_model param dtype: {dtype}")
        enable_vram_management(
            self.cond_stage_model,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                RMSNorm: AutoWrappedModule,
                torch.nn.Embedding: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=onload_device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device_type,
            ),
        )
        dtype = next(iter(self.denoiser.parameters())).dtype
        logger.info(f"denoiser param dtype: {dtype}")
        enable_vram_management(
            self.denoiser,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=onload_device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device_type,
            ),
            max_num_param=self.num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=onload_device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device_type,
            ),
        )
        dtype = next(iter(self.first_stage_model.parameters())).dtype
        logger.info(f"first_stage_model param dtype: {dtype}")
        enable_vram_management(
            self.first_stage_model,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                CausalConv: AutoWrappedModule,
                CausalConvAfterNorm: AutoWrappedModule,
                Upsample2D: AutoWrappedModule
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=onload_device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device_type,
            ),
        )
        self.enable_cpu_offload()
        logger.info("StepVideoModelFlow: end enable_vram_management")

    
    def encode_prompt(
        self,
        input_prompt: str,
        neg_magic: str = '',
        pos_magic: str = '',
    ):
        prompts = [input_prompt + pos_magic]
        bs = len(prompts)
        prompts += [neg_magic] * bs
        
        prompt_embeds, prompt_embeds_mask = self.cond_stage_model(prompts)
        clip_embedding, _ = self.cond_stage_2_model(prompts)
        
        len_clip = clip_embedding.shape[1]
        prompt_embeds_mask = torch.nn.functional.pad(prompt_embeds_mask, (len_clip, 0), value=1)   ## pad attention_mask with clip's length 

        return prompt_embeds, clip_embedding, prompt_embeds_mask

    def check_inputs(self, num_frames, width, height):
        num_frames = max(num_frames//17*17, 1)
        width = max(width//16*16, 16)
        height = max(height//16*16, 16)
        return num_frames, width, height

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: 64,
        height: int = 544,
        width: int = 992,
        num_frames: int = 204,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_frames, width, height = self.check_inputs(num_frames, width, height)
        shape = (
            batch_size,
            max(num_frames//17*3, 1),
            num_channels_latents,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )   # b,f,c,h,w
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if generator is None:
            generator = torch.Generator(device=device)

        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return latents


    @torch.inference_mode()
    def inference(self, config: DictConfig, device=torch.cuda.current_device()):
        # init vars
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank
    
        # load input
        prompt_list = self.load_inference_inputs(config.prompt_file, config.mode)
        if len(prompt_list) > 1:
            logger.warning("Stepvideo currently does not support batch inference, we will sample at a time")
        
        videos = []
        gpu = []
        time = []
        for prompt in prompt_list:
            if rank == 0:
                result_with_metrics = self.single_inference(prompt, config)
                video  = result_with_metrics['result']
                videos.append(video)
                gpu.append(result_with_metrics.get('gpu', -1.0))
                time.append(result_with_metrics.get('time', -1.0))
        
        if rank == 0:
            logger.info("Saving videos")
            filenames = self.process_savename(prompt_list, config.n_samples_prompt)
            processor = VideoProcessor(config.savedir)
            for video, filename in zip(videos, filenames):
                processor.postprocess_video(video, filename)
            self.save_metrics(gpu=gpu, time=time, config=config, savedir=config.savedir)
        
    
    @monitor_resources(return_metrics=True)
    def single_inference(self, prompt, config: DictConfig):
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank

        neg_magic = config.uncond_prompt
        pos_magic = config.pos_prompt
        batch_size = config.bs
        time_shift = config.time_shift
        num_inference_steps = config.num_inference_steps
        unconditional_guidance_scale = config.unconditional_guidance_scale
        do_classifier_free_guidance = unconditional_guidance_scale > 1.0
        # 3. Encode input prompt
        logger.info("loading cond_stage_model and cond_stage_2_model")
        self.load_models_to_device(['cond_stage_model', 'cond_stage_2_model'])

        logger.info("encoding prompt")
        prompt_embeds, prompt_embeds_2, prompt_attention_mask = self.encode_prompt(
            input_prompt=prompt,
            neg_magic=neg_magic,
            pos_magic=pos_magic
        )

        denoiser_dtype = self.denoiser.dtype
        prompt_embeds = prompt_embeds.to(denoiser_dtype).to(device)
        prompt_attention_mask = prompt_attention_mask.to(denoiser_dtype).to(device)
        prompt_embeds_2 = prompt_embeds_2.to(denoiser_dtype).to(device)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            time_shift=time_shift,
            device=device
        )

        # 5. Prepare latent variables
        logger.info("preparing latents")
        num_channels_latents = self.denoiser.config.in_channels
        latents = self.prepare_latents(
            batch_size * config.n_samples_prompt,
            num_channels_latents,
            config.height,
            config.width,
            config.frames,
            torch.bfloat16,
            device,
            torch.Generator(device=device).manual_seed(config.seed)
        ).to(device)

        # 7. Denoising loop
        logger.info("loading denoiser")
        self.load_models_to_device(['denoiser'])

        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(denoiser_dtype)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype).to(device)

                noise_pred = self.denoiser(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    encoder_hidden_states_2=prompt_embeds_2,
                    return_dict=False,
                )
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + unconditional_guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=latents
                )
                
                progress_bar.update()

        if not torch.distributed.is_initialized() or int(torch.distributed.get_rank())==0:
            self.load_models_to_device(['first_stage_model'])
            video = self.first_stage_model.decode(latents.to(denoiser_dtype).to(device) / self.scale_factor)
            return video


    def from_pretrained(self,
                        ckpt_path: Optional[Union[str, Path]] = None,
                        denoiser_ckpt_path: Optional[Union[str, Path]] = None,
                        lora_ckpt_path: Optional[Union[str, Path]] = None,
                        ignore_missing_ckpts: bool = False):
        logger.info("StepVideoModelFlow: start load weight")
        self.load_lib(ckpt_path)
        self.first_stage_model.load_weight()
        self.cond_stage_2_model.load_weight()
        logger.info("StepVideoModelFlow: end load weight")

        if self.tensor_parallel_degree > 1:
            logger.info("StepVideoModelFlow: apply tensor parallel")
            tp_applicator = TensorParallelApplicator(get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank())
            tp_applicator.apply_to_model(self.denoiser)    
        
    def training_step(self, batch, batch_idx):
        model_offload: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda"
        first_stage_key = self.first_stage_key
        cond_stage_key = self.cond_stage_key

        if model_offload:
            self.first_stage_model.to(device)
        latents = torch.stack(self.first_stage_model.encode(batch[first_stage_key])).to(dtype=dtype, device=device).detach()
        if model_offload:
            self.first_stage_model.to('cpu')
            self.cond_stage_model.to(device)
        text_cond_embed, text_cond_embed_mask  = self.cond_stage_model(batch[cond_stage_key], device)
        if model_offload:
            self.cond_stage_model.to('cpu')

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
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss
    
    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        pass