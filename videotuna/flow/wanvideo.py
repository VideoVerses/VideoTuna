import torch
from loguru import logger
import random
import os
import math
import torch.distributed as dist
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from PIL import Image
from datetime import datetime
import sys
from omegaconf import OmegaConf, DictConfig

from videotuna.base.generation_base import GenerationBase
from videotuna.utils.common_utils import instantiate_from_config
from videotuna.utils.args_utils import VideoMode
import videotuna.models.wan.wan as wan
from videotuna.models.wan.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from videotuna.models.wan.wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from videotuna.models.wan.wan.utils.utils import cache_video, cache_image, str2bool

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "inputs/i2v/576x1024/i2v_input.JPG",
    },
}

class WanVideoModelFlow(GenerationBase):
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
        task: str = "t2v-14B",            
        ckpt_path: Optional[str] = None,    
        offload_model: Optional[bool] = None, 
        ulysses_size: int = 1,             
        ring_size: int = 1,               
        t5_fsdp: bool = False,           
        t5_cpu: bool = False,             
        dit_fsdp: bool = False,            
        use_prompt_extend: bool = False,   
        prompt_extend_method: str = "local_qwen",  
        prompt_extend_model: Optional[str] = None,  
        prompt_extend_target_lang: str = "zh",    
        seed: int = -1,
        *args, **kwargs
    ):
        logger.info("WanVideo flow: starting init")
        assert ckpt_path is not None, "Please specify the checkpoint directory."
        assert task in WAN_CONFIGS, f"Unsupport task: {task}"
        assert task in EXAMPLE_PROMPT, f"Unsupport task: {task}"
        super().__init__(
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            denoiser_config=denoiser_config,
            scheduler_config=scheduler_config,
            cond_stage_2_config=cond_stage_2_config,
            lora_config=lora_config,
            trainable_components=[]
        )
        logger.info("WanVideo flow: class init finished")
        self.task = task
        self.ckpt_path = ckpt_path
        self.use_prompt_extend = use_prompt_extend
        self.prompt_extend_model = prompt_extend_model
        self.prompt_extend_target_lang = prompt_extend_target_lang
        self.seed = seed
        self.offload_model = offload_model
        self.ulysses_size = ulysses_size
        self.ring_size = ring_size

        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank

        if offload_model is None:
            offload_model = False if world_size > 1 else True
            logger.info(
                f"offload_model is not specified, set to {offload_model}.")
        if world_size > 1:
            pass
            # torch.cuda.set_device(local_rank)
            # dist.init_process_group(
            #     backend="nccl",
            #     init_method="env://",
            #     rank=rank,
            #     world_size=world_size)
            # logger.info("WanVideo flow: Init Process Group")
        else:
            assert not (
                t5_fsdp or dit_fsdp
            ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
            assert not (
                ulysses_size > 1 or ring_size > 1
            ), f"context parallel are not supported in non-distributed environments."
        
        if ulysses_size > 1 or ring_size > 1:
            assert ulysses_size * ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
            from xfuser.core.distributed import (initialize_model_parallel,
                                                init_distributed_environment)
            init_distributed_environment(
                rank=dist.get_rank(), world_size=dist.get_world_size())

            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=ring_size,
                ulysses_degree=ulysses_size,
            )
            logger.info("WanVideo flow: Init Ring/Ulysses Seqeunce Parallel Process Group")

        if use_prompt_extend:
            if prompt_extend_method == "dashscope":
                self.prompt_expander = DashScopePromptExpander(
                    model_name=prompt_extend_model, is_vl="i2v" in task)
            elif prompt_extend_method == "local_qwen":
                self.prompt_expander = QwenPromptExpander(
                    model_name=prompt_extend_model,
                    is_vl="i2v" in task,
                    device=rank)
            else:
                raise NotImplementedError(
                    f"Unsupport prompt_extend_method: {prompt_extend_method}")
            logger.info("WanVideo flow: Set Prompt Extention")

        cfg = WAN_CONFIGS[task]
        self.cfg = cfg
        if ulysses_size > 1:
            assert cfg.num_heads % ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{ulysses_size=}`."

        logger.info(f"WanVideo flow: model config: {cfg}")

        if dist.is_initialized():
            seed = [seed] if rank == 0 else [None]
            dist.broadcast_object_list(seed, src=0)
            seed = seed[0]
            logger.info(f"WanVideo flow: broadcast seed")
        

        if "t2v" in task or "t2i" in task:
            logger.info("Creating WanT2V pipeline.")
            self.wan_t2v = wan.WanT2V(
                config=cfg,
                checkpoint_dir=ckpt_path,
                device_id=device,
                rank=rank,
                t5_fsdp=t5_fsdp,
                dit_fsdp=dit_fsdp,
                use_usp=(ulysses_size > 1 or ring_size > 1),
                t5_cpu=t5_cpu,
                first_stage_model=self.first_stage_model,
                cond_stage_model=self.cond_stage_model,
                denoiser=self.denoiser
            )
        else:
            logger.info("Creating WanI2V pipeline.")
            self.wan_i2v = wan.WanI2V(
                config=cfg,
                checkpoint_dir=ckpt_path,
                device_id=device,
                rank=rank,
                t5_fsdp=t5_fsdp,
                dit_fsdp=dit_fsdp,
                use_usp=(ulysses_size > 1 or ring_size > 1),
                t5_cpu=t5_cpu,
                first_stage_model=self.first_stage_model,
                cond_stage_model=self.cond_stage_model,
                cond_stage_2_model=self.cond_stage_2_model,
                denoiser=self.denoiser
            )
        
    def _validate_args(self, args):        
        # Size reassign and check
        args.size = f"{args.width}*{args.height}"
        logger.info(f"setting size = width*height == {args.size}")
        assert args.size in SUPPORTED_SIZES[
            self.task], f"Unsupport size {args.size} for task {self.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[self.task])}"

    def inference_t2v(self, args: DictConfig):
        # init vars
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank

        frames = args.frames
        size = args.size
        sample_shift = args.time_shift
        sample_solver = args.solver
        sampling_steps = args.num_inference_steps
        guide_scale = args.unconditional_guidance_scale

        # load input
        prompt_list = self.load_inference_inputs(args.prompt_file, args.mode)
        if len(prompt_list) > 1:
            logger.warning("WanVideo currently does not support batch inference, we will run sample at a time")
        
        videos = []
        gpu = []
        time = []
        for prompt in prompt_list:
            logger.info(f"Input prompt: {prompt}")
            if self.use_prompt_extend:
                logger.info("Extending prompt ...")
                if rank == 0:
                    prompt_output = self.prompt_expander(
                        prompt,
                        tar_lang=self.prompt_extend_target_lang,
                        seed=self.seed)
                    if prompt_output.status == False:
                        logger.info(
                            f"Extending prompt failed: {prompt_output.message}")
                        logger.info("Falling back to original prompt.")
                        input_prompt = prompt
                    else:
                        input_prompt = prompt_output.prompt
                    input_prompt = [input_prompt]
                else:
                    input_prompt = [None]
                if dist.is_initialized():
                    dist.broadcast_object_list(input_prompt, src=0)
                prompt = input_prompt[0]
                logger.info(f"Extended prompt: {prompt}")

            logger.info(
                f"Generating {'image' if 't2i' in self.task else 'video'} ...")
            result_with_metrics = self.wan_t2v.generate(
                prompt,
                size=SIZE_CONFIGS[size],
                frame_num=frames,
                shift=sample_shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=self.seed,
                offload_model=self.offload_model)
            video = result_with_metrics['result']
            videos.append(video)

            gpu.append(result_with_metrics.get('gpu', -1.0))
            time.append(result_with_metrics.get('time', -1.0))

        if rank == 0:
            logger.info("Saving videos")
            filenames = self.process_savename(prompt_list, args.n_samples_prompt)
            self.save_videos(torch.stack(videos).unsqueeze(dim=1), args.savedir, filenames, fps=args.savefps)
            self.save_metrics(gpu=gpu, time=time, config=args, savedir=args.savedir)

    def inference_i2v(self, args: DictConfig):
        # init vars
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank

        frames = args.frames
        size = args.size
        sample_shift = args.time_shift
        sample_solver = args.solver
        sampling_steps = args.num_inference_steps
        guide_scale = args.unconditional_guidance_scale

        prompt_list, image_list = self.load_inference_inputs(args.prompt_dir, args.mode)
        assert len(prompt_list) == len(image_list), "prompt and image number should match"
        
        if len(prompt_list) > 0:
            logger.warning("WanVideo currently does not support batch inference, we will run sample at a time")
            
        videos = []
        gpu = []
        time = []
        for prompt, image_path in zip(prompt_list, image_list):
            logger.info(f"Input prompt: {prompt}")
            logger.info(f"Input image: {image_path}")

            img = Image.open(image_path).convert("RGB")
            if self.use_prompt_extend:
                logger.info("Extending prompt ...")
                if rank == 0:
                    prompt_output = self.prompt_expander(
                        prompt,
                        tar_lang=self.prompt_extend_target_lang,
                        image=img,
                        seed=self.seed)
                    if prompt_output.status == False:
                        logger.info(
                            f"Extending prompt failed: {prompt_output.message}")
                        logger.info("Falling back to original prompt.")
                        input_prompt = prompt
                    else:
                        input_prompt = prompt_output.prompt
                    input_prompt = [input_prompt]
                else:
                    input_prompt = [None]
                if dist.is_initialized():
                    dist.broadcast_object_list(input_prompt, src=0)
                prompt = input_prompt[0]
                logger.info(f"Extended prompt: {prompt}")


            logger.info("Generating video ...")
            result_with_metrics = self.wan_i2v.generate(
                prompt,
                img,
                max_area=MAX_AREA_CONFIGS[size],
                frame_num=frames,                                  
                shift=sample_shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=self.seed,
                offload_model=self.offload_model)
            
            video = result_with_metrics['result']
            video = video.cpu()
            videos.append(video)
            gpu.append(result_with_metrics.get('gpu', -1.0))
            time.append(result_with_metrics.get('time', -1.0))
            del result_with_metrics
            
        if rank == 0:
            logger.info("Saving videos")
            filenames = self.process_savename(prompt_list, args.n_samples_prompt)
            self.save_videos(torch.stack(videos).unsqueeze(dim=1), args.savedir, filenames, fps=args.savefps)
            self.save_metrics(gpu=gpu, time=time, config=args, savedir=args.savedir)

    @torch.no_grad()
    def inference(self, args: DictConfig): 
        # check input  
        self._validate_args(args) 

        # t2v mode
        if args.mode == VideoMode.T2V.value:  
            self.inference_t2v(args)
        # i2v mode
        elif args.mode == VideoMode.I2V.value:
            self.inference_i2v(args)
        else:
            raise ValueError("Error: invalid mode, we currently only support t2v and i2v for wanvideo")

    def from_pretrained(self,
                        ckpt_path: Optional[Union[str, Path]] = None,
                        denoiser_ckpt_path: Optional[Union[str, Path]] = None,
                        lora_ckpt_path: Optional[Union[str, Path]] = None,
                        ignore_missing_ckpts: bool = False):
        if "t2v" in self.task or "t2i" in self.task:
            self.wan_t2v.load_weight()
            #this is only used to load trained denoiser_ckpt_path, 
            #so we set ignore missing ckpts avoid duplicate loading
            self.load_denoiser(ckpt_path, denoiser_ckpt_path, True)
        else:
            self.wan_i2v.load_weight()
            self.load_denoiser(ckpt_path, denoiser_ckpt_path, True)
    
    def enable_vram_management(self):
        if "t2v" in self.task or "t2i" in self.task:
            self.wan_t2v.enable_vram_management()
        else:
            self.wan_i2v.enable_vram_management()
    
    def training_step(self, batch, batch_idx):
        #self.first_stage_model.disable_cache()
        if "t2v" in self.task or "t2i" in self.task:
            loss = self.wan_t2v.training_step(batch, batch_idx, self.first_stage_key, self.cond_stage_key)
        else:
            loss = self.wan_i2v.training_step(batch, batch_idx, self.first_stage_key, self.cond_stage_key)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss
    
    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        pass