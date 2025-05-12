import os
import time
import random
import functools
from typing import List, Optional, Tuple, Union, Dict, Any
from omegaconf import DictConfig

from pathlib import Path
from loguru import logger

import torch
import torch.distributed as dist
from videotuna.models.hunyuan.hyvideo_i2v.constants import PROMPT_TEMPLATE, NEGATIVE_PROMPT, PRECISION_TO_TYPE, NEGATIVE_PROMPT_I2V
from videotuna.models.hunyuan.hyvideo_i2v.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3DWrapper
from videotuna.models.hunyuan.hyvideo_i2v.modules.models import HYVideoDiffusionTransformerWrapper
from videotuna.models.hunyuan.hyvideo_i2v.text_encoder import TextEncoder, TextEncoderWrapper
from videotuna.models.hunyuan.hyvideo_i2v.utils.data_utils import align_to, get_closest_ratio, generate_crop_size_list
from videotuna.models.hunyuan.hyvideo_i2v.utils.lora_utils import load_lora_for_pipeline
from videotuna.models.hunyuan.hyvideo_i2v.modules.posemb_layers import get_nd_rotary_pos_embed
from videotuna.models.hunyuan.hyvideo_i2v.modules.fp8_optimization import convert_fp8_linear
from videotuna.models.hunyuan.hyvideo_i2v.diffusion.schedulers import FlowMatchDiscreteScheduler
from videotuna.models.hunyuan.hyvideo_i2v.diffusion.pipelines import HunyuanVideoPipeline
from videotuna.models.hunyuan.hyvideo_i2v.utils.file_utils import save_videos_grid
from videotuna.base.generation_base import GenerationBase
from videotuna.utils.common_utils import monitor_resources
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from safetensors.torch import load_file

try:
    import xfuser
    from xfuser.core.distributed import (
        get_sequence_parallel_world_size,
        get_sequence_parallel_rank,
        get_sp_group,
        initialize_model_parallel,
        init_distributed_environment
    )
except:
    xfuser = None
    get_sequence_parallel_world_size = None
    get_sequence_parallel_rank = None
    get_sp_group = None
    initialize_model_parallel = None
    init_distributed_environment = None


###############################################
# 20250308 pftq: Riflex workaround to fix 192-frame-limit bug, credit to Kijai for finding it in ComfyUI and thu-ml for making it
# https://github.com/thu-ml/RIFLEx/blob/main/riflex_utils.py
from diffusers.models.embeddings import get_1d_rotary_pos_embed
import numpy as np
from typing import Union,Optional
def get_1d_rotary_pos_embed_riflex(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
):
    """
    RIFLEx: Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        k (`int`, *optional*, defaults to None): the index for the intrinsic frequency in RoPE
        L_test (`int`, *optional*, defaults to None): the number of frames for inference
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    freqs = 1.0 / (
            theta ** (torch.arange(0, dim, 2, device=pos.device)[: (dim // 2)].float() / dim)
    )  # [D/2]

    # === Riflex modification start ===
    # Reduce the intrinsic frequency to stay within a single period after extrapolation (see Eq. (8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply by 0.9 to keep the extrapolated length below 90% of a single period.
    if k is not None:
        freqs[k-1] = 0.9 * 2 * torch.pi / L_test
    # === Riflex modification end ===

    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


###############################################

def parallelize_transformer(pipe):
    transformer = pipe.transformer
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        return_dict: bool = True,
    ):
        if x.shape[-2] // 2 % get_sequence_parallel_world_size() == 0:
            # try to split x by height
            split_dim = -2
        elif x.shape[-1] // 2 % get_sequence_parallel_world_size() == 0:
            # try to split x by width
            split_dim = -1
        else:
            raise ValueError(f"Cannot split video sequence into ulysses_degree x ring_degree ({get_sequence_parallel_world_size()}) parts evenly")

        # patch sizes for the temporal, height, and width dimensions are 1, 2, and 2.
        temporal_size, h, w = x.shape[2], x.shape[3] // 2, x.shape[4] // 2

        x = torch.chunk(x, get_sequence_parallel_world_size(),dim=split_dim)[get_sequence_parallel_rank()]

        dim_thw = freqs_cos.shape[-1]
        freqs_cos = freqs_cos.reshape(temporal_size, h, w, dim_thw)
        freqs_cos = torch.chunk(freqs_cos, get_sequence_parallel_world_size(),dim=split_dim - 1)[get_sequence_parallel_rank()]
        freqs_cos = freqs_cos.reshape(-1, dim_thw)
        dim_thw = freqs_sin.shape[-1]
        freqs_sin = freqs_sin.reshape(temporal_size, h, w, dim_thw)
        freqs_sin = torch.chunk(freqs_sin, get_sequence_parallel_world_size(),dim=split_dim - 1)[get_sequence_parallel_rank()]
        freqs_sin = freqs_sin.reshape(-1, dim_thw)
        
        from xfuser.core.long_ctx_attention import xFuserLongContextAttention
        
        for block in transformer.double_blocks + transformer.single_blocks:
            block.hybrid_seq_parallel_attn = xFuserLongContextAttention()

        output = original_forward(
            x,
            t,
            text_states,
            text_mask,
            text_states_2,
            freqs_cos,
            freqs_sin,
            guidance,
            return_dict,
        )

        return_dict = not isinstance(output, tuple)
        sample = output["x"]
        sample = get_sp_group().all_gather(sample, dim=split_dim)
        output["x"] = sample
        return output

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward


class HunyuanVideoFlow(GenerationBase):
    def __init__(
        self,
        first_stage_config: Dict[str, Any],
        cond_stage_config: Dict[str, Any],
        denoiser_config: Dict[str, Any],
        scheduler_config: Optional[Dict[str, Any]] = None,
        cond_stage_2_config: Optional[Dict[str, Any]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        use_cpu_offload=False,
        device=0,
        logger=None,
        #parallel
        ulysses_degree: int = 1,
        ring_degree: int = 1,
        use_fp8: bool = False,
        #lora
        use_lora: bool = False,
        lora_path: str = '',
        lora_scale: float = 1.0,
        lora_rank: int = 64,
        #path settings
        ckpt_path: str = '',
        dit_weight: str = '',
        #vae
        vae_type: str = '884-16c-hy',
        vae_tiling: bool = True,
        vae_precision: str = 'fp16',
        #i2v settings
        i2v_mode: bool = True,
        i2v_condition_type: str = 'token_replace',
        #model
        rope_theta: int = 256,
        precision: str = 'bf16',
        disable_autocast: bool = False,
        *args, **kwargs
    ):
        super().__init__(
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            denoiser_config=denoiser_config,
            scheduler_config=scheduler_config,
            cond_stage_2_config=cond_stage_2_config,
            lora_config=lora_config,
            trainable_components=[]
        )
        self.use_cpu_offload = use_cpu_offload
        self.device_type = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.vae_type = vae_type
        self.vae_tiling = vae_tiling
        self.vae_precision = vae_precision
        self.precision = precision
        self.disable_autocast = disable_autocast

        #parallel
        self.ulysses_degree = ulysses_degree
        self.ring_degree = ring_degree
        self.use_fp8 = use_fp8
        #model !!!
        self.dit_weight = dit_weight
        self.ckpt_path = ckpt_path
        self.rope_theta = rope_theta

        #i2v setting
        self.i2v_mode = i2v_mode
        self.i2v_condition_type = i2v_condition_type
        #lora config
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_path = lora_path
        self.lora_scale = lora_scale

        text_encoder : TextEncoderWrapper = self.cond_stage_model
        text_encoder_2 : TextEncoder = self.cond_stage_2_model
        model: HYVideoDiffusionTransformerWrapper = self.denoiser
        vae : AutoencoderKLCausal3DWrapper = self.first_stage_model
        self.pipeline = HunyuanVideoPipeline(
            vae=vae.vae,
            text_encoder=text_encoder.text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=model.model,
            scheduler=self.scheduler,
            progress_bar_config=logger,
            precision=precision,
            vae_precision=vae_precision,
            disable_autocast=disable_autocast
        )

        if self.i2v_mode:
            self.default_negative_prompt = NEGATIVE_PROMPT_I2V
            if self.use_lora:
                self.pipeline = load_lora_for_pipeline(
                    self.pipeline, self.lora_path, LORA_PREFIX_TRANSFORMER="Hunyuan_video_I2V_lora", alpha=self.lora_scale,
                    device=self.device_type,
                    is_parallel=(self.ulysses_degree > 1 or self.ring_degree > 1))
                logger.info(f"load lora {self.lora_path} into pipeline, lora scale is {self.lora_scale}.")
        else:
            self.default_negative_prompt = NEGATIVE_PROMPT

    def from_pretrained(self,
                        ckpt_path: Optional[Union[str, Path]] = None,
                        denoiser_ckpt_path: Optional[Union[str, Path]] = None,
                        lora_ckpt_path: Optional[Union[str, Path]] = None,
                        ignore_missing_ckpts: bool = False,
                        device: str = "cuda"):
        """
        Initialize the Inference pipeline.
    
        Args:
            pretrained_model_path (str or pathlib.Path): The model path, including t2v, text encoder and vae checkpoints.
            args (argparse.Namespace): The arguments for the pipeline.
            device (int): The device for inference. Default is None.
        """
        logger.info(f"Got text-to-video model root path: {ckpt_path}")
        
        # ========================================================================
        # Initialize Distributed Environment
        # ========================================================================
        # 20250316 pftq: Modified to extract rank and world_size early for sequential loading
        if self.ulysses_degree > 1 or self.ring_degree > 1:
            assert xfuser is not None, "Ulysses Attention and Ring Attention requires xfuser package."
            assert self.use_cpu_offload is False, "Cannot enable use_cpu_offload in the distributed environment."
            # 20250316 pftq: Set local rank and device explicitly for NCCL
            local_rank = int(os.environ['LOCAL_RANK'])
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(local_rank)  # 20250316 pftq: Set CUDA device explicitly
            dist.init_process_group("nccl")  # 20250316 pftq: Removed device_id, rely on set_device
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            assert world_size == self.ring_degree * self.ulysses_degree, \
                "number of GPUs should be equal to ring_degree * ulysses_degree."
            init_distributed_environment(rank=rank, world_size=world_size)
            initialize_model_parallel(
                sequence_parallel_degree=world_size,
                ring_degree=self.ring_degree,
                ulysses_degree=self.ulysses_degree,
            )
        else:
            rank = 0  # 20250316 pftq: Default rank for single GPU
            world_size = 1  # 20250316 pftq: Default world_size for single GPU
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
    
        torch.set_grad_enabled(False)
    
        # ========================================================================
        # Build main model, VAE, and text encoder sequentially on rank 0
        # ========================================================================
        # 20250316 pftq: Load models only on rank 0, then broadcast
        if rank == 0:
            logger.info("Building model...")
            model: HYVideoDiffusionTransformerWrapper = self.denoiser
            self.denoiser.load_weight()
            if self.use_fp8:
                convert_fp8_linear(model, self.dit_weight, original_dtype=PRECISION_TO_TYPE[self.precision])
            self.denoiser.eval()
    
            # VAE
            vae : AutoencoderKLCausal3DWrapper = self.first_stage_model
            vae.load_weight()
            s_ratio = self.first_stage_model.vae.config.spatial_compression_ratio
            t_ratio = self.first_stage_model.vae.config.time_compression_ratio
            vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}
            vae = self.first_stage_model

            #encoder
            text_encoder : TextEncoderWrapper = self.cond_stage_model
            text_encoder_2 : TextEncoder = self.cond_stage_2_model
        else:
            # 20250316 pftq: Initialize as None on non-zero ranks
            model = None
            vae = None
            vae_kwargs = None
            text_encoder = None
            text_encoder_2 = None
    
        # 20250316 pftq: Broadcast models to all ranks
        if world_size > 1:
            logger.info(f"Rank {rank}: Starting broadcast synchronization")
            dist.barrier()  # Ensure rank 0 finishes loading before broadcasting
            if rank != 0:
                # Reconstruct model skeleton on non-zero ranks
                self.denoiser : HYVideoDiffusionTransformerWrapper
                self.denoiser.load_weight()
                self.denoiser.eval()
                model = self.denoiser

                # VAE
                vae : AutoencoderKLCausal3DWrapper = self.first_stage_model
                vae.load_weight()
                s_ratio = self.first_stage_model.vae.config.spatial_compression_ratio
                t_ratio = self.first_stage_model.vae.config.time_compression_ratio
                vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}
                vae = self.first_stage_model
                vae = vae.to(device)

                #encoder
                text_encoder : TextEncoderWrapper = self.cond_stage_model.to(device)
                text_encoder_2 : TextEncoder = self.cond_stage_2_model.to(device)
                
            # Broadcast model parameters with logging
            logger.info(f"Rank {rank}: Broadcasting model parameters")
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            model.eval()
            logger.info(f"Rank {rank}: Broadcasting VAE parameters")
            for param in vae.parameters():
                dist.broadcast(param.data, src=0)
            # 20250316 pftq: Use broadcast_object_list for vae_kwargs
            logger.info(f"Rank {rank}: Broadcasting vae_kwargs")
            vae_kwargs_list = [vae_kwargs] if rank == 0 else [None]
            dist.broadcast_object_list(vae_kwargs_list, src=0)
            vae_kwargs = vae_kwargs_list[0]
            logger.info(f"Rank {rank}: Broadcasting text_encoder parameters")
            for param in text_encoder.parameters():
                dist.broadcast(param.data, src=0)
            if text_encoder_2 is not None:
                logger.info(f"Rank {rank}: Broadcasting text_encoder_2 parameters")
                for param in text_encoder_2.parameters():
                    dist.broadcast(param.data, src=0)

        if self.use_cpu_offload:
            self.pipeline.enable_sequential_cpu_offload()
        else:
            self.pipeline = self.pipeline.to(device)
    
        if self.ulysses_degree > 1 or self.ring_degree > 1:
            parallelize_transformer(self.pipeline)
        

    @staticmethod
    def parse_size(size):
        if isinstance(size, int):
            size = [size]
        if not isinstance(size, (list, tuple)):
            raise ValueError(f"Size must be an integer or (height, width), got {size}.")
        if len(size) == 1:
            size = [size[0], size[0]]
        if len(size) != 2:
            raise ValueError(f"Size must be an integer or (height, width), got {size}.")
        return size

    # 20250317 pftq: Modified to use Riflex when >192 frames
    def get_rotary_pos_embed(self, video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2  # B, C, F, H, W -> F, H, W
        model = self.pipeline.transformer
        # Compute latent sizes based on VAE type
        if "884" in self.vae_type:
            latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
        elif "888" in self.vae_type:
            latents_size = [(video_length - 1) // 8 + 1, height // 8, width // 8]
        else:
            latents_size = [video_length, height // 8, width // 8]
    
        # Compute rope sizes
        if isinstance(model.patch_size, int):
            assert all(s % model.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({model.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // model.patch_size for s in latents_size]
        elif isinstance(model.patch_size, list):
            assert all(
                s % model.patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({model.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // model.patch_size[idx] for idx, s in enumerate(latents_size)]
    
        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # Pad time axis
    
        # 20250316 pftq: Add RIFLEx logic for > 192 frames
        L_test = rope_sizes[0]  # Latent frames
        L_train = 25  # Training length from HunyuanVideo
        actual_num_frames = video_length  # Use input video_length directly
    
        head_dim = model.hidden_size // model.heads_num
        rope_dim_list = model.rope_dim_list or [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) must equal head_dim"
    
        if actual_num_frames > 192:
            k = 2+((actual_num_frames + 3) // (4 * L_train))
            k = max(4, min(8, k))
            logger.debug(f"actual_num_frames = {actual_num_frames} > 192, RIFLEx applied with k = {k}")
    
            # Compute positional grids for RIFLEx
            axes_grids = [torch.arange(size, device=self.device_type, dtype=torch.float32) for size in rope_sizes]
            grid = torch.meshgrid(*axes_grids, indexing="ij")
            grid = torch.stack(grid, dim=0)  # [3, t, h, w]
            pos = grid.reshape(3, -1).t()  # [t * h * w, 3]
    
            # Apply RIFLEx to temporal dimension
            freqs = []
            for i in range(3):
                if i == 0:  # Temporal with RIFLEx
                    freqs_cos, freqs_sin = get_1d_rotary_pos_embed_riflex(
                        rope_dim_list[i],
                        pos[:, i],
                        theta=self.rope_theta,
                        use_real=True,
                        k=k,
                        L_test=L_test
                    )
                else:  # Spatial with default RoPE
                    freqs_cos, freqs_sin = get_1d_rotary_pos_embed_riflex(
                        rope_dim_list[i],
                        pos[:, i],
                        theta=self.rope_theta,
                        use_real=True,
                        k=None,
                        L_test=None
                    )
                freqs.append((freqs_cos, freqs_sin))
                logger.debug(f"freq[{i}] shape: {freqs_cos.shape}, device: {freqs_cos.device}")
    
            freqs_cos = torch.cat([f[0] for f in freqs], dim=1)
            freqs_sin = torch.cat([f[1] for f in freqs], dim=1)
            logger.debug(f"freqs_cos shape: {freqs_cos.shape}, device: {freqs_cos.device}")
        else:
            # 20250316 pftq: Original code for <= 192 frames
            logger.debug(f"actual_num_frames = {actual_num_frames} <= 192, using original RoPE")
            freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
                rope_dim_list,
                rope_sizes,
                theta=self.rope_theta,
                use_real=True,
                theta_rescale_factor=1,
            )
            logger.debug(f"freqs_cos shape: {freqs_cos.shape}, device: {freqs_cos.device}")
    
        return freqs_cos, freqs_sin


    @monitor_resources(return_metrics=True)
    def single_inference(self, 
                         prompt, 
                         i2v_image_path, 
                         target_video_length,
                         generator,
                         config : DictConfig):
        height=config.height
        width=config.width
        video_length=config.frames
        seed=config.seed
        negative_prompt=config.uncond_prompt
        infer_steps=config.num_inference_steps
        guidance_scale=config.unconditional_guidance_scale
        flow_shift=config.time_shift
        embedded_guidance_scale=config.embedded_guidance_scale
        batch_size=config.bs
        num_videos_per_prompt=config.n_samples_prompt
        i2v_mode=config.i2v_mode
        i2v_resolution=config.i2v_resolution
        i2v_condition_type=config.i2v_condition_type
        i2v_stability=config.i2v_stability
        ulysses_degree=config.ulysses_degree
        ring_degree=config.ring_degree
        xdit_adaptive_size=config.xdit_adaptive_size
        if not isinstance(prompt, str):
            raise TypeError(f"`prompt` must be a string, but got {type(prompt)}")
        prompt = [prompt.strip()]

        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.default_negative_prompt
        if guidance_scale == 1.0:
            negative_prompt = ""
        if not isinstance(negative_prompt, str):
            raise TypeError(
                f"`negative_prompt` must be a string, but got {type(negative_prompt)}"
            )
        negative_prompt = [negative_prompt.strip()]

        img_latents = None
        semantic_images = None
        if i2v_mode:
            if i2v_resolution == "720p":
                bucket_hw_base_size = 960
            elif i2v_resolution == "540p":
                bucket_hw_base_size = 720
            elif i2v_resolution == "360p":
                bucket_hw_base_size = 480
            else:
                raise ValueError(f"i2v_resolution: {i2v_resolution} must be in [360p, 540p, 720p]")

            semantic_images = [Image.open(i2v_image_path).convert('RGB')]
            origin_size = semantic_images[0].size

            crop_size_list = generate_crop_size_list(bucket_hw_base_size, 32)
            aspect_ratios = np.array([round(float(h)/float(w), 5) for h, w in crop_size_list])
            closest_size, closest_ratio = get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)

            if ulysses_degree != 1 or ring_degree != 1:
                closest_size = (height, width)
                resize_param = min(closest_size)
                center_crop_param = closest_size

                if xdit_adaptive_size:
                    original_h, original_w = origin_size[1], origin_size[0]
                    target_h, target_w = height, width

                    scale_w = target_w / original_w
                    scale_h = target_h / original_h
                    scale = max(scale_w, scale_h)

                    new_w = int(original_w * scale)
                    new_h = int(original_h * scale)
                    resize_param = (new_h, new_w)
                    center_crop_param = (target_h, target_w)
            else:
                resize_param = min(closest_size)
                center_crop_param = closest_size

            ref_image_transform = transforms.Compose([
                transforms.Resize(resize_param),
                transforms.CenterCrop(center_crop_param),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            semantic_image_pixel_values = [ref_image_transform(semantic_image) for semantic_image in semantic_images]
            semantic_image_pixel_values = torch.cat(semantic_image_pixel_values).unsqueeze(0).unsqueeze(2).to(self.device_type)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                img_latents = self.pipeline.vae.encode(semantic_image_pixel_values).latent_dist.mode()
                img_latents.mul_(self.pipeline.vae.config.scaling_factor)

            target_height, target_width = closest_size

        freqs_cos, freqs_sin = self.get_rotary_pos_embed(
            target_video_length, target_height, target_width
        )
        n_tokens = freqs_cos.shape[0]

        debug_str = f"""
                        height: {target_height}
                        width: {target_width}
                video_length: {target_video_length}
                        prompt: {prompt}
                    neg_prompt: {negative_prompt}
                        seed: {seed}
                infer_steps: {infer_steps}
        num_videos_per_prompt: {num_videos_per_prompt}
                guidance_scale: {guidance_scale}
                    n_tokens: {n_tokens}
                    flow_shift: {flow_shift}
    embedded_guidance_scale: {embedded_guidance_scale}
                i2v_stability: {i2v_stability}"""
        if ulysses_degree != 1 or ring_degree != 1:
            debug_str += f"""
                ulysses_degree: {ulysses_degree}
                ring_degree: {ring_degree}
            xdit_adaptive_size: {xdit_adaptive_size}"""
        logger.debug(debug_str)

        samples = self.pipeline(
            prompt=prompt,
            height=target_height,
            width=target_width,
            video_length=target_video_length,
            num_inference_steps=infer_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            output_type="pil",
            freqs_cis=(freqs_cos, freqs_sin),
            n_tokens=n_tokens,
            embedded_guidance_scale=embedded_guidance_scale,
            data_type="video" if target_video_length > 1 else "image",
            is_progress_bar=True,
            vae_ver=self.vae_type,
            enable_tiling=self.vae_tiling,
            i2v_mode=i2v_mode,
            i2v_condition_type=i2v_condition_type,
            i2v_stability=i2v_stability,
            img_latents=img_latents,
            semantic_images=semantic_images,
        )[0]
        return samples

    @torch.no_grad()
    def inference(
        self,
        config : DictConfig,
        **kwargs,
    ):
        height=config.height
        width=config.width
        video_length=config.frames
        seed=config.seed
        batch_size=config.bs
        num_videos_per_prompt=config.n_samples_prompt
        out_dict = dict()

        prompt_list, image_path_list = self.load_inference_inputs(config.prompt_dir, config.mode)
        if len(prompt_list) > 1:
            logger.warning("HunyuanVideo currently does not support batch inference, we will sample at a time")
    
        # seeds
        seeds = self.set_seed(seed, batch_size, num_videos_per_prompt)
        generator = [torch.Generator(self.device_type).manual_seed(seed) for seed in seeds]
        out_dict["seeds"] = seeds

        # video input
        self.check_video_input(height, width, video_length)
        target_height = align_to(height, 16)
        target_width = align_to(width, 16)
        target_video_length = video_length
        out_dict["size"] = (target_height, target_width, target_video_length)
        filenames = self.process_savename(prompt_list, config.n_samples_prompt)

        samples = []
        gpu = []
        time = []
        for i, (prompt, i2v_image_path) in enumerate(zip(prompt_list, image_path_list)):
            result_with_metrics = self.single_inference(prompt, i2v_image_path, target_video_length, generator, config)
            sample = result_with_metrics['result']
            samples.append(sample)
            gpu.append(result_with_metrics.get('gpu', -1.0))
            time.append(result_with_metrics.get('time', -1.0))

            # Save samples
            if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
                save_videos_grid(sample, f"{config.savedir}/{filenames[i]}.mp4", fps=24)
        
        self.save_metrics(gpu=gpu, time=time, config=config, savedir=config.savedir)
        out_dict['samples'] = samples
        out_dict['prompts'] = prompt_list
        return out_dict

    def check_video_input(self, height, width, video_length):
        if width <= 0 or height <= 0 or video_length <= 0:
            raise ValueError(
                f"`height` and `width` and `video_length` must be positive integers, got height={height}, width={width}, video_length={video_length}"
            )
        if (video_length - 1) % 4 != 0:
            raise ValueError(
                f"`video_length-1` must be a multiple of 4, got {video_length}"
            )

        logger.info(
            f"Input (height, width, video_length) = ({height}, {width}, {video_length})"
        )

    def set_seed(self, seed, batch_size, num_videos_per_prompt):
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [
                random.randint(0, 1_000_000)
                for _ in range(batch_size * num_videos_per_prompt)
            ]
        elif isinstance(seed, int):
            seeds = [
                seed + i
                for _ in range(batch_size)
                for i in range(num_videos_per_prompt)
            ]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [
                    int(seed[i]) + j
                    for i in range(batch_size)
                    for j in range(num_videos_per_prompt)
                ]
            elif len(seed) == batch_size * num_videos_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt(batch_size) or "
                    f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got {seed}."
                )
        else:
            raise ValueError(
                f"Seed must be an integer, a list of integers, or None, got {seed}."
            )
            
        return seeds
    

    def enable_vram_management(self):
        pass
