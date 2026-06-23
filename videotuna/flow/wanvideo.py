import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from PIL import Image

import videotuna.models.wan.wan as wan
from videotuna.base.generation_base import GenerationBase
from videotuna.models.wan.wan.configs import (
    MAX_AREA_CONFIGS,
    SIZE_CONFIGS,
    SUPPORTED_SIZES,
    WAN_CONFIGS,
)
from videotuna.models.wan.wan.utils.prompt_extend import (
    DashScopePromptExpander,
    QwenPromptExpander,
)
from videotuna.utils.args_utils import VideoMode
from videotuna.utils.attention import maybe_compile_denoiser
from videotuna.utils.common_utils import monitor_resources
from videotuna.utils.device_utils import (
    gpu_is_available,
    require_xfuser_sequence_parallel,
)
from videotuna.utils.logging_config import (
    bound_logger,
    phase_from_wan_task,
    resolve_device_label,
)
from videotuna.utils.wan_training import (
    compute_wan_flow_matching_loss,
    init_wan_training_denoisers,
)

_BOXING_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves "
    "fight intensely on a spotlighted stage."
)
_BEACH_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a "
    "surfboard. The fluffy-furred feline gazes directly at the camera with a "
    "relaxed expression. Blurred beach scenery forms the background featuring "
    "crystal-clear waters, distant green hills, and a blue sky dotted with "
    "white clouds. The cat assumes a naturally relaxed posture, as if "
    "savoring the sea breeze and warm sunlight. A close-up shot highlights "
    "the feline's intricate details and the refreshing atmosphere of the "
    "seaside."
)

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": _BOXING_PROMPT,
    },
    "t2v-14B": {
        "prompt": _BOXING_PROMPT,
    },
    "t2v-A14B": {
        "prompt": _BOXING_PROMPT,
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt": _BEACH_PROMPT,
        "image": "inputs/i2v/576x1024/i2v_input.JPG",
    },
    "i2v-A14B": {
        "prompt": _BEACH_PROMPT,
        "image": "inputs/i2v/576x1024/i2v_input.JPG",
    },
}


class WanVideoModelFlow(GenerationBase):
    prompt_expander: DashScopePromptExpander | QwenPromptExpander | None = None

    """
    Training and inference flow for YourModel.

    This model inherits from GenerationFlow, which is a base class for all
    generative models.
    """

    def __init__(
        self,
        first_stage_config: Dict[str, Any],
        cond_stage_config: Dict[str, Any],
        denoiser_config: Dict[str, Any],
        scheduler_config: Optional[Dict[str, Any]] = None,
        cond_stage_2_config: Optional[Dict[str, Any]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        gradient_checkpointing: bool = True,
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
        *args,
        **kwargs,
    ):
        phase = phase_from_wan_task(task)
        self._log = bound_logger(phase=phase, flow="wanvideo")
        self._log.info("WanVideo flow: starting init")
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
            trainable_components=[],
        )
        self.apply_denoiser_gradient_checkpointing(gradient_checkpointing)
        self._log.info("WanVideo flow: class init finished")
        self.task = task
        self.ckpt_path = ckpt_path
        self.use_prompt_extend = use_prompt_extend
        self.prompt_extend_model = prompt_extend_model
        self.prompt_extend_target_lang = prompt_extend_target_lang
        self.seed = seed
        self.offload_model = offload_model
        self.ulysses_size = ulysses_size
        self.ring_size = ring_size
        self.use_sp = ulysses_size > 1 or ring_size > 1

        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank
        if gpu_is_available():
            device_label = resolve_device_label(torch.device(f"cuda:{local_rank}"))
        else:
            device_label = "cpu"
        self._log = self._log.bind(device=device_label)

        if offload_model is None:
            offload_model = False if world_size > 1 else True
            self._log.info("offload_model is not specified, set to {}.", offload_model)
        if world_size > 1:
            if gpu_is_available():
                torch.cuda.set_device(local_rank)
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    rank=rank,
                    world_size=world_size,
                )
            self._log.info("WanVideo flow: Init Process Group")
        else:
            assert not (
                t5_fsdp or dit_fsdp
            ), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
            assert not (
                ulysses_size > 1 or ring_size > 1
            ), "context parallel are not supported in non-distributed environments."

        if ulysses_size > 1 or ring_size > 1:
            require_xfuser_sequence_parallel("WanVideoModelFlow")
            assert ulysses_size * ring_size == world_size, (
                "The number of ulysses_size and ring_size should be equal to "
                "the world size."
            )
            from xfuser.core.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )

            init_distributed_environment(
                rank=dist.get_rank(), world_size=dist.get_world_size()
            )

            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=ring_size,
                ulysses_degree=ulysses_size,
            )
            self._log.info(
                "WanVideo flow: Init Ring/Ulysses Seqeunce Parallel Process Group"
            )

        if use_prompt_extend:
            if prompt_extend_method == "dashscope":
                self.prompt_expander = DashScopePromptExpander(
                    model_name=prompt_extend_model, is_vl="i2v" in task
                )
            elif prompt_extend_method == "local_qwen":
                self.prompt_expander = QwenPromptExpander(
                    model_name=prompt_extend_model, is_vl="i2v" in task, device=rank
                )
            else:
                raise NotImplementedError(
                    f"Unsupport prompt_extend_method: {prompt_extend_method}"
                )
            self._log.info("WanVideo flow: Set Prompt Extention")

        cfg = WAN_CONFIGS[task]
        self.cfg = cfg
        if ulysses_size > 1:
            num_heads = getattr(cfg, "num_heads", None)
            assert num_heads is not None, "Wan config missing num_heads"
            assert num_heads % ulysses_size == 0, (
                f"`num_heads={num_heads}` cannot be divided evenly by "
                f"`ulysses_size={ulysses_size}`."
            )

        self._log.info(f"WanVideo flow: model config: {cfg}")

        if dist.is_initialized():
            seed_list: list[int | None] = [seed] if rank == 0 else [None]
            dist.broadcast_object_list(seed_list, src=0)
            broadcast_seed = seed_list[0]
            assert broadcast_seed is not None
            seed = broadcast_seed
            self.seed = seed
            self._log.info("WanVideo flow: broadcast seed")

        use_sp = self.use_sp
        if "t2v" in task or "t2i" in task:
            self._log.info("Creating WanT2V pipeline.")
            self.wan_t2v = wan.WanT2V(
                config=cfg,
                checkpoint_dir=ckpt_path,
                device_id=device,
                rank=rank,
                t5_fsdp=t5_fsdp,
                dit_fsdp=dit_fsdp,
                use_sp=use_sp,
                t5_cpu=t5_cpu,
            )
        else:
            self._log.info("Creating WanI2V pipeline.")
            self.wan_i2v = wan.WanI2V(
                config=cfg,
                checkpoint_dir=ckpt_path,
                device_id=device,
                rank=rank,
                t5_fsdp=t5_fsdp,
                dit_fsdp=dit_fsdp,
                use_sp=use_sp,
                t5_cpu=t5_cpu,
            )

        init_wan_training_denoisers(self)

    def _validate_args(self, args):
        # Size reassign and check
        args.size = f"{args.width}*{args.height}"
        self._log.info(f"setting size = width*height == {args.size}")
        supported = ", ".join(SUPPORTED_SIZES[self.task])
        assert args.size in SUPPORTED_SIZES[self.task], (
            f"Unsupport size {args.size} for task {self.task}, "
            f"supported sizes are: {supported}"
        )

    def inference_t2v(self, args: DictConfig):
        # init vars
        rank = int(os.getenv("RANK", 0))
        int(os.getenv("WORLD_SIZE", 1))
        int(os.getenv("LOCAL_RANK", 0))

        frames = args.frames
        size = args.size
        sample_shift = args.time_shift
        sample_solver = args.solver
        sampling_steps = args.num_inference_steps
        guide_scale = args.unconditional_guidance_scale

        # load input
        prompt_list = self.load_inference_inputs(args.prompt_file, args.mode)
        if len(prompt_list) > 1:
            self._log.info("Processing prompts sequentially (batch size 1 per prompt).")

        videos = []
        gpu = []
        time = []
        for prompt in prompt_list:
            self._log.info(f"Input prompt: {prompt}")
            if self.use_prompt_extend:
                assert self.prompt_expander is not None
                self._log.info("Extending prompt ...")
                if rank == 0:
                    prompt_output = self.prompt_expander(
                        prompt, tar_lang=self.prompt_extend_target_lang, seed=self.seed
                    )
                    assert prompt_output is not None
                    if prompt_output.status is False:
                        self._log.info(
                            f"Extending prompt failed: {prompt_output.message}"
                        )
                        self._log.info("Falling back to original prompt.")
                        input_prompt = prompt
                    else:
                        input_prompt = prompt_output.prompt
                    input_prompt = [input_prompt]
                else:
                    input_prompt = [None]
                if dist.is_initialized():
                    dist.broadcast_object_list(input_prompt, src=0)
                prompt = input_prompt[0]
                self._log.info(f"Extended prompt: {prompt}")

            self._log.info(
                f"Generating {'image' if 't2i' in self.task else 'video'} ..."
            )

            @monitor_resources(return_metrics=True, frames=frames)
            def _run_generate():
                return self.wan_t2v.generate(
                    prompt,
                    size=SIZE_CONFIGS[size],
                    frame_num=frames,
                    shift=sample_shift,
                    sample_solver=sample_solver,
                    sampling_steps=sampling_steps,
                    guide_scale=guide_scale,
                    seed=self.seed,
                    offload_model=self.offload_model,
                )

            result_with_metrics = _run_generate()
            video = result_with_metrics["result"]
            videos.append(video)

            gpu.append(
                result_with_metrics.get("peak_vram_gb")
                or result_with_metrics.get("gpu", -1.0)
            )
            time.append(
                result_with_metrics.get("wall_time_s")
                or result_with_metrics.get("time", -1.0)
            )

        if rank == 0:
            self._log.info("Saving videos")
            filenames = self.process_savename(prompt_list, args.n_samples_prompt)
            self.save_videos(
                torch.stack(videos).unsqueeze(dim=1),
                args.savedir,
                filenames,
                fps=args.savefps,
            )
            self.save_metrics(
                gpu=gpu, time=time, config=args, savedir=args.savedir, frames=frames
            )

    def inference_i2v(self, args: DictConfig):
        # init vars
        rank = int(os.getenv("RANK", 0))
        int(os.getenv("WORLD_SIZE", 1))
        int(os.getenv("LOCAL_RANK", 0))

        frames = args.frames
        size = args.size
        sample_shift = args.time_shift
        sample_solver = args.solver
        sampling_steps = args.num_inference_steps
        guide_scale = args.unconditional_guidance_scale

        prompt_list, image_list = self.load_inference_inputs(args.prompt_dir, args.mode)
        assert len(prompt_list) == len(
            image_list
        ), "prompt and image number should match"

        if len(prompt_list) > 1:
            self._log.info("Processing prompts sequentially (batch size 1 per prompt).")

        videos = []
        gpu = []
        time = []
        for prompt, image_path in zip(prompt_list, image_list):
            self._log.info(f"Input prompt: {prompt}")
            self._log.info(f"Input image: {image_path}")

            img = Image.open(image_path).convert("RGB")
            if self.use_prompt_extend:
                assert self.prompt_expander is not None
                self._log.info("Extending prompt ...")
                if rank == 0:
                    prompt_output = self.prompt_expander(
                        prompt,
                        tar_lang=self.prompt_extend_target_lang,
                        image=img,
                        seed=self.seed,
                    )
                    assert prompt_output is not None
                    if prompt_output.status is False:
                        self._log.info(
                            f"Extending prompt failed: {prompt_output.message}"
                        )
                        self._log.info("Falling back to original prompt.")
                        input_prompt = prompt
                    else:
                        input_prompt = prompt_output.prompt
                    input_prompt = [input_prompt]
                else:
                    input_prompt = [None]
                if dist.is_initialized():
                    dist.broadcast_object_list(input_prompt, src=0)
                prompt = input_prompt[0]
                self._log.info(f"Extended prompt: {prompt}")

            self._log.info("Generating video ...")

            @monitor_resources(return_metrics=True, frames=frames)
            def _run_generate():
                return self.wan_i2v.generate(
                    prompt,
                    img,
                    max_area=MAX_AREA_CONFIGS[size],
                    frame_num=frames,
                    shift=sample_shift,
                    sample_solver=sample_solver,
                    sampling_steps=sampling_steps,
                    guide_scale=guide_scale,
                    seed=self.seed,
                    offload_model=self.offload_model,
                )

            result_with_metrics = _run_generate()
            video = result_with_metrics["result"]
            video = video.cpu()
            videos.append(video)
            gpu.append(
                result_with_metrics.get("peak_vram_gb")
                or result_with_metrics.get("gpu", -1.0)
            )
            time.append(
                result_with_metrics.get("wall_time_s")
                or result_with_metrics.get("time", -1.0)
            )
            del result_with_metrics

        if rank == 0:
            self._log.info("Saving videos")
            filenames = self.process_savename(prompt_list, args.n_samples_prompt)
            self.save_videos(
                torch.stack(videos).unsqueeze(dim=1),
                args.savedir,
                filenames,
                fps=args.savefps,
            )
            self.save_metrics(
                gpu=gpu, time=time, config=args, savedir=args.savedir, frames=frames
            )

    @torch.inference_mode()
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
            raise ValueError(
                "Error: invalid mode, we currently only support t2v and i2v "
                "for wanvideo"
            )

    def from_pretrained(
        self,
        ckpt_path: Optional[Union[str, Path]] = None,
        denoiser_ckpt_path: Optional[Union[str, Path]] = None,
        lora_ckpt_path: Optional[Union[str, Path]] = None,
        ignore_missing_ckpts: bool = False,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        if denoiser_ckpt_path is not None or ckpt_path is not None:
            self.load_denoiser(ckpt_path, denoiser_ckpt_path, True)
        if not self.use_sp:
            if "t2v" in self.task or "t2i" in self.task:
                self.wan_t2v.low_noise_model = cast(
                    Any, maybe_compile_denoiser(self.wan_t2v.low_noise_model)
                )
                self.wan_t2v.high_noise_model = cast(
                    Any, maybe_compile_denoiser(self.wan_t2v.high_noise_model)
                )
            else:
                self.wan_i2v.low_noise_model = cast(
                    Any, maybe_compile_denoiser(self.wan_i2v.low_noise_model)
                )
                self.wan_i2v.high_noise_model = cast(
                    Any, maybe_compile_denoiser(self.wan_i2v.high_noise_model)
                )

    def enable_vram_management(self) -> None:
        self._log.info(
            "WanVideoModelFlow: VRAM handled via offload_model in generate(); no-op"
        )

    def training_step(self, batch, batch_idx):
        loss = compute_wan_flow_matching_loss(self, batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        pass
