import importlib
import json
import os
import time
from argparse import Namespace
from functools import wraps
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import psutil
import torch
import torch.distributed as dist
from colorama import Fore, Style
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from videotuna.settings import get_settings
from videotuna.utils.attention import (
    get_attn_backend_requested,
    get_resolved_attn_backend,
    get_torch_compile_mode,
)
from videotuna.utils.device_utils import (
    detect_compute_backend,
    gpu_is_available,
    synchronize_accelerator,
)
from videotuna.utils.inference_cli import resolve_offload_mode
from videotuna.utils.lora_utils import parameter_matches_lora_target

precision_to_dtype = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def get_resize_crop_region_for_grid(src, target):
    """
    Returns the centered crop region grid for a resized image to the target
    size while preserving aspect ratio.
    src: (h, w)
    target: (h, w)
    """

    h, w = src
    th, tw = target

    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def check_istarget(name, para_list):
    """
    name: full name of source para
    para_list: partial name of target para
    """
    return parameter_matches_lora_target(name, para_list)


def get_dtype_from_str(dtype_str):
    import torch

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)  # 默认返回float32


def get_params(config, resolve=True):
    params = config.get("params")
    if params is None:
        return dict()

    if resolve and isinstance(params, DictConfig):
        return OmegaConf.to_container(params, resolve=True)
    return params


# resolve will make params dict type rather than DictConfig type
def instantiate_from_config(config, resolve=False) -> Any:
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    target = config["target"]
    is_videotuna_diffusers_flow = target.endswith("DiffusersVideoFlow")
    if not is_videotuna_diffusers_flow and (
        "diffusers" in target
        or target.startswith("transformers")
        or config.get("use_from_pretrained", False)
    ):
        params = get_params(config, resolve)
        if isinstance(params.get("pretrained_model_name_or_path"), str):
            local_path = os.path.abspath(params["pretrained_model_name_or_path"])
            if os.path.isdir(local_path):
                params = dict(params)
                params["local_files_only"] = True
        return get_obj_from_str(config["target"]).from_pretrained(**params)
    return get_obj_from_str(config["target"])(**get_params(config, resolve))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_npz_from_dir(data_dir):
    data = [
        np.load(os.path.join(data_dir, data_name))["arr_0"]
        for data_name in os.listdir(data_dir)
    ]
    data = np.concatenate(data, axis=0)
    return data


def load_npz_from_paths(data_paths):
    data = [np.load(data_path)["arr_0"] for data_path in data_paths]
    data = np.concatenate(data, axis=0)
    return data


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def setup_dist(args):
    if dist.is_initialized():
        return
    if gpu_is_available():
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group("nccl", init_method="env://")
    else:
        torch.distributed.init_process_group("gloo", init_method="env://")


def print_green(text):
    print(Fore.GREEN + text + Style.RESET_ALL)


def print_red(text):
    print(Fore.RED + text + Style.RESET_ALL)


def print_yellow(text):
    print(Fore.YELLOW + text + Style.RESET_ALL)


def _build_sample_metrics(
    time_used: float,
    gpu_mem_used: Optional[float],
    frames: int,
) -> Dict[str, Any]:
    peak = round(gpu_mem_used, 2) if gpu_mem_used is not None else None
    wall = round(time_used, 2)
    spf = round(wall / frames, 4) if frames > 0 else None
    return {
        "time": wall,
        "wall_time_s": wall,
        "gpu": peak,
        "peak_vram_gb": peak,
        "seconds_per_frame": spf,
    }


def _current_cuda_device_index() -> int:
    if not gpu_is_available():
        return 0
    return torch.cuda.current_device()


def _peak_vram_stats(device_index: int) -> tuple[float | None, float | None]:
    if not gpu_is_available():
        return None, None
    allocated = torch.cuda.max_memory_allocated(device_index) / (1024**3)
    reserved = torch.cuda.max_memory_reserved(device_index) / (1024**3)
    return round(allocated, 2), round(reserved, 2)


def _strip_non_serializable_metrics(sample: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(sample)
    result = cleaned.pop("result", None)
    if result is not None and not isinstance(
        result, (str, int, float, bool, list, dict, type(None))
    ):
        cleaned["result_type"] = type(result).__name__
    return cleaned


def monitor_resources(
    return_metrics: bool = True,
    frames: int = 1,
    inference_config: Optional[Any] = None,
    device_index: Optional[int] = None,
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            start_time = time.time()
            start_cpu_mem = process.memory_info().rss / 1024 / 1024 / 1024  # GB

            dev_idx = device_index
            if dev_idx is None and gpu_is_available():
                dev_idx = _current_cuda_device_index()

            if gpu_is_available():
                torch.cuda.reset_peak_memory_stats(dev_idx)
                synchronize_accelerator()

            result = func(*args, **kwargs)

            end_time = time.time()
            end_cpu_mem = process.memory_info().rss / 1024 / 1024 / 1024  # GB

            time_used = end_time - start_time
            cpu_mem_used = end_cpu_mem - start_cpu_mem

            logger.info(f"Time used: {time_used:.2f} seconds")
            logger.info(f"CPU memory change: {cpu_mem_used:.2f} GB")
            peak_alloc, peak_reserved = _peak_vram_stats(dev_idx or 0)
            if peak_alloc is not None:
                logger.info(f"Peak GPU memory allocated: {peak_alloc:.2f} GB")
            if peak_reserved is not None:
                logger.info(f"Peak GPU memory reserved: {peak_reserved:.2f} GB")

            if return_metrics:
                sample = _build_sample_metrics(time_used, peak_alloc, frames)
                sample["cpu"] = round(cpu_mem_used, 2)
                sample["peak_vram_reserved_gb"] = peak_reserved
                sample["attention_backend"] = get_resolved_attn_backend()
                sample["attention_backend_requested"] = get_attn_backend_requested()
                sample["attention_backend_resolved"] = get_resolved_attn_backend()
                sample["compute_backend"] = detect_compute_backend()
                compile_on = get_settings().torch_compile
                sample["torch_compile"] = compile_on
                sample["compile_mode"] = (
                    get_torch_compile_mode() if compile_on else None
                )
                sample["result"] = result
                if dev_idx is not None and gpu_is_available():
                    sample["gpu_index"] = dev_idx
                    sample["gpu_name"] = torch.cuda.get_device_name(dev_idx)
                if inference_config is not None:
                    sample["offload_mode"] = resolve_offload_mode(inference_config)
                    sample["dtype"] = getattr(inference_config, "dtype", None)
                    sample["memory_preset"] = getattr(
                        inference_config, "memory_preset", None
                    )
                    sample["requested_device"] = getattr(
                        inference_config, "device", None
                    )
                return sample
            return result

        return wrapper

    return decorator


def save_metrics(
    savedir: str,
    config: Optional[Union[DictConfig, Namespace, Any]] = None,
    *,
    metrics: Optional[Dict[str, Any]] = None,
    gpu: Optional[List[float]] = None,
    time: Optional[List[float]] = None,
    frames: int = 1,
):
    """Write metrics.json (and legacy metric.json) beside inference outputs."""
    config_dict = None
    if config is not None:
        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        elif isinstance(config, Namespace):
            config_dict = vars(config)
        elif hasattr(config, "items"):
            config_dict = dict(config)

    if metrics is None:
        per_sample = []
        gpu_list = gpu or []
        time_list = time or []
        for g, t in zip(gpu_list, time_list):
            per_sample.append(
                {
                    "peak_vram_gb": g,
                    "wall_time_s": t,
                    "seconds_per_frame": (
                        round(t / frames, 4) if frames > 0 and t else None
                    ),
                }
            )
        metrics = {
            "per_sample": per_sample,
            "gpu": gpu_list,
            "time": time_list,
            "attention_backend": get_resolved_attn_backend(),
            "attention_backend_requested": get_attn_backend_requested(),
            "attention_backend_resolved": get_resolved_attn_backend(),
            "torch_compile": get_settings().torch_compile,
        }
        if config is not None:
            metrics["offload_mode"] = resolve_offload_mode(config)
            metrics["dtype"] = getattr(config, "dtype", None)
            metrics["memory_preset"] = getattr(config, "memory_preset", None)
            compile_on = metrics["torch_compile"]
            metrics["compile_mode"] = get_torch_compile_mode() if compile_on else None

    if metrics.get("per_sample"):
        metrics["per_sample"] = [
            _strip_non_serializable_metrics(s) if isinstance(s, dict) else s
            for s in metrics["per_sample"]
        ]
    metrics = _strip_non_serializable_metrics(metrics)

    if config_dict is not None:
        metrics["config"] = config_dict

    if metrics.get("per_sample"):
        peaks = [
            s.get("peak_vram_gb")
            for s in metrics["per_sample"]
            if s.get("peak_vram_gb") is not None
        ]
        times = [
            s.get("wall_time_s")
            for s in metrics["per_sample"]
            if s.get("wall_time_s") is not None
        ]
        if peaks:
            metrics["peak_vram_gb"] = max(peaks)
        if times:
            metrics["wall_time_s"] = sum(times)
            metrics["seconds_per_frame"] = (
                round(metrics["wall_time_s"] / frames, 4) if frames > 0 else None
            )

    os.makedirs(savedir, exist_ok=True)
    metrics_path = os.path.join(savedir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    legacy_path = os.path.join(savedir, "metric.json")
    with open(legacy_path, "w") as f:
        json.dump(metrics, f, indent=4)


def get_dist_info():
    try:
        local_rank = int(os.environ.get("LOCAL_RANK") or 0)
        global_rank = int(os.environ.get("RANK") or 0)
        num_rank = int(os.environ.get("WORLD_SIZE") or 1)
    except (TypeError, ValueError):
        local_rank, global_rank, num_rank = 0, 0, 1
    return local_rank, global_rank, num_rank
