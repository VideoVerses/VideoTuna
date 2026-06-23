import argparse
import os
import time
from enum import Enum
from pathlib import Path
from typing import Union

import torch
from colorama import Fore, Style
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from videotuna.training.wan_lora.config import (
    WanLoraTrainConfig,
    validated_config_to_dictconfig,
)
from videotuna.utils.config_mapping import apply_config_mappings
from videotuna.utils.lightning_utils import add_trainer_args_to_parser


class VideoMode(Enum):
    I2V = "i2v"
    T2V = "t2v"


MANDATORY_INFERENCE_ARGS = ["savedir"]


def prepare_train_args(parser: argparse.ArgumentParser):
    """
    Prepare the arguments by updating the config with the command line arguments.

    :param parser: The command line arguments.
    :param config: The config object.
    :return: The updated args, config object.
    """
    ## let parser recognize Trainer args
    parser = add_trainer_args_to_parser(Trainer, parser)

    ## let parser recognize and replace yaml configs such as flow.target or train.ckpt
    args, unknown = parser.parse_known_args()

    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    merged = OmegaConf.merge(*configs, cli)
    if not isinstance(merged, DictConfig):
        raise TypeError(f"Expected YAML mapping config, got {type(merged).__name__}")
    config = merged

    ## parser args replace train config
    train_config = config.get("train", OmegaConf.create())
    for k, v in vars(args).items():
        if k not in train_config.keys():
            train_config[k] = v
        else:
            if v is not None:
                train_config[k] = v

    apply_config_mappings(config, section="train")
    logger.info(f"All Config: {OmegaConf.to_yaml(config)}")

    def resolve_dtype(dtype_str):
        mapping = {
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
            "torch.bfloat16": torch.bfloat16,
        }
        return mapping.get(dtype_str)

    if not OmegaConf.has_resolver("dtype_resolver"):
        OmegaConf.register_new_resolver("dtype_resolver", resolve_dtype)

    ## extract trainer config
    trainer_config = config.train.lightning.trainer
    for k in get_nondefault_trainer_args(args):
        trainer_config[k] = getattr(args, k)

    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Training config must resolve to a mapping")
    validated = WanLoraTrainConfig.model_validate(resolved)
    return validated_config_to_dictconfig(validated)


def get_nondefault_trainer_args(args):
    parser = argparse.ArgumentParser()
    parser = add_trainer_args_to_parser(Trainer, parser)

    default_trainer_args = parser.parse_args([])
    return sorted(
        k
        for k in vars(default_trainer_args)
        if getattr(args, k) != getattr(default_trainer_args, k)
    )


def prepare_inference_args(args: argparse.Namespace, config: DictConfig) -> DictConfig:
    """
    Prepare the arguments by updating the config with the command line arguments.

    :param args: The command line arguments.
    :param config: The config object.
    :return: The updated config object.
    """
    from videotuna.utils.inference_cli import (
        prepare_cli_inference_args,
        validate_cpu_offload_flags,
    )
    from videotuna.utils.inference_profile import resolve_inference_profile

    prepare_cli_inference_args(args)

    # update the config with the command line arguments
    inference_config = config.pop("inference", OmegaConf.create())
    for k, v in vars(args).items():
        if k not in inference_config.keys():
            inference_config[k] = v
        else:
            if v is not None:
                inference_config[k] = v

    resolve_inference_profile(inference_config)
    validate_cpu_offload_flags(inference_config)

    check_args(inference_config)
    inference_config.savedir = process_savedir(inference_config.savedir)
    config.inference = inference_config
    print_inference_config(inference_config)

    apply_config_mappings(config, section="inference")

    logger.info(f"All Config: {OmegaConf.to_yaml(config)}")

    # resolve interpolation first
    def resolve_dtype(dtype_str):
        mapping = {
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
            "torch.bfloat16": torch.bfloat16,
        }
        return mapping.get(dtype_str)

    if not OmegaConf.has_resolver("dtype_resolver"):
        OmegaConf.register_new_resolver("dtype_resolver", resolve_dtype)
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Inference config must resolve to a mapping")
    config = OmegaConf.create(resolved, flags={"allow_objects": True})
    return config


def check_args(inference_config: DictConfig):
    """
    Check if all the mandatory arguments are provided.

    :param inference_config: The inferenceconfig object.
    """
    for k, v in inference_config.items():
        if k in MANDATORY_INFERENCE_ARGS and v is None:
            raise ValueError(f"The argument {k} is mandatory but not provided.")


def process_savedir(savedir: str):
    """
    Process the savedir.
    Add the current time to the savedir.
    Remove empty directories.

    :param savedir: The savedir config.
    :return: The processed savedir.
    """

    save_time = time.strftime("%Y%m%d_%H%M%S")
    savedir = os.path.join(savedir, save_time)

    # create the savedir
    Path(savedir).mkdir(parents=True, exist_ok=True)

    return savedir


def print_inference_config(inference_config: DictConfig):
    """
    Print the basic information of the inference config.
    Such as the mode, savedir, the seed, the height, width, frames, fps,
    n_samples_prompt, bs.

    :param inference_config: The inference config.
    """
    # Color definitions
    HEADER = Fore.CYAN
    KEY = Fore.GREEN
    VALUE = Fore.WHITE
    BORDER = Fore.BLUE
    RESET = Style.RESET_ALL

    # Header
    border = f"{BORDER}{'=' * 60}{RESET}"
    title = f"{HEADER}Inference Configuration{RESET}"

    print(border)
    print(f"{title:^60}")
    print(border)

    # Config items
    def print_item(key: str, value: Union[int, str, float, None]):
        if value is not None:
            print(f"{KEY}{key:<18}{RESET}: {VALUE}{value}{RESET}")

    config_items = {
        "Mode": inference_config.mode,
        "Save Directory": inference_config.savedir,
        "Height": inference_config.height,
        "Width": inference_config.width,
        "Frames": inference_config.frames,
        "FPS": inference_config.fps,
        "Seed": inference_config.seed,
        "Sample Batch Size": inference_config.bs,
        "Samples per Prompt": inference_config.n_samples_prompt,
    }

    for key, value in config_items.items():
        print_item(key, value)

    # Footer
    print(border)
