import argparse
import json
import time
from colorama import Fore, Style
from omegaconf import OmegaConf, MissingMandatoryValue
from pathlib import Path
from typing import Union
import torch
from enum import Enum
from pytorch_lightning import Trainer
from videotuna.utils.lightning_utils import add_trainer_args_to_parser
from loguru import logger
import os

class VideoMode(Enum):
    I2V = "i2v"
    T2V = "t2v"


MANDATORY_INFERENCE_ARGS = ["savedir"]


def prepare_train_args(parser: argparse.Namespace):
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
    config = OmegaConf.merge(*configs, cli)

    ## parser args replace train config 
    train_config = config.get("train", OmegaConf.create())
    for k, v in vars(args).items():
        if not k in train_config.keys():
            train_config[k] = v
        else:
            if v is not None:
                train_config[k] = v

    if OmegaConf.select(config, 'train.mapping') is not None:
        for source_path, target_path in config.train.mapping.items():
            if not path_exists(config, source_path):
                raise ValueError(f"Error: invalid mapping {source_path} not exists")
            if not path_exists(config, target_path):
                raise ValueError(f"Error: invalid mapping {target_path} not exists")
            
            value = OmegaConf.select(config, source_path)
            if value is not None:
                OmegaConf.update(config, target_path, value)
                logger.info(f"update {target_path} by {source_path} value: {value}")
    logger.info(f"All Config: {OmegaConf.to_yaml(config)}")
    def resolve_dtype(dtype_str):
        mapping = {
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
            "torch.bfloat16": torch.bfloat16,
        }
        return mapping.get(dtype_str)
    OmegaConf.register_new_resolver("dtype_resolver", resolve_dtype)

    ## extract trainer config
    trainer_config = config.train.lightning.trainer 
    for k in get_nondefault_trainer_args(args):
        trainer_config[k] = getattr(args, k)
    return config

def get_nondefault_trainer_args(args):
    parser = argparse.ArgumentParser()
    parser = add_trainer_args_to_parser(Trainer, parser)

    default_trainer_args = parser.parse_args([])
    return sorted(
        k
        for k in vars(default_trainer_args)
        if getattr(args, k) != getattr(default_trainer_args, k)
    )

# omegaconf has bug, does not work as expected
def path_exists(cfg, path):
    try:
        OmegaConf.select(cfg, path, throw_on_missing=True)
        return True
    except MissingMandatoryValue:
        return False

def prepare_inference_args(args: argparse.Namespace, config: OmegaConf):
    """
    Prepare the arguments by updating the config with the command line arguments.

    :param args: The command line arguments.
    :param config: The config object.
    :return: The updated config object.
    """

    # update the config with the command line arguments
    inference_config = config.pop("inference", OmegaConf.create())
    for k, v in vars(args).items():
        if not k in inference_config.keys():
            inference_config[k] = v
        else:
            if v is not None:
                inference_config[k] = v
                
    check_args(inference_config)
    inference_config.savedir = process_savedir(inference_config.savedir)    
    config.inference = inference_config
    print_inference_config(inference_config)


    #update flow config with inference mapping config
    if OmegaConf.select(config, 'inference.mapping') is not None:
        for source_path, target_path in config.inference.mapping.items():
            if not path_exists(config, source_path):
                raise ValueError(f"Error: invalid mapping {source_path} not exists")
            if not path_exists(config, target_path):
                raise ValueError(f"Error: invalid mapping {target_path} not exists")
            
            value = OmegaConf.select(config, source_path)
            if value is not None:
                OmegaConf.update(config, target_path, value)
                logger.info(f"update {target_path} by {source_path} value: {value}")

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
    OmegaConf.register_new_resolver("dtype_resolver", resolve_dtype)
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config, flags={"allow_objects": True})
    return config

def check_args(inference_config: OmegaConf):
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


def print_inference_config(inference_config: OmegaConf):
    """
    Print the basic information of the inference config.
    Such as the mode, savedir, the seed, the height, width, frames, fps, n_samples_prompt, bs.

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
    
