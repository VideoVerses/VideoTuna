import argparse
import datetime
import os
import sys

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.cli import LightningCLI
from transformers import logging as transf_logging

# sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(0, os.getcwd())
from videotuna.base.generation_base import GenerationBase
from videotuna.utils.args_utils import prepare_train_args
from videotuna.utils.common_utils import instantiate_from_config, get_dist_info
from videotuna.utils.lightning_utils import add_trainer_args_to_parser
from videotuna.utils.train_utils import (
    check_config_attribute,
    get_autoresume_path,
    get_empty_params_comparedwith_sd,
    get_trainer_callbacks,
    get_trainer_logger,
    get_trainer_strategy,
    init_workspace,
    load_checkpoints,
    set_logger,
)


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--seed", "-s", type=int, default=20230211, help="seed for seed_everything"
    )
    parser.add_argument(
        "--name", "-n", type=str, default="", help="experiment name, as saving folder"
    )

    parser.add_argument(
        "--base",
        "-b",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "--logdir",
        "-l",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="pretrained current model checkpoint"
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="resume from full-info checkpoint",
    )
    parser.add_argument(
        "--trained_ckpt", type=str, default=None, help="denoiser full checkpoint"
    )
    parser.add_argument(
        "--lorackpt", type=str, default=None, help="denoiser lora checkpoint"
    )
    return parser


def setup_logger(config: DictConfig):
    ## 1. dist info
    local_rank, global_rank, num_rank = get_dist_info()

    ## 2. config
    train_config : DictConfig = config.get("train", OmegaConf.create())
    lightning_config : DictConfig = train_config.get("lightning", OmegaConf.create())

    ## 3. init logger
    seed_everything(train_config.seed)
    transf_logging.set_verbosity_error()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    workdir, ckptdir, cfgdir, loginfo = init_workspace(
        train_config.name, train_config.logdir, config, lightning_config, global_rank
    )
    logger = set_logger(
        logfile=os.path.join(loginfo, "log_%d:%s.txt" % (global_rank, now))
    )
    train_config['workdir'] = workdir
    train_config['ckptdir'] = ckptdir
    return logger

if __name__ == "__main__":
    ## prepare args and logger
    local_rank, global_rank, num_rank = get_dist_info()
    parser = get_parser()
    config = prepare_train_args(parser)
    logger = setup_logger(config)   

    ## load flow
    logger.info("@lightning version: %s [>=2.0 required]" % pl.__version__)
    logger.info("***** Configuring Model *****")
    train_config: DictConfig = config['train']
    flow_config: DictConfig = config['flow']
    flow : GenerationBase = instantiate_from_config(flow_config, resolve=True)
    flow.from_pretrained(train_config['ckpt'], train_config['trained_ckpt'], train_config['lorackpt'])

    ## load trainer
    flow.init_trainer(train_config)
    trainer = flow.trainer
    data = flow.data

    ## train
    logger.info("***** Running the Loop *****")
    try:
        logger.info(f"<Training in {trainer.strategy.__class__.__name__} Mode>")
        if trainer.strategy.__class__.__name__  == "DeepSpeedStrategy":
            logger.info("deepspeed needs autocast")
            with torch.cuda.amp.autocast():
                trainer.fit(flow, data, ckpt_path=train_config.resume_ckpt)
        else:
            trainer.fit(flow, data, ckpt_path=train_config.resume_ckpt)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
