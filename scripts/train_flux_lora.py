import os
import sys

sys.path.insert(0, os.getcwd())

import argparse
import json
import logging
import time
from os import environ
from pathlib import Path

import torch.distributed as dist
from pytorch_lightning import Trainer

from videotuna.third_party.flux import log_format
from videotuna.third_party.flux.training.model import Model
from videotuna.third_party.flux.training.model_data import ModelData
from videotuna.third_party.flux.training.state_tracker import StateTracker

logger = logging.getLogger("SimpleTuner")
logger.setLevel(environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


def config_process(config):
    # add timestamp to the output_dir
    output_dir = Path(config["--output_dir"])
    time_str = time.strftime("%Y%m%d%H%M%S")

    folder_name = output_dir.stem
    name_list = folder_name.split("_")
    if len(name_list[-1]) == 14:
        folder_name = "_".join(name_list[:-1])
    folder_name = f"{folder_name}_{time_str}"
    output_dir = output_dir.parent / folder_name
    config["--output_dir"] = str(output_dir)

    with open(args.config_path, "w") as f:
        json.dump(config, f, indent=4)
    return config


def main(args):
    try:
        import multiprocessing

        multiprocessing.set_start_method("fork")
    except Exception as e:
        logger.error(
            "Failed to set the multiprocessing start method to 'fork'. Unexpected behaviour such as high memory overhead or poor performance may result."
            f"\nError: {e}"
        )
    try:
        # load config files
        with open(args.config_path) as f:
            config = json.load(f)
        with open(args.data_config_path) as f:
            data_config = json.load(f)
        # process config
        config = config_process(config)

        data_dir = data_config[0]["instance_data_dir"]
        dm = ModelData(data_dir)
        dm.create_dataset()
        dm.setup()
        print("dataset setup done!")
        train_dataloader = dm.train_dataloader()
        test_dataloader = dm.test_dataloader()
        print("loaded dataloaders")
        model = Model()
        model.run()
        print("loaded model")
        trainer = Trainer(
            accelerator="gpu",
            max_epochs=config["--num_train_epochs"],
            strategy="ddp",
            limit_train_batches=1490,
            logger=False,
        )
        print("loaded Trainer, training...")
        # print("model params:", list(model.parameters()))
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        trainer.fit(model, datamodule=dm)

        print("train finished")

    except Exception as e:
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument(
        "--data_config_path", type=str, help="Path to the config of data file"
    )
    args = parser.parse_args()

    main(args)
