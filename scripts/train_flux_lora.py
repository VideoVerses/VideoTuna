"""Train Flux LoRA adapters using the first-party Diffusers trainer."""

import argparse
import multiprocessing

from videotuna.settings import get_settings
from videotuna.training.flux_lora.train import run_training
from videotuna.utils.logging_config import bound_logger, configure_logging

logger = bound_logger(phase="t2i", flow="flux_lora")


def main(args: argparse.Namespace) -> None:
    configure_logging(level=get_settings().log_level)
    try:
        multiprocessing.set_start_method("fork")
    except Exception as exc:
        logger.warning("Could not set multiprocessing start method to 'fork': {}", exc)
    run_training(args.config_path, args.data_config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Flux LoRA (Diffusers + PEFT)"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Training config JSON"
    )
    parser.add_argument(
        "--data_config_path",
        type=str,
        required=True,
        help="Path to multidatabackend JSON",
    )
    main(parser.parse_args())
