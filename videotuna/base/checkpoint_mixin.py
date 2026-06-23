import os
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from loguru import logger

from videotuna.utils.common_utils import print_green, print_yellow


class CheckpointMixin:
    def load_first_stage(
        self, ckpt_path: Union[str, Path], ignore_missing_ckpts: bool = False
    ) -> None:
        path = os.path.join(str(ckpt_path), self.first_stage_model_path)
        if os.path.exists(path):
            assert self.first_stage_model is not None
            self.first_stage_model = self.load_model(self.first_stage_model, path)
            print_green("Successfully loaded first_stage_model from checkpoint.")
        elif ignore_missing_ckpts:
            print_yellow("Checkpoint of first_stage_model file not found. Ignoring.")
        else:
            raise FileNotFoundError("Checkpoint of first_stage_model file not found.")

    def load_cond_stage(
        self, ckpt_path: Union[str, Path], ignore_missing_ckpts: bool = False
    ) -> None:
        path = os.path.join(str(ckpt_path), self.cond_stage_model_path)
        if os.path.exists(path):
            assert self.cond_stage_model is not None
            self.cond_stage_model = self.load_model(self.cond_stage_model, path)
            print_green("Successfully loaded cond_stage_model from checkpoint.")
        elif ignore_missing_ckpts:
            print_yellow("Checkpoint of cond_stage_model file not found. Ignoring.")
        else:
            raise FileNotFoundError("Checkpoint of cond_stage_model file not found.")

    def load_cond_stage_2(
        self, ckpt_path: Union[str, Path], ignore_missing_ckpts: bool = False
    ) -> None:
        if self.cond_stage_2_model is None:
            return

        path = os.path.join(str(ckpt_path), self.cond_stage_2_model_path)
        if os.path.exists(path):
            self.cond_stage_2_model = self.load_model(self.cond_stage_2_model, path)
            print_green("Successfully loaded cond_stage_2_model from checkpoint.")
        elif ignore_missing_ckpts:
            print_yellow("Checkpoint of cond_stage_2_model file not found. Ignoring.")
        else:
            raise FileNotFoundError("Checkpoint of cond_stage_2_model file not found.")

    def load_denoiser(
        self,
        ckpt_path: Optional[Union[str, Path]] = None,
        denoiser_ckpt_path: Optional[Union[str, Path]] = None,
        ignore_missing_ckpts: bool = False,
    ) -> None:
        if ckpt_path is None and denoiser_ckpt_path is None:
            return
        path = os.path.join(str(ckpt_path or ""), self.denoiser_path)
        if denoiser_ckpt_path is not None:
            path = str(denoiser_ckpt_path)

        if os.path.exists(path):
            assert self.denoiser is not None
            self.denoiser = self.load_model(self.denoiser, path)
            print_green("Successfully loaded denoiser from checkpoint.")
        elif ignore_missing_ckpts:
            print_yellow("Checkpoint of denoiser file not found. Ignoring.")
        else:
            raise FileNotFoundError("Checkpoint of denoiser file not found.")

    def load_lora(
        self,
        lora_ckpt_path: Optional[Union[str, Path]] = None,
        ignore_missing_ckpts: bool = False,
    ) -> None:
        if not self.use_lora:
            return

        lora_path = self.lora_path
        if lora_ckpt_path is not None:
            lora_path = str(lora_ckpt_path)

        if lora_path is not None and os.path.exists(lora_path):
            assert self.denoiser is not None
            self.load_model(self.denoiser, lora_path, strict=False)
            print_green("Successfully loaded denoiser from checkpoint.")
        elif ignore_missing_ckpts:
            print_yellow("Checkpoint of denoiser file not found. Ignoring.")
        else:
            raise FileNotFoundError("Checkpoint of denoiser file not found.")

    def from_pretrained(
        self,
        ckpt_path: Optional[Union[str, Path]] = None,
        denoiser_ckpt_path: Optional[Union[str, Path]] = None,
        lora_ckpt_path: Optional[Union[str, Path]] = None,
        ignore_missing_ckpts: bool = False,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        assert ckpt_path is not None, "Please provide a valid checkpoint path."
        ckpt_str = str(ckpt_path)
        denoiser_path = (
            str(denoiser_ckpt_path) if denoiser_ckpt_path is not None else None
        )
        lora_path = str(lora_ckpt_path) if lora_ckpt_path is not None else None

        # can ovrride following methods
        self.load_first_stage(ckpt_str, ignore_missing_ckpts)
        self.load_cond_stage(ckpt_str, ignore_missing_ckpts)
        self.load_cond_stage_2(ckpt_str, ignore_missing_ckpts)
        self.load_denoiser(ckpt_str, denoiser_path, ignore_missing_ckpts)
        self.load_lora(lora_path, ignore_missing_ckpts)

    @staticmethod
    def load_model(
        model: nn.Module, ckpt_path: Optional[Union[str, Path]] = None, strict=True
    ):
        """
        Loads the weights of the model from a checkpoint file.

        :param model: The model to be loaded.
        :param ckpt_path: Path to the checkpoint file.
        """
        assert ckpt_path is not None, "Please provide a valid checkpoint path."

        ckpt_path = Path(ckpt_path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=strict
            )
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            logger.info(
                f"{num_updated_keys} parameters are loaded from {ckpt_path}. "
                f"{num_unexpected_keys} parameters are unexpected."
            )
            return model
        else:
            raise FileNotFoundError("Checkpoint of model file not found.")
