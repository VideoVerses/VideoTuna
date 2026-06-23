from typing import Any, Dict, List, Optional, Union, cast

import torch.nn as nn
from loguru import logger
from peft import get_peft_model

from videotuna.base.component_loader import Component
from videotuna.utils.common_utils import instantiate_from_config, print_green
from videotuna.utils.lora_utils import (
    collect_lora_parameter_names,
    resolve_lora_target_modules,
)


class LoraTrainingMixin:
    def instantiate_lora(self, config: Optional[Dict[str, Any]]) -> None:
        self.use_lora = False
        if config is not None:
            logger.info("creating lora")
            transformer_adapter_config = instantiate_from_config(config)
            assert self.denoiser is not None
            if transformer_adapter_config is not None and hasattr(
                transformer_adapter_config, "target_modules"
            ):
                transformer_adapter_config.target_modules = resolve_lora_target_modules(
                    self.denoiser, transformer_adapter_config.target_modules
                )
            self.denoiser = get_peft_model(
                cast(Any, self.denoiser),
                cast(Any, transformer_adapter_config),
                autocast_adapter_dtype=False,
            )
            self.lora_params = collect_lora_parameter_names(self.denoiser)
            self.denoiser.requires_grad_(False)
            for name, param in self.denoiser.named_parameters():
                if name in self.lora_params:
                    param.requires_grad_(True)
            self.use_lora = True
            self.lora_path = config.get("ckpt_path")
            logger.info(
                f"self.use_lora: {self.use_lora} self.lora_path: {self.lora_path} "
                f"self.lora_params: {self.lora_params}"
            )

    def set_trainable_components(
        self,
        components: Union[str, List[str]] = [],
    ):
        """
        Sets the components of the generative model that should be trainable.

        :param components: The components to be set as trainable.
        """
        if isinstance(components, str):
            components = [components]

        # eval all components
        for component in self.components:
            model = getattr(self, component)
            if model is None or not isinstance(model, nn.Module):
                logger.info(
                    f"Skipping eval component {component} since it is not set or "
                    "not module"
                )
                continue

            model.eval()
            model.requires_grad_(False)

        # train selected components
        for component in components:
            model = getattr(self, component)
            if model is None:
                raise ValueError(f"Invalid component name: {component}")

            if not isinstance(model, nn.Module):
                logger.info(
                    f"Skipping train component {component} since it is not module"
                )
                continue

            # if denoiser lora, make sure only lora params require grad
            if component == Component.DENOISER.value and self.use_lora:
                ## TODO how to define lora module
                model.train()
                for name, param in model.named_parameters():
                    if name in self.lora_params:
                        param.requires_grad_(True)
            else:
                model.train()
                model.requires_grad_(True)

        print_green(f"Set the following components as trainable: {components}")
