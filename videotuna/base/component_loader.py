import enum
from typing import Any, Dict, Optional

from loguru import logger

from videotuna.utils.common_utils import instantiate_from_config


class Component(str, enum.Enum):
    DENOISER = "denoiser"
    FIRST_STAGE_MODEL = "first_stage_model"
    COND_STAGE_MODEL = "cond_stage_model"
    COND_STAGE_2_MODEL = "cond_stage_2_model"
    SCHEDULER = "scheduler"

    def get_component_path(self) -> str:
        return f"{self.value}.ckpt"


class LoadingMethod(str, enum.Enum):
    FIXED = "fixed"
    CONFIG = "config"


class ComponentLoader:
    def instantiate_scheduler(self, config: Optional[Dict[str, Any]]) -> None:
        if config is not None:
            logger.info("creating scheduler")
            self.diffusion_scheduler = self.scheduler = instantiate_from_config(config)
            self.components.append(Component.SCHEDULER.value)

    def instantiate_first_stage(self, config: Optional[Dict[str, Any]]) -> None:
        """
        Instantiates the first stage model of the generative process.

        :param config: Dictionary containing configuration for the first stage model.
        """
        if config is None:
            return
        logger.info("creating first stage")
        model = instantiate_from_config(config)
        assert model is not None
        self.first_stage_model = model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
        self.components.append(Component.FIRST_STAGE_MODEL.value)
        self.first_stage_model_path = config.get(
            "ckpt_path", f"{Component.FIRST_STAGE_MODEL.value}.ckpt"
        )
        logger.info(f"self.first_stage_model_path: {self.first_stage_model_path}")

    def instantiate_cond_stage(self, config: Optional[Dict[str, Any]]) -> None:
        """
        Instantiates the conditional stage model of the generative process.

        :param config: Dictionary containing configuration for the conditional
            stage model.
        """
        if config is None:
            return
        from videotuna.utils.quantization import apply_quantization_to_config_params

        logger.info("creating cond stage")
        cfg = config
        if cfg is not None and isinstance(cfg, dict) and cfg.get("params"):
            cfg = dict(cfg)
            cfg["params"] = apply_quantization_to_config_params(dict(cfg["params"]))
        model = instantiate_from_config(cfg)
        assert model is not None
        self.cond_stage_model = model.eval()
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False
        self.components.append(Component.COND_STAGE_MODEL.value)
        self.cond_stage_model_path = config.get(
            "ckpt_path", f"{Component.COND_STAGE_MODEL.value}.ckpt"
        )
        logger.info(f"self.cond_stage_model_path: {self.cond_stage_model_path}")

    def instantiate_cond_stage_2(self, config: Optional[Dict[str, Any]]) -> None:
        """
        Instantiates the conditional stage model of the generative process.

        :param config: Dictionary containing configuration for the conditional
            stage model.
        """
        self.cond_stage_2_model = None
        if config is not None:
            logger.info("creating cond stage 2")
            model = instantiate_from_config(config)
            assert model is not None
            self.cond_stage_2_model = model.eval()
            for param in self.cond_stage_2_model.parameters():
                param.requires_grad = False
            self.components.append(Component.COND_STAGE_2_MODEL.value)
            self.cond_stage_2_model_path = config.get(
                "ckpt_path", f"{Component.COND_STAGE_2_MODEL.value}.ckpt"
            )
            logger.info(f"self.cond_stage_2_model_path: {self.cond_stage_2_model_path}")

    def instantiate_denoiser(self, config: Optional[Dict[str, Any]]) -> None:
        """
        Instantiates the denoiser model of the generative process.

        :param config: Dictionary containing configuration for the denoiser model.
        """
        if config is None:
            return
        logger.info("creating denoiser")
        model = instantiate_from_config(config)
        assert model is not None
        self.denoiser = model.eval()
        for param in self.denoiser.parameters():
            param.requires_grad = False
        self.components.append(Component.DENOISER.value)
        self.denoiser_path = config.get("ckpt_path", f"{Component.DENOISER.value}.ckpt")
        logger.info(f"self.denoiser_path: {self.denoiser_path}")

    def apply_denoiser_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Enable gradient checkpointing on the denoiser only."""
        denoiser = getattr(self, "denoiser", None)
        if denoiser is None:
            return
        if hasattr(denoiser, "activation_checkpointing"):
            denoiser.activation_checkpointing = enabled
            logger.info(f"Wan denoiser activation_checkpointing={enabled}")
            return
        base_model = getattr(denoiser, "base_model", denoiser)
        model = getattr(base_model, "model", base_model)
        if enabled and hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
            logger.info("Enabled diffusers gradient checkpointing on denoiser")
        elif not enabled and hasattr(model, "disable_gradient_checkpointing"):
            model.disable_gradient_checkpointing()
            logger.info("Disabled diffusers gradient checkpointing on denoiser")
