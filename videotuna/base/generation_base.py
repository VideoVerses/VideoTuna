from loguru import logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from colorama import Fore, Style

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from peft import get_peft_model
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer
import enum

from videotuna.base.train_base import TrainBase
from videotuna.base.inference_base import InferenceBase
from videotuna.utils.common_utils import instantiate_from_config, print_green, print_yellow, get_dist_info
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

class GenerationBase(TrainBase, InferenceBase):
    """
    The GenerationFlow class is a generative model class that inherits from both TrainBase and InferenceBase.
    It manages the instantiation of different stages of a generative process, including a denoiser and a scheduler.
    It also configures optimizers and learning rate schedulers for training.

    The main components of the model are:
        - `first_stage`: a VAE model that encodes the input video into a latent space and decodes it back to the original video.
        - `cond_stage`: a conditional model that takes the latent space and the conditioning text as input and generates the output video.
        - `denoiser`: a denoiser model that takes the noisy output of the `cond_stage` and tries to remove the noise.
        - `scheduler`: a scheduler that controls denosing and sampling.
    """

    def __init__(self,
                 first_stage_config: Dict[str, Any],
                 cond_stage_config: Dict[str, Any],
                 denoiser_config: Dict[str, Any],
                 scheduler_config: Dict[str, Any] = None,
                 cond_stage_2_config: Dict[str, Any] = None,
                 lora_config: Dict[str, Any] = None,
                 trainable_components: Union[str, List[str]] = [],
                 ):
        """
        Initializes the GenerationFlow class with configurations for different stages and components.

        :param first_stage_config: Dictionary containing configuration for the first stage model.
        :param cond_stage_config: Dictionary containing configuration for the conditional stage model.
        :param cond_stage_2_config: Dictionary containing configuration for the conditional stage model 2, can be none.
        :param denoiser_config: Dictionary containing configuration for the denoiser model.
        :param scheduler_config: Dictionary containing configuration for the diffusion scheduler.
        :param trainable_components: The components of the model that should be trainable.
        """
        super().__init__()

        # instantiate the modules
        self.components = []
        # 1. denoiser
        self.instantiate_denoiser(denoiser_config)

        # 2. first stage
        self.instantiate_first_stage(first_stage_config)

        # 3. cond stage
        self.instantiate_cond_stage(cond_stage_config)

        # 4. cond stage 2
        self.instantiate_cond_stage_2(cond_stage_2_config)

        # 5. lora: will set is_lora and lora_params
        self.instantiate_lora(lora_config)

        # 6. scheduler
        self.instantiate_scheduler(scheduler_config)

        # config
        self.first_stage_config = first_stage_config
        self.cond_stage_config = cond_stage_config
        self.cond_stage_2_config = cond_stage_2_config
        self.denoiser_config = denoiser_config
        self.scheduler_config = scheduler_config
        self.lora_config = lora_config
        
        # set trainable components
        # be aware: loaded weight will overide requrie_grad attribute etc
        # make sure call it again after loading weight
        self.set_trainable_components(trainable_components)
        
    def instantiate_scheduler(self, config: Dict[str, Any]):
        if config is not None:
            logger.info("creating scheduler")
            self.diffusion_scheduler = self.scheduler = instantiate_from_config(config)
            self.components.append(Component.SCHEDULER.value)
        
    def instantiate_lora(self, config: Dict[str, Any]):
        self.use_lora = False
        if config is not None:
            logger.info("creating lora")
            transformer_adapter_config = instantiate_from_config(config)   
            self.denoiser = get_peft_model(self.denoiser, transformer_adapter_config)
            self.lora_params = set([name for name, param in self.denoiser.named_parameters() if param.requires_grad and 'lora' in name])
            self.denoiser.requires_grad_(False)
            self.denosier = self.denoiser.eval()
            self.use_lora = True
            self.lora_path = config.get("ckpt_path")
            logger.info(f"self.use_lora: {self.use_lora} self.lora_path: {self.lora_path} self.lora_params: {self.lora_params}")

    def instantiate_first_stage(self, config: Dict[str, Any]):
        """
        Instantiates the first stage model of the generative process.

        :param config: Dictionary containing configuration for the first stage model.
        """
        logger.info("creating first stage")
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
        self.components.append(Component.FIRST_STAGE_MODEL.value)
        self.first_stage_model_path = config.get("ckpt_path", f"{Component.FIRST_STAGE_MODEL.value}.ckpt")
        logger.info(f"self.first_stage_model_path: {self.first_stage_model_path}")
    
    def instantiate_cond_stage(self, config: Dict[str, Any]):
        """
        Instantiates the conditional stage model of the generative process.

        :param config: Dictionary containing configuration for the conditional stage model.
        """
        logger.info("creating cond stage")
        model = instantiate_from_config(config)
        self.cond_stage_model = model.eval()
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False
        self.components.append(Component.COND_STAGE_MODEL.value)
        self.cond_stage_model_path = config.get("ckpt_path", f"{Component.COND_STAGE_MODEL.value}.ckpt")
        logger.info(f"self.cond_stage_model_path: {self.cond_stage_model_path}")
        
    def instantiate_cond_stage_2(self, config: Dict[str, Any]):
        """
        Instantiates the conditional stage model of the generative process.

        :param config: Dictionary containing configuration for the conditional stage model.
        """
        self.cond_stage_2_model = None
        if config is not None:
            logger.info("creating cond stage 2")
            model = instantiate_from_config(config)
            self.cond_stage_2_model = model.eval()
            for param in self.cond_stage_2_model.parameters():
                param.requires_grad = False
            self.components.append(Component.COND_STAGE_2_MODEL.value)
            self.cond_stage_2_model_path = config.get("ckpt_path", f"{Component.COND_STAGE_2_MODEL.value}.ckpt")
            logger.info(f"self.cond_stage_2_model_path: {self.cond_stage_2_model_path}")
    
    def instantiate_denoiser(self, config: Dict[str, Any]):
        """
        Instantiates the denoiser model of the generative process.

        :param config: Dictionary containing configuration for the denoiser model.
        """
        logger.info("creating denoiser")
        model = instantiate_from_config(config)
        self.denoiser = model.eval()
        for param in self.denoiser.parameters():
            param.requires_grad = False
        self.components.append(Component.DENOISER.value)
        self.denoiser_path = config.get("ckpt_path", f"{Component.DENOISER.value}.ckpt")
        logger.info(f"self.denoiser_path: {self.denoiser_path}")

    def configure_lr_config(self, lr_config: Dict[str, Any], bs: int, num_rank: int):
        base_lr = lr_config['base_learning_rate']
        if lr_config.get("scale_lr", True):
            lr_config['learning_rate'] = num_rank * bs * base_lr
        else:
            lr_config['learning_rate'] = base_lr
        self.lr_config = lr_config

    def configure_optimizers(self):
        """
        Configures the optimizers and learning rate schedulers for the generative model.

        :return: A list containing the optimizer and optionally a list containing the learning rate scheduler.
        """
        lr_config = self.lr_config
        lr = lr_config['learning_rate']
        params = [p for p in self.parameters() if p.requires_grad]
        logger.info(f"@Training [{len(params)}] Full Paramters.")

        ## optimizer
        if self.trainer.strategy.__class__.__name__ == 'DeepSpeedStrategy':
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer = DeepSpeedCPUAdam(params, lr=lr)
        else: 
            optimizer = torch.optim.AdamW(params, lr=lr)

        ## lr scheduler
        if lr_config.get('lr_scheduler_config', None):
            logger.info("Setting up LambdaLR scheduler...")
            lr_scheduler = self.configure_lr_schedulers(optimizer)
            return [optimizer], [lr_scheduler]
        
        return optimizer

    def configure_lr_schedulers(self, optimizer):
        """
        Configures the learning rate scheduler based on the provided configuration.

        :param optimizer: The optimizer for which the scheduler is being configured.
        :return: A dictionary containing the scheduler, interval, and frequency.
        """
        lr_scheduler_config = self.lr_config.lr_scheduler_config
        assert 'target' in lr_scheduler_config
        scheduler_name = lr_scheduler_config.target.split('.')[-1]
        interval = lr_scheduler_config.interval
        frequency = lr_scheduler_config.frequency
        if scheduler_name == "LambdaLRScheduler":
            scheduler = instantiate_from_config(lr_scheduler_config)
            scheduler.start_step = self.global_step
            lr_scheduler = {
                            'scheduler': LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                            'interval': interval,
                            'frequency': frequency
            }
        elif scheduler_name == "CosineAnnealingLRScheduler":
            scheduler = instantiate_from_config(lr_scheduler_config)
            decay_steps = scheduler.decay_steps
            last_step = -1 if self.global_step == 0 else scheduler.start_step
            lr_scheduler = {
                            'scheduler': CosineAnnealingLR(optimizer, T_max=decay_steps, last_epoch=last_step),
                            'interval': interval,
                            'frequency': frequency
            }
        else:
            raise NotImplementedError
        return lr_scheduler
    
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
                logger.info(f"Skipping eval component {component} since it is not set or not module")
                continue

            model.eval()
            model.requires_grad_(False)

        # train selected components
        for component in components:
            model = getattr(self, component)
            if model is None:
                raise ValueError(f"Invalid component name: {component}")
        
            if not isinstance(model, nn.Module):
                logger.info(f"Skipping train component {component} since it is not module")
                continue
            
            #if denoiser lora, make sure only lora params require grad
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
    
    
    def load_first_stage(self, 
                         ckpt_path: str,
                         ignore_missing_ckpts: bool = False):
        path = os.path.join(ckpt_path, self.first_stage_model_path)
        if os.path.exists(path):
            self.first_stage_model = self.load_model(self.first_stage_model, path)
            print_green("Successfully loaded first_stage_model from checkpoint.")
        elif ignore_missing_ckpts:
            print_yellow("Checkpoint of first_stage_model file not found. Ignoring.")
        else:
            raise FileNotFoundError("Checkpoint of first_stage_model file not found.")

    
    def load_cond_stage(self, 
                         ckpt_path: str,
                         ignore_missing_ckpts: bool = False):
        path = os.path.join(ckpt_path, self.cond_stage_model_path)
        if os.path.exists(path):
            self.cond_stage_model = self.load_model(self.cond_stage_model, path)
            print_green("Successfully loaded cond_stage_model from checkpoint.")
        elif ignore_missing_ckpts:
            print_yellow("Checkpoint of cond_stage_model file not found. Ignoring.")
        else:
            raise FileNotFoundError("Checkpoint of cond_stage_model file not found.")

    def load_cond_stage_2(self, 
                         ckpt_path: str,
                         ignore_missing_ckpts: bool = False):
        if self.cond_stage_2_model is None:
            return
        
        path = os.path.join(ckpt_path, self.cond_stage_2_model_path)
        if os.path.exists(path):
            self.cond_stage_2_model = self.load_model(self.cond_stage_2_model, path)
            print_green("Successfully loaded cond_stage_2_model from checkpoint.")
        elif ignore_missing_ckpts:
            print_yellow("Checkpoint of cond_stage_2_model file not found. Ignoring.")
        else:
            raise FileNotFoundError("Checkpoint of cond_stage_2_model file not found.")
    def load_denoiser(self, 
                    ckpt_path: str = None,
                    denoiser_ckpt_path: str = None,
                    ignore_missing_ckpts: bool = False):
        path = os.path.join(ckpt_path, self.denoiser_path)
        if denoiser_ckpt_path is not None:
            path = denoiser_ckpt_path

        if os.path.exists(path):
            self.denoiser = self.load_model(self.denoiser, path)
            print_green("Successfully loaded denoiser from checkpoint.")
        elif ignore_missing_ckpts:
            print_yellow("Checkpoint of denoiser file not found. Ignoring.")
        else:
            raise FileNotFoundError("Checkpoint of denoiser file not found.")
            
    def load_lora(self,
                lora_ckpt_path: str = None,
                ignore_missing_ckpts: bool = False):
        if not self.use_lora:
            return
        
        lora_path = self.lora_path
        if lora_ckpt_path is not None:
            lora_path = lora_ckpt_path
        
        if os.path.exists(lora_path):
            self.load_model(self.denoiser, lora_path, strict=False)
            print_green("Successfully loaded denoiser from checkpoint.")
        elif ignore_missing_ckpts:
            print_yellow("Checkpoint of denoiser file not found. Ignoring.")
        else:
            raise FileNotFoundError("Checkpoint of denoiser file not found.")
        
    def from_pretrained(self,
                        ckpt_path: Optional[Union[str, Path]] = None,
                        denoiser_ckpt_path: Optional[Union[str, Path]] = None,
                        lora_ckpt_path: Optional[Union[str, Path]] = None,
                        ignore_missing_ckpts: bool = False) -> None:
        assert ckpt_path is not None, "Please provide a valid checkpoint path."

        #can ovrride following methods
        self.load_first_stage(ckpt_path, ignore_missing_ckpts)
        self.load_cond_stage(ckpt_path, ignore_missing_ckpts)
        self.load_cond_stage_2(ckpt_path, ignore_missing_ckpts)
        self.load_denoiser(ckpt_path, denoiser_ckpt_path, ignore_missing_ckpts)
        self.load_lora(lora_ckpt_path, ignore_missing_ckpts)
    
    def enable_vram_management(self):
        logger.info("enable_vram_management: default moving to cuda")
        self.cuda()
    

    def enable_cpu_offload(self):
        self.cpu_offload = True


    def load_models_to_device(self, loadmodel_names=[], device='cuda'):
        skip_components = ['scheduler']
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            logger.info("cpu offload is closed, skipping")
            return
        # offload the unneeded models to cpu
        for model_name in self.components:
            if model_name in skip_components:
                logger.info(f"{model_name} no need cpu offload, skipping")
                continue

            if model_name not in loadmodel_names:
                model = getattr(self, model_name)
                if model is not None:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        logger.info(f"{model_name} cpu offloading using offload method")
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        logger.info(f"{model_name} cpu offloading using to cpu method")
                        model.cpu()

        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if model is not None:
                if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                    logger.info(f"{model_name} onloading using onload method")
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()
                else:
                    logger.info(f"{model_name} onloading using to device method")
                    model.to(device)
        # fresh the cuda cache
        torch.cuda.empty_cache()
    
    @staticmethod
    def load_model(model: nn.Module, ckpt_path: Optional[Union[str, Path]] = None, strict=True):
        """
        Loads the weights of the model from a checkpoint file.

        :param model: The model to be loaded.
        :param ckpt_path: Path to the checkpoint file.
        """
        assert ckpt_path is not None, "Please provide a valid checkpoint path."

        ckpt_path = Path(ckpt_path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            logger.info(f"{num_updated_keys} parameters are loaded from {ckpt_path}. {num_unexpected_keys} parameters are unexpected.")
            return model
        else:
            raise FileNotFoundError("Checkpoint of model file not found.")

    
    def init_trainer(self, train_config: DictConfig):
        # 1. basic info setup
        local_rank, global_rank, num_rank = get_dist_info()

        debug = train_config['debug']
        workdir = train_config['workdir']
        ckptdir = train_config['ckptdir']
        lightning_config: DictConfig = train_config.get("lightning")
        trainer_config: DictConfig = lightning_config.get("trainer")
        self.first_stage_key = train_config.first_stage_key
        self.cond_stage_key = train_config.cond_stage_key
        self.logdir = workdir

        # 2. lr
        lr_config: DictConfig = train_config.get("lr_config")
        bs = train_config['data']['params']['batch_size']
        self.lr_config = OmegaConf.to_container(lr_config, resolve=True)
        self.configure_lr_config(self.lr_config, bs=bs, num_rank=num_rank)
        
        # 3. dataset
        logger.info("***** Configuring Data *****")
        data = instantiate_from_config(train_config['data'])
        self.data = data
        data.setup()
        for k in data.datasets:
            logger.info(
                f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}"
            )

        ## 4. lightning trainer config
        logger.info(f"trainer_config: {trainer_config}")
        num_nodes = trainer_config['num_nodes']
        ngpu_per_node = trainer_config['devices']
        logger.info(f"Running on {num_rank}={num_nodes}x{ngpu_per_node} GPUs")
        logger.info("***** Configuring Trainer *****")

        # 4.1 trainer gpu
        if "accelerator" not in trainer_config:
            trainer_config["accelerator"] = "gpu"

        ## 4.2 logger
        trainer_kwargs = dict()
        trainer_kwargs["num_sanity_val_steps"] = 0
        logger_cfg = get_trainer_logger(lightning_config, workdir, debug)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
        logger.info(f"logger save_dir: {trainer_kwargs['logger'].save_dir}")

        ## 4.3 callback
        callbacks_cfg = get_trainer_callbacks(
            lightning_config, workdir, ckptdir
        )
        callbacks_cfg["image_logger"]["params"]["save_dir"] = workdir
        trainer_kwargs["callbacks"] = [
            instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
        ]

        ## 4.4 strategy
        strategy_cfg = get_trainer_strategy(lightning_config)
        trainer_kwargs["strategy"] = (
            strategy_cfg
            if type(strategy_cfg) == str
            else instantiate_from_config(OmegaConf.to_container(strategy_cfg))
        )
        trainer_kwargs["sync_batchnorm"] = False

        ## 4.5 create Trainer
        logger.info(f"trainer_kwargs: {trainer_kwargs}")
        from pytorch_lightning.profilers import PyTorchProfiler
        profiler = PyTorchProfiler(emit_nvtx=True)
        trainer = Trainer(**trainer_config, **trainer_kwargs,  profiler=profiler)
        self.trainer = trainer

        ## 5. allow user
        def melk(*args, **kwargs):
            ## run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last_summoning.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb
                pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        ## since loaded weight will ovrride params, make sure it is been handled
        if trainer.strategy.__class__.__name__ == 'DeepSpeedStrategy':
            logger.info(f"Make parameter contiguous in case deepseed does not allow non contigouous data")
            for param in self.parameters(): param.data = param.data.contiguous()
        self.set_trainable_components([Component.DENOISER.value])