import datetime
import logging
import os
import time

import numpy as np
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from weakref import proxy
from collections import OrderedDict
from typing_extensions import override
from typing import Any, Literal, Optional, Union
from loguru import logger

mainlogger = logging.getLogger("mainlogger")

import pytorch_lightning as pl
import torch
import torchvision
from torch import Tensor
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT

from .save_video import log_local, prepare_to_log


class LoraModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """
        the hook in pl.module and ModelCheckpoint is slight different.
        pl.Module: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-save-checkpoint
        ModelCheckpoint: https://pytorch-lightning.readthedocs.io/en/1.5.10/extensions/generated/pytorch_lightning.callbacks.ModelCheckpoint.html
        """
        # only save lora
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        for k in list(state_dict.keys()):
            if "lora" not in k:
                del state_dict[k]

        if "state_dict" in checkpoint:
            checkpoint["state_dict"] = state_dict
        else:
            checkpoint = state_dict

        return checkpoint


class VideoTunaModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, 
                 save_flow: bool = True,
                 save_only_selected_model: bool = True,
                 selected_model: Optional[Union[str, list]] = None,
                 *args, **kwargs):
        assert save_flow or save_only_selected_model, "At least one of `save_flow` and `save_only_trained_model` should be True."
        super().__init__(*args, **kwargs)
        self.save_flow = save_flow
        self.save_only_selected_model = save_only_selected_model
        self.selected_model = selected_model
    
    @override
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Save checkpoint on train batch end if we meet the criteria for `every_n_train_steps`"""
        if self._should_skip_saving_checkpoint(trainer):
            return
        skip_batch = self._every_n_train_steps < 1 or (trainer.global_step % self._every_n_train_steps != 0)

        train_time_interval = self._train_time_interval
        skip_time = True
        now = time.monotonic()
        if train_time_interval:
            prev_time_check = self._last_time_checked
            skip_time = prev_time_check is None or (now - prev_time_check) < train_time_interval.total_seconds()
            # in case we have time differences across ranks
            # broadcast the decision on whether to checkpoint from rank 0 to avoid possible hangs
            skip_time = trainer.strategy.broadcast(skip_time)

        if skip_batch and skip_time:
            return
        if not skip_time:
            self._last_time_checked = now

        monitor_candidates = self._monitor_candidates(trainer)
        self._save_last_checkpoint(trainer, monitor_candidates, pl_module)  # only save the last checkpoint
    
    @override
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    @override
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    @override
    def _save_last_checkpoint(
        self,
        trainer: "pl.Trainer",
        monitor_candidates: dict[str, Tensor],
        pl_module: "pl.LightningModule",
    ) -> None:
        if not self.save_last:
            return

        # filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST)
        filepath = self._format_ckpt_path(monitor_candidates, prefix="flow")

        if self._enable_version_counter:
            version_cnt = self.STARTING_VERSION
            while self.file_exists(filepath, trainer) and filepath != self.last_model_path:
                filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST, ver=version_cnt)
                version_cnt += 1

        # set the last model path before saving because it will be part of the state.
        previous, self.last_model_path = self.last_model_path, filepath
        if self.save_last == "link" and self._last_checkpoint_saved and self.save_top_k != 0:
            self._link_checkpoint(trainer, self._last_checkpoint_saved, filepath)
        else:
            self._save_checkpoint(trainer, filepath, pl_module)
        if previous and self._should_remove_checkpoint(trainer, previous, filepath):
            self._remove_checkpoint(trainer, previous)

    @override
    def _save_checkpoint(
        self,
        trainer: "pl.Trainer",
        filepath: str,
        pl_module: "pl.LightningModule",
    ) -> None:
        if self.save_flow:
            # save all the state including the model, optimizer, and any state that the user has added
            self._save_flow_checkpoint(trainer, pl_module, filepath)
        if self.save_only_selected_model:
            # only save the trained parameters
            self._save_training_checkpoint(trainer, pl_module, filepath)

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))
    
    def _save_flow_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        filepath
    ) -> None:
        """Save the whole model."""
        # check the save path
        original_dirpath_list = filepath.split('/')
        new_dirpath_list = original_dirpath_list[:-1] + ['flow']
        new_dirpath = '/'.join(new_dirpath_list)
        if not os.path.exists(new_dirpath):
            os.makedirs(new_dirpath)

        new_filepath = os.path.join(new_dirpath, original_dirpath_list[-1])
        trainer.save_checkpoint(new_filepath, self.save_weights_only)
    
    @rank_zero_only
    def _save_training_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        filepath
    ) -> None:
        """Save only the trained model."""
        # check the save path
        original_dirpath_list = filepath.split('/')
        new_dirpath_list = original_dirpath_list[:-1] + ['only_trained_model']
        new_dirpath = '/'.join(new_dirpath_list)
        if not os.path.exists(new_dirpath):
            os.makedirs(new_dirpath)

        if trainer.strategy.__class__.__name__  == "DeepSpeedStrategy":
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
            original_filename = original_dirpath_list[-1]
            deepspeed_flow_path = original_dirpath_list[:-1] + ['flow', original_filename]
            state_dict = get_fp32_state_dict_from_zero_checkpoint('/'.join(deepspeed_flow_path))
    
            for seleted in self.selected_model:
                new_state_dict = {name.replace(f"{seleted}.", ""): param for name, param in state_dict.items() if name.startswith(seleted)}
                save_dict = {'state_dict': new_state_dict}
                new_filename = original_filename.replace('flow', seleted)
                new_filepath = os.path.join(new_dirpath, new_filename)
                torch.save(save_dict, new_filepath)
                logger.info(f"Deepspeed Saving model {seleted} with {len(new_state_dict)} params to {new_filepath}")
        else:
            original_filename = original_dirpath_list[-1]
            for seleted in self.selected_model:
                model = getattr(pl_module, seleted)
                state_dict = model.state_dict()
                save_dict = {'state_dict': state_dict}
                new_filename = original_filename.replace('flow', seleted)
                new_filepath = os.path.join(new_dirpath, new_filename)
                torch.save(save_dict, new_filepath)
                logger.info(f"Saving model {seleted} with {len(state_dict)} params  to {new_filepath}")
    
    def _format_ckpt_path(
        self,
        monitor_candidates: dict[str, Tensor],
        prefix: str = None
    ) -> str:
        """Format the checkpoint path with the current values of monitored quantities."""
        epoch = monitor_candidates.get("epoch").item()
        step = monitor_candidates.get("step").item()

        if 'epoch' in self.filename and 'step' in self.filename:
            format_filename = self.filename.format(epoch=epoch, step=step)
        elif 'epoch' in self.filename and 'step' not in self.filename:
            format_filename = self.filename.format(epoch=epoch)
        elif 'epoch' not in self.filename and 'step' in self.filename:
            format_filename = self.filename.format(step=step)
        else:
            format_filename = self.filename
        
        if prefix is not None:
            format_filename = prefix + '-' + format_filename + '.ckpt'
    
        filepath = os.path.join(self.dirpath, format_filename)
        
        return filepath


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        save_dir,
        max_images=8,
        clamp=True,
        rescale=True,
        to_local=False,
        log_images_kwargs=None,
    ):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.to_local = to_local
        self.clamp = clamp
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        if self.to_local:
            ## default save dir
            self.save_dir = os.path.join(save_dir, "images")
            os.makedirs(os.path.join(self.save_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "val"), exist_ok=True)

    def log_to_tensorboard(self, pl_module, batch_logs, filename, split, save_fps=10):
        """log images and videos to tensorboard"""
        global_step = pl_module.global_step
        for key in batch_logs:
            value = batch_logs[key]
            tag = "gs%d-%s/%s-%s" % (global_step, split, filename, key)
            if isinstance(value, list) and isinstance(value[0], str):
                captions = " |------| ".join(value)
                pl_module.logger.experiment.add_text(
                    tag, captions, global_step=global_step
                )
            elif isinstance(value, torch.Tensor) and value.dim() == 5:
                video = value
                n = video.shape[0]
                video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w
                frame_grids = [
                    torchvision.utils.make_grid(framesheet, nrow=int(n))
                    for framesheet in video
                ]  # [3, n*h, 1*w]
                grid = torch.stack(
                    frame_grids, dim=0
                )  # stack in temporal dim [t, 3, n*h, w]
                grid = (grid + 1.0) / 2.0
                grid = grid.unsqueeze(dim=0)
                pl_module.logger.experiment.add_video(
                    tag, grid, fps=save_fps, global_step=global_step
                )
            elif isinstance(value, torch.Tensor) and value.dim() == 4:
                img = value
                grid = torchvision.utils.make_grid(img, nrow=int(n))
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                pl_module.logger.experiment.add_image(
                    tag, grid, global_step=global_step
                )
            else:
                pass

    @rank_zero_only
    def log_batch_imgs(self, pl_module, batch, batch_idx, split="train"):
        """generate images, then save and log to tensorboard"""
        skip_freq = self.batch_freq if split == "train" else 5
        if (batch_idx + 1) % skip_freq == 0:
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                log_func = pl_module.log_images
                batch_logs = log_func(batch, split=split, **self.log_images_kwargs)

            ## process: move to CPU and clamp
            batch_logs = prepare_to_log(batch_logs, self.max_images, self.clamp)
            torch.cuda.empty_cache()

            filename = "ep{}_idx{}_rank{}".format(
                pl_module.current_epoch, batch_idx, pl_module.global_rank
            )
            if self.to_local:
                mainlogger.info("Log [%s] batch <%s> to local ..." % (split, filename))
                filename = "gs{}_".format(pl_module.global_step) + filename
                log_local(
                    batch_logs,
                    os.path.join(self.save_dir, split),
                    filename,
                    save_fps=10,
                )
            else:
                mainlogger.info(
                    "Log [%s] batch <%s> to tensorboard ..." % (split, filename)
                )
                self.log_to_tensorboard(
                    pl_module, batch_logs, filename, split, save_fps=10
                )
            mainlogger.info("Finish!")

            if is_train:
                pl_module.train()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None
    ):
        if self.batch_freq != -1 and pl_module.logdir:
            self.log_batch_imgs(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None
    ):
        ## different with validation_step() that saving the whole validation set and only keep the latest,
        ## it records the performance of every validation (without overwritten) by only keep a subset
        if self.batch_freq != -1 and pl_module.logdir:
            self.log_batch_imgs(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, "calibrate_grad_norm"):
            if (
                pl_module.calibrate_grad_norm and batch_idx % 25 == 0
            ) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int):
        # Reset the memory use counter
        # lightning update
        gpu_index = trainer.strategy.root_device.index
        torch.cuda.reset_peak_memory_stats(gpu_index)
        torch.cuda.synchronize(gpu_index)
        self.start_time = time.time()

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int):
        gpu_index = trainer.strategy.root_device.index
        torch.cuda.synchronize(gpu_index)
        max_memory = torch.cuda.max_memory_allocated(gpu_index) / 2**20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
            pass
