import logging
import os
import json
import random
import time
import numpy as np
from einops import rearrange, repeat
from tqdm import tqdm, trange
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import make_grid

from videotuna.utils.ema import LitEma
from videotuna.models.lvdm.ddpm3d import DiffusionWrapper
from videotuna.utils.distributions import DiagonalGaussianDistribution
from videotuna.schedulers.ddim import DDIMSampler
from videotuna.base.generation_base import GenerationBase
from videotuna.utils.common_utils import instantiate_from_config, print_green, print_yellow
from videotuna.models.lvdm.modules.utils import (
    default,
    disabled_train,
    exists,
    extract_into_tensor,
    noise_like,
)

mainlogger = logging.getLogger("mainlogger")



class VideocrafterFlow(GenerationBase):
    """
    Training and inference flow for VideoCrafter.

    THis model inherits from GenerationFlow, which is a base class for all generative models.

    The main components of the model are:
        - `first_stage`: a VAE model that encodes the input video into a latent space and decodes it back to the original video.
        - `cond_stage`: a conditional model that takes the latent space and the conditioning text as input and generates the output video.
        - `denoiser`: a denoiser model that takes the noisy output of the `cond_stage` and tries to remove the noise, which is the most important part of the model.
        - `scheduler`: a scheduler that controls denosing and sampling.
    """

    def __init__(
        self,
        first_stage_config: Dict[str, Any],
        cond_stage_config: Dict[str, Any],
        denoiser_config: Dict[str, Any],
        scheduler_config: Optional[Dict[str, Any]] = None,
        cond_stage_2_config: Optional[Dict[str, Any]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        loss_type: str = "l2",
        ckpt_path: Optional[Union[str, Path]] = None,
        ignore_keys: List[str] = [],
        load_only_unet: bool = False,
        monitor: Optional[str] = None,
        use_ema: bool = True,
        first_stage_key: str = "image",
        image_size: int = 256,
        channels: int = 3,
        log_every_t: int = 100,
        clip_denoised: bool = True,
        original_elbo_weight: float = 0.0,
        l_simple_weight: float = 1.0,
        conditioning_key: Optional[str] = None,
        parameterization: str = "eps",  # all assuming fixed variance schedules
        use_positional_encodings: bool = False,
        cond_stage_key: str = "caption",
        cond_stage_trainable: bool = False,
        cond_stage_forward: Optional[callable] = None,
        uncond_prob: float = 0.2,
        uncond_type: str = "empty_seq",
        scale_factor: float = 1.0,
        scale_by_std: bool = False,
        fps_condition_type: str = 'fs',
        # added for LVDM
        encoder_type: str = "2d",
        frame_cond: Optional[Dict[str, Any]] = None,
        only_model: bool = False,
        use_scale: bool = False,  # dynamic rescaling
        scale_a: int = 1,
        scale_b: float = 0.3,
        mid_step: int = 400,
        fix_scale_bug: bool = False,
        interp_mode: bool = False,
        logdir: Optional[Union[str, Path]] = None,
        rand_cond_frame: bool = False,
        empty_params_only: bool = False,
        *args, **kwargs
    ):
        super().__init__(
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            cond_stage_2_config=cond_stage_2_config,
            denoiser_config=denoiser_config,
            scheduler_config=scheduler_config,
            lora_config=lora_config,
        )
        # DDPMFlow related
        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        mainlogger.info(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")

        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t

        # model related 
        self.first_stage_key = first_stage_key
        self.channels = channels
        self.temporal_length = denoiser_config['params'].get('temporal_length', 16)
        self.image_size = image_size
        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(self.denoiser, conditioning_key)

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            mainlogger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        print('scheduler config type: ', type(scheduler_config))
        scheduler_config['parameterization'] = self.parameterization
        self.num_timesteps = self.scheduler.num_timesteps

        # others 
        if monitor is not None:
            self.monitor = monitor
        
        self.loss_type = loss_type

        # LVDM related
        self.scale_by_std = scale_by_std
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        conditioning_key = default(conditioning_key, 'crossattn')

        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.empty_params_only = empty_params_only
        self.fps_condition_type = fps_condition_type

        # scale factor
        self.use_scale=use_scale
        if self.use_scale:
            self.scale_a=scale_a
            self.scale_b=scale_b
            if fix_scale_bug:
                scale_step=self.num_timesteps-mid_step
            else: #bug
                scale_step = self.num_timesteps

            scale_arr1 = np.linspace(scale_a, scale_b, mid_step)
            scale_arr2 = np.full(scale_step, scale_b)
            scale_arr = np.concatenate((scale_arr1, scale_arr2))
            scale_arr_prev = np.append(scale_a, scale_arr[:-1])
            to_torch = partial(torch.tensor, dtype=torch.float32)
            self.register_buffer('scale_arr', to_torch(scale_arr))

        try:
            self.num_downs = len(first_stage_config['params'].ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        
        self.clip_denoised = False
        self.cond_stage_forward = cond_stage_forward
        self.encoder_type = encoder_type
        assert(encoder_type in ["2d", "3d"])
        self.uncond_prob = uncond_prob
        self.classifier_free_guidance = True if uncond_prob > 0 else False
        assert(uncond_type in ["zero_embed", "empty_seq"])
        self.uncond_type = uncond_type

        # future frame prediction
        self.frame_cond = frame_cond
        if self.frame_cond:
            frame_len = self.temporal_length
            cond_mask = torch.zeros(frame_len, dtype=torch.float32)
            cond_mask[:self.frame_cond] = 1.0
            self.cond_mask = cond_mask[None,None,:,None,None]
            mainlogger.info("---training for %d-frame conditoning T2V"%(self.frame_cond))
        else:
            self.cond_mask = None
                
        self.logdir = logdir
        self.rand_cond_frame = rand_cond_frame
        self.interp_mode = interp_mode
    
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                mainlogger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    mainlogger.info(f"{context}: Restored training weights")
    
    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=None):
        # only for very first batch, reset the self.scale_factor
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            mainlogger.info("### USING STD-RESCALING ###")
            x = self.get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            mainlogger.info(f"setting self.scale_factor to {self.scale_factor}")
            mainlogger.info("### USING STD-RESCALING ###")
            mainlogger.info(f"std={z.flatten().std()}")
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)
    
    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_first_stage_encoding(self, encoder_posterior, noise=None):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample(noise=noise)
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z
    
    @torch.no_grad()
    def encode_first_stage(self, x):
        if self.encoder_type == "2d" and x.dim() == 5:
            return self.encode_first_stage_2DAE(x)
        encoder_posterior = self.first_stage_model.encode(x)
        results = self.get_first_stage_encoding(encoder_posterior).detach()
        return results
    
    def encode_first_stage_2DAE(self, x):
        """encode frame by frame"""
        b, _, t, _, _ = x.shape
        results = torch.cat([self.get_first_stage_encoding(self.first_stage_model.encode(x[:,:,i])).detach().unsqueeze(2) for i in range(t)], dim=2)
        return results
    
    def decode_first_stage_2DAE(self, z, **kwargs):
        """decode frame by frame"""
        _, _, t, _, _ = z.shape
        results = torch.cat([self.first_stage_model.decode(z[:,:,i], **kwargs).unsqueeze(2) for i in range(t)], dim=2)
        return results

    def _decode_core(self, z, **kwargs):
        z = 1. / self.scale_factor * z

        if self.encoder_type == "2d" and z.dim() == 5:
            return self.decode_first_stage_2DAE(z)
        results = self.first_stage_model.decode(z, **kwargs)
        return results

    @torch.no_grad()
    def decode_first_stage(self, z, **kwargs):
        return self._decode_core(z, **kwargs)

    def differentiable_decode_first_stage(self, z, **kwargs):
        """same as decode_first_stage but without decorator"""
        return self._decode_core(z, **kwargs)
    
    def get_input(self, batch, k):
        x = batch[k]
        """
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        """
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, is_imgbatch=False):
        ## image/video shape: b, c, t, h, w
        data_key = 'jpg' if is_imgbatch else self.first_stage_key
        x = self.get_input(batch, data_key)
        if is_imgbatch:
            ## pack image as video
            #x = x[:,:,None,:,:]
            b = x.shape[0] // self.temporal_length
            x = rearrange(x, '(b t) c h w -> b c t h w', b=b, t=self.temporal_length)
        x_ori = x
        ## encode video frames x to z via a 2D encoder
        z = self.encode_first_stage(x)
                
        ## get caption condition
        cond_key = 'txt' if is_imgbatch else self.cond_stage_key
        cond = batch[cond_key]
        if random_uncond and self.uncond_type == 'empty_seq':
            for i, ci in enumerate(cond):
                if random.random() < self.uncond_prob:
                    cond[i] = ""
        if isinstance(cond, dict) or isinstance(cond, list):
            cond_emb = self.get_learned_conditioning(cond)
        else:
            cond_emb = self.get_learned_conditioning(cond.to(self.device))
        if random_uncond and self.uncond_type == 'zero_embed':
            for i, ci in enumerate(cond):
                if random.random() < self.uncond_prob:
                    cond_emb[i] = torch.zeros_like(ci)

        out = [z, cond_emb]
        ## optional output: self-reconst or caption
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x_ori, xrec])
        if return_original_cond:
            out.append(cond)

        return out

    def forward(self, x, c, **kwargs):
        if 't' in kwargs:
            t = kwargs.pop('t')
        else:
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.use_scale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, **kwargs)

    def shared_step(self, batch, random_uncond, **kwargs):
        is_imgbatch = False
        if "loader_img" in batch.keys():
            ratio = 10.0 / self.temporal_length
            if random.uniform(0.,10.) < ratio:
                is_imgbatch = True
                batch = batch["loader_img"]
            else:
                batch = batch["loader_video"]
        else:
            pass

        x, c = self.get_batch_input(batch, random_uncond=random_uncond, is_imgbatch=is_imgbatch)
        loss, loss_dict = self(x, c, is_imgbatch=is_imgbatch, **kwargs)
        return loss, loss_dict
    
    def apply_model(self, x_noisy, t, cond, **kwargs):
        if self.model.conditioning_key == "crossattn_stdit":
            key = "c_crossattn_stdit"
            cond = {key: [cond["y"]], "mask": [cond["mask"]]}  # support mask for T5
        else:
            if isinstance(cond, dict):
                # hybrid case, cond is exptected to be a dict
                pass
            else:
                if not isinstance(cond, list):
                    cond = [cond]
                key = (
                    "c_concat"
                    if self.model.conditioning_key == "concat"
                    else "c_crossattn"
                )
                cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond, **kwargs)

        if isinstance(x_recon, tuple):
            return x_recon[0]
        else:
            return x_recon
    
    def get_loss(self, pred, target, mean=True):

        if target.size()[1] != pred.size()[1]:
            c = target.size()[1]
            pred = pred[
                :, :c, ...
            ]  # opensora, only previous 4 channels used for calculating loss.

        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss
    
    def p_losses(self, x_start, cond, t, noise=None, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.scheduler.q_sample(x_start=x_start, t=t, noise=noise)
        if self.frame_cond:
            if self.cond_mask.device is not self.device:
                self.cond_mask = self.cond_mask.to(self.device)
            ## condition on fist few frames
            x_noisy = x_start * self.cond_mask + (1.-self.cond_mask) * x_noisy
        model_output = self.apply_model(x_noisy, t, cond, **kwargs)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.scheduler.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()
        
        if self.frame_cond:
            ## [b,c,t,h,w]: only care about the predicted part (avoid disturbance)
            model_output = model_output[:,:,self.frame_cond:,:,:]
            target = target[:,:,self.frame_cond:,:,:]
        
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])

        if torch.isnan(loss_simple).any():
            print(f"loss_simple exists nan: {loss_simple}")
            for i in range(loss_simple.shape[0]):
                if torch.isnan(loss_simple[i]).any():
                    loss_simple[i] = torch.zeros_like(loss_simple[i])

        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        if self.scheduler.logvar.device is not self.device:
            self.scheduler.logvar = self.scheduler.logvar.to(self.device)
        logvar_t = self.scheduler.logvar[t]
        # logvar_t = self.logvar[t.item()].to(self.device) # device conflict when ddp shared
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.scheduler.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.scheduler.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        if self.original_elbo_weight > 0:
            loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3, 4))
            loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
            loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
            loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict   

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, random_uncond=self.classifier_free_guidance)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        #self.log("epoch/global_step", self.global_step.float(), prog_bar=True, logger=True, on_step=True, on_epoch=False)
        '''
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        '''
        if (batch_idx+1) % self.log_every_t == 0:
            mainlogger.info(f"batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}")
        return loss
    
    def _get_denoise_row_from_list(self, samples, desc=''):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_log_timesteps = len(denoise_row)

        denoise_row = torch.stack(denoise_row)  # n_log_timesteps, b, C, H, W
        
        if denoise_row.dim() == 5:
            # img, num_imgs= n_log_timesteps * bs, grid_size=[bs,n_log_timesteps]
            # batch:col, different samples, 
            # n:rows, different steps for one sample
            denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
            denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=n_log_timesteps)
        elif denoise_row.dim() == 6:
            # video, grid_size=[n_log_timesteps*bs, t]
            video_length = denoise_row.shape[3]
            denoise_grid = rearrange(denoise_row, 'n b c t h w -> b n c t h w')
            denoise_grid = rearrange(denoise_grid, 'b n c t h w -> (b n) c t h w')
            denoise_grid = rearrange(denoise_grid, 'n c t h w -> (n t) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=video_length)
        else:
            raise ValueError

        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, sample=True, ddim_steps=200, ddim_eta=1., plot_denoise_rows=False, \
                    unconditional_guidance_scale=1.0, **kwargs):
        """ log images for LatentDiffusion """
        ## TBD: currently, classifier_free_guidance sampling is only supported by DDIM
        use_ddim = ddim_steps is not None
        log = dict()
        z, c, x, xrec, xc = self.get_batch_input(batch, random_uncond=False,
                                                return_first_stage_outputs=True,
                                                return_original_cond=True)
        N, _, T, H, W = x.shape
        # TODO fix data type 
        log["inputs"] = x.to(torch.bfloat16)
        log["reconst"] = xrec
        log["condition"] = xc
        
        if sample:
            # get uncond embedding for classifier-free guidance sampling
            if unconditional_guidance_scale != 1.0:
                if isinstance(c, dict):
                    if "y" in c:
                        c_emb = c["y"]
                        c_cat = None # set default value is None
                    else:
                        c_cat, c_emb = c["c_concat"][0], c["c_crossattn"][0]
                else:
                    c_emb = c
                
                # TODO fix data type 
                z = z.to(torch.bfloat16)
                c_emb = c_emb.to(torch.bfloat16)

                # get uc: unconditional condition for classifier-free guidance sampling
                if self.uncond_type == "empty_seq":
                    prompts = N * [""]
                    uc = self.get_learned_conditioning(prompts)
                elif self.uncond_type == "zero_embed":
                    uc = torch.zeros_like(c_emb)
                # make uc for hybrid condition case
                if isinstance(c, dict) and c_cat is not None:
                    uc = {"c_concat": [c_cat], "c_crossattn": [uc]}
            else:
                uc = None

            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta,
                                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                                         unconditional_conditioning=uc, mask=self.cond_mask, x0=z, **kwargs)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log
    
    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False, x_T=None, verbose=True, callback=None, \
                      timesteps=None, mask=None, x0=None, img_callback=None, start_T=None, log_every_t=None, **kwargs):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.device
        b = shape[0]        
        # sample an initial noise
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)

        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.scheduler.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.scheduler.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.scheduler.p_sample(img, cond, ts, clip_denoised=self.clip_denoised, **kwargs)
            if mask is not None:
                img_orig = self.scheduler.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None, \
               verbose=True, timesteps=None, mask=None, x0=None, shape=None, **kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.temporal_length, *self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps,
                                  mask=mask, x0=x0, **kwargs)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):        
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.temporal_length, *self.image_size)
            # kwargs.update({"clean_cond": True})
            samples, intermediates =ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size, return_intermediates=True, **kwargs)

        return samples, intermediates
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch, random_uncond=False)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch, random_uncond=False)
            loss_dict_ema = {key + "_ema": loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(
            loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True
        )
        self.log_dict(
            loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True
        )
    
    def sample_batch_t2v(
            self,
            prompts: List[str],
            fps: int,
            noise_shape: Optional[tuple] = None,
            n_samples_prompt: int = 1,
            ddim_steps: int = 50,
            ddim_eta: float = 1.0,
            cfg_scale: float = 1.0,
            temporal_cfg_scale: Optional[float] = None,
            uncond_prompt: str = "",
            **kwargs,
        ) -> None:
        """
        Sample a batch of text-to-video (T2V) sequences.

        :param model: The model used for generating the video.
        :param sampler: The sampler used for sampling the video frames.
        :param prompts: A list of text prompts for generating the video.
        :param noise_shape: The shape of the noise input for the model.
        :param fps: Frames per second for the generated video.
        :param n_samples_prompt: Number of samples per prompt. Default is 1.
        :param ddim_steps: Number of DDIM steps for the sampling process. Default is 50.
        :param ddim_eta: The eta parameter for DDIM. Default is 1.0.
        :param cfg_scale: The scale for classifier-free guidance. Default is 1.0.
        :param temporal_cfg_scale: The scale for temporal classifier-free guidance. Default is None.
        :param uncond_prompt: The unconditional prompt for classifier-free guidance. Default is an empty string.
        :param kwargs: Additional keyword arguments.
        """
        # ----------------------------------------------------------------------------------
        # make cond & uncond for t2v
        uncond_prompt = "" if uncond_prompt is None else uncond_prompt
        batch_size = noise_shape[0]
        text_emb = self.get_learned_conditioning(prompts)
        fps = torch.tensor([fps] * batch_size).to(self.device).long()
        cond = {"c_crossattn": [text_emb], "fps": fps}

        if cfg_scale != 1.0:  # unconditional guidance
            uc_text_emb = self.get_learned_conditioning(batch_size * [uncond_prompt])
            uncond = {k: v for k, v in cond.items()}
            uncond.update({"c_crossattn": [uc_text_emb]})
        else:
            uncond = None

        # ----------------------------------------------------------------------------------
        # sampling
        batch_samples = []
        for _ in range(n_samples_prompt):  # iter over batch of prompts
            samples, _ = self.ddim_sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=batch_size,
                shape=noise_shape[1:],
                verbose=False,
                unconditional_guidance_scale=cfg_scale,
                unconditional_conditioning=uncond,
                eta=ddim_eta,
                temporal_length=noise_shape[2],
                conditional_guidance_scale_temporal=temporal_cfg_scale,
                **kwargs,
            )
            res = self.decode_first_stage(samples)
            batch_samples.append(res)
        batch_samples = torch.stack(batch_samples, dim=1)
        return batch_samples
    
    @torch.no_grad()
    def inference(self, args, **kwargs):
        # create inference sampler
        self.ddim_sampler = DDIMSampler(self)
        # load prompt list
        prompt_list = self.load_inference_inputs(args.prompt_file, mode=args.mode)

        # TODO: inference on multiple gpus

        # noise shape
        args.frames = self.temporal_length if args.frames is None else args.frames
        h, w, frames, channels = (
            args.height // 8,
            args.width // 8,
            args.frames,
            self.channels,
        )

        # -----------------------------------------------------------------
        # inference
        format_file = {}
        start = time.time()
        n_iters = len(prompt_list) // args.bs + (
            1 if len(prompt_list) % args.bs else 0
        )
        with torch.no_grad():
            for idx in trange(0, n_iters, desc="Sample Iters"):
                prompts = prompt_list[idx * args.bs : (idx + 1) * args.bs]
                filenames = self.process_savename(prompts, args.n_samples_prompt)
                ## inference
                bs = args.bs if args.bs == len(prompts) else len(prompts)
                noise_shape = [bs, channels, frames, h, w]
                if args.mode == "t2v":
                    batch_samples = self.sample_batch_t2v(
                        prompts,
                        args.fps,
                        noise_shape,
                        args.n_samples_prompt,
                        args.ddim_steps,
                        args.ddim_eta,
                        args.unconditional_guidance_scale,
                        args.unconditional_guidance_scale_temporal,
                        args.uncond_prompt,
                    )

                if args.standard_vbench:
                    self.save_videos_vbench(
                        batch_samples, args.savedir, prompts, format_file, fps=args.savefps
                    )
                else:
                    self.save_videos(batch_samples, args.savedir, filenames, fps=args.savefps)

        if args.standard_vbench:
            with open(os.path.join(args.savedir, "info.json"), "w") as f:
                json.dump(format_file, f)

        print_green(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")