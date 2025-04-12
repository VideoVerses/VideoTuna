"""
wild mixture of
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import logging
import random
from contextlib import contextmanager
from functools import partial

import numpy as np
import peft
import pytorch_lightning as pl
import torch
from einops import rearrange, repeat
from pytorch_lightning.utilities import rank_zero_only
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torchvision.utils import make_grid
from tqdm import tqdm

from videotuna.base.ddim import DDIMSampler
from videotuna.base.distributions import DiagonalGaussianDistribution
from videotuna.base.ema import LitEma

# import rlhf utils
from videotuna.lvdm.models.rlhf_utils.batch_ddim import batch_ddim_sampling
from videotuna.lvdm.models.rlhf_utils.reward_fn import aesthetic_loss_fn
from videotuna.lvdm.modules.encoders.ip_resampler import ImageProjModel, Resampler
from videotuna.lvdm.modules.utils import default, disabled_train, extract_into_tensor
from videotuna.utils.common_utils import instantiate_from_config

__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}


mainlogger = logging.getLogger("mainlogger")


class DDPMFlow(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
        self,
        unet_config,
        diffusion_scheduler_config,
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor=None,
        use_ema=True,
        first_stage_key="image",
        image_size=256,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        original_elbo_weight=0.0,
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",  # all assuming fixed variance schedules
        scheduler_config=None,  # learning rate scheduler config
        use_positional_encodings=False,
        lora_args=[],
        *args,
        **kwargs,
    ):
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
            "v",
        ], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        mainlogger.info(
            f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode"
        )

        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t

        # model related
        self.first_stage_key = first_stage_key
        self.cond_stage_model = None
        self.channels = channels
        self.temporal_length = unet_config.params.get("temporal_length", 16)
        self.image_size = image_size  # try conv?
        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        # load lora models
        # peft lora config. The arg name is algned with peft.
        self.lora_args = lora_args
        if len(lora_args) > 0:
            self.lora_ckpt_path = getattr(lora_args, "lora_ckpt", None)
            self.lora_rank = getattr(lora_args, "lora_rank", 4)
            self.lora_alpha = getattr(lora_args, "lora_alpha", 1)
            self.lora_dropout = getattr(lora_args, "lora_dropout", 0.0)
            self.target_modules = getattr(
                lora_args, "target_modules", ["to_k", "to_v", "to_q"]
            )
            # peft set other paramtere requires_grad to False
            self.lora_config = peft.LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=self.target_modules,  # only diffusion_model has these modules
                lora_dropout=self.lora_dropout,
            )

        # count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            mainlogger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # this is learning rate scheduler..
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        # TODO to be implemented
        print("scheduler config type: ", type(diffusion_scheduler_config))
        diffusion_scheduler_config.parameterization = self.parameterization
        self.diffusion_scheduler = instantiate_from_config(diffusion_scheduler_config)
        self.num_timesteps = self.diffusion_scheduler.num_timesteps

        # # this can be used to initiate schedulers
        # self.v_posterior = v_posterior
        # self.original_elbo_weight = original_elbo_weight
        # self.l_simple_weight = l_simple_weight

        # self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
        #                        linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        # self.learn_logvar = learn_logvar
        # self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        # if self.learn_logvar:
        #     self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        # others
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(
                ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet
            )

        self.loss_type = loss_type

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

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        # TODO logvar keys
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    mainlogger.info("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = (
            self.load_state_dict(sd, strict=False)
            if not only_model
            else self.model.load_state_dict(sd, strict=False)
        )
        mainlogger.info(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            mainlogger.info(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            mainlogger.info(f"Unexpected Keys: {unexpected}")

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.diffusion_scheduler.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="Sampling t",
            total=self.num_timesteps,
        ):
            img = self.diffusion_scheduler.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                clip_denoised=self.clip_denoised,
            )
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(
            (batch_size, channels, image_size, image_size),
            return_intermediates=return_intermediates,
        )

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

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.diffusion_scheduler.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "v":
            target = self.diffusion_scheduler.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(
                f"Paramterization {self.parameterization} not yet supported"
            )

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = "train" if self.training else "val"

        loss_dict.update({f"{log_prefix}/loss_simple": loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.diffusion_scheduler.lvlb_weights[t] * loss).mean()
        loss_dict.update({f"{log_prefix}/loss_vlb": loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f"{log_prefix}/loss": loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        """
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        """
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

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

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(
                    batch_size=N, return_intermediates=True
                )

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.diffusion_scheduler.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def load_lora_from_ckpt(self, model, path):
        lora_state_dict = torch.load(path)["state_dict"]
        copy_tracker = {key: False for key in lora_state_dict}
        for n, p in model.named_parameters():
            if "lora" in n:
                lora_n = f"model.{n}"
            else:
                continue
            if lora_n in lora_state_dict:
                if copy_tracker[lora_n]:
                    raise RuntimeError(
                        f"Parameter {lora_n} has already been copied once."
                    )
                print(f"Copying parameter {lora_n}")
                with torch.no_grad():
                    p.copy_(lora_state_dict[lora_n])
                copy_tracker[lora_n] = True
            else:
                raise RuntimeError(f"Parameter {lora_n} not found in lora_state_dict.")
        # check parameter load intergrity
        for key, copied in copy_tracker.items():
            if not copied:
                raise RuntimeError(
                    f"Parameter {key} from lora_state_dict was not copied to the model."
                )
                # print(f"Parameter {key} from lora_state_dict was not copied to the model.")
        print("All Parameters was copied successfully.")

    def inject_lora(self):
        """inject lora into the denoising module.
        The self.model should be a instance of pl.LightningModule or nn.Module.
        """
        # TODO: we can support inistantiate from config in the future. Now we test correctness.
        # for simplicity, we just inject denoising model. not injecting condition model.
        self.model = peft.get_peft_model(self.model, self.lora_config)

        self.model.print_trainable_parameters()

        if self.lora_ckpt_path is not None:
            self.load_lora_from_ckpt(self.model, self.lora_ckpt_path)


class LVDMFlow(DDPMFlow):
    """main class"""

    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        diffusion_scheduler_config,
        cond_stage_key="caption",
        cond_stage_trainable=False,
        cond_stage_forward=None,
        conditioning_key=None,
        uncond_prob=0.2,
        uncond_type="empty_seq",
        scale_factor=1.0,
        scale_by_std=False,
        fps_condition_type="fs",
        # added for LVDM
        encoder_type="2d",
        frame_cond=None,
        only_model=False,
        use_scale=False,  # dynamic rescaling
        scale_a=1,
        scale_b=0.3,
        mid_step=400,
        fix_scale_bug=False,
        interp_mode=False,
        logdir=None,
        rand_cond_frame=False,
        empty_params_only=False,
        *args,
        **kwargs,
    ):
        self.scale_by_std = scale_by_std
        # for backwards compatibility after implementation of DiffusionWrapper
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        conditioning_key = default(conditioning_key, "crossattn")

        # the init func of the base class will initiate a diffusion_scheduler
        super().__init__(
            conditioning_key=conditioning_key,
            diffusion_scheduler_config=diffusion_scheduler_config,
            *args,
            **kwargs,
        )

        # self.diffusion_scheduler = instantiate_from_config(diffusion_scheduler_config)

        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.empty_params_only = empty_params_only
        self.fps_condition_type = fps_condition_type

        # scale factor
        self.use_scale = use_scale
        if self.use_scale:
            self.scale_a = scale_a
            self.scale_b = scale_b
            if fix_scale_bug:
                scale_step = self.num_timesteps - mid_step
            else:  # bug
                scale_step = self.num_timesteps

            scale_arr1 = np.linspace(scale_a, scale_b, mid_step)
            scale_arr2 = np.full(scale_step, scale_b)
            scale_arr = np.concatenate((scale_arr1, scale_arr2))
            scale_arr_prev = np.append(scale_a, scale_arr[:-1])
            to_torch = partial(torch.tensor, dtype=torch.float32)
            self.register_buffer("scale_arr", to_torch(scale_arr))

        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except Exception:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.first_stage_config = first_stage_config
        self.cond_stage_config = cond_stage_config
        self.clip_denoised = False

        self.cond_stage_forward = cond_stage_forward
        self.encoder_type = encoder_type
        assert encoder_type in ["2d", "3d"]
        self.uncond_prob = uncond_prob
        self.classifier_free_guidance = True if uncond_prob > 0 else False
        assert uncond_type in ["zero_embed", "empty_seq"]
        self.uncond_type = uncond_type

        ## future frame prediction
        self.frame_cond = frame_cond
        if self.frame_cond:
            # frame_len = self.model.diffusion_model.temporal_length
            frame_len = self.temporal_length
            cond_mask = torch.zeros(frame_len, dtype=torch.float32)
            cond_mask[: self.frame_cond] = 1.0
            ## b,c,t,h,w
            self.cond_mask = cond_mask[None, None, :, None, None]
            mainlogger.info(
                "---training for %d-frame conditoning T2V" % (self.frame_cond)
            )
        else:
            self.cond_mask = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model=only_model)
            self.restarted_from_ckpt = True

        self.logdir = logdir
        self.rand_cond_frame = rand_cond_frame
        self.interp_mode = interp_mode

    def _freeze_model(self):
        for name, para in self.model.diffusion_model.named_parameters():
            para.requires_grad = False

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=None):
        # only for very first batch, reset the self.scale_factor
        if (
            self.scale_by_std
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
            and not self.restarted_from_ckpt
        ):
            assert (
                self.scale_factor == 1.0
            ), "rather not use custom rescaling and std-rescaling simultaneously"
            # set rescale weight to 1./std of encodings
            mainlogger.info("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer("scale_factor", 1.0 / z.flatten().std())
            mainlogger.info(f"setting self.scale_factor to {self.scale_factor}")
            mainlogger.info("### USING STD-RESCALING ###")
            mainlogger.info(f"std={z.flatten().std()}")

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model = model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(
                self.cond_stage_model.encode
            ):
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
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
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
        results = torch.cat(
            [
                self.get_first_stage_encoding(self.first_stage_model.encode(x[:, :, i]))
                .detach()
                .unsqueeze(2)
                for i in range(t)
            ],
            dim=2,
        )
        return results

    def decode_first_stage_2DAE(self, z, **kwargs):
        """decode frame by frame"""
        _, _, t, _, _ = z.shape
        results = torch.cat(
            [
                self.first_stage_model.decode(z[:, :, i], **kwargs).unsqueeze(2)
                for i in range(t)
            ],
            dim=2,
        )
        return results

    def _decode_core(self, z, **kwargs):
        z = 1.0 / self.scale_factor * z

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

    def get_batch_input(
        self,
        batch,
        random_uncond,
        return_first_stage_outputs=False,
        return_original_cond=False,
        is_imgbatch=False,
    ):
        ## image/video shape: b, c, t, h, w
        data_key = "jpg" if is_imgbatch else self.first_stage_key
        x = super().get_input(batch, data_key)
        if is_imgbatch:
            ## pack image as video
            # x = x[:,:,None,:,:]
            b = x.shape[0] // self.temporal_length
            x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=self.temporal_length)
        x_ori = x
        ## encode video frames x to z via a 2D encoder
        z = self.encode_first_stage(x)

        ## get caption condition
        cond_key = "txt" if is_imgbatch else self.cond_stage_key
        cond = batch[cond_key]
        if random_uncond and self.uncond_type == "empty_seq":
            for i, ci in enumerate(cond):
                if random.random() < self.uncond_prob:
                    cond[i] = ""
        if isinstance(cond, dict) or isinstance(cond, list):
            cond_emb = self.get_learned_conditioning(cond)
        else:
            cond_emb = self.get_learned_conditioning(cond.to(self.device))
        if random_uncond and self.uncond_type == "zero_embed":
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
        if "t" in kwargs:
            t = kwargs.pop("t")
        else:
            t = torch.randint(
                0, self.num_timesteps, (x.shape[0],), device=self.device
            ).long()
        if self.use_scale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, **kwargs)

    def shared_step(self, batch, random_uncond, **kwargs):
        is_imgbatch = False
        if "loader_img" in batch.keys():
            ratio = 10.0 / self.temporal_length
            if random.uniform(0.0, 10.0) < ratio:
                is_imgbatch = True
                batch = batch["loader_img"]
            else:
                batch = batch["loader_video"]
        else:
            pass

        x, c = self.get_batch_input(
            batch, random_uncond=random_uncond, is_imgbatch=is_imgbatch
        )
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

    def p_losses(self, x_start, cond, t, noise=None, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.diffusion_scheduler.q_sample(x_start=x_start, t=t, noise=noise)
        if self.frame_cond:
            if self.cond_mask.device is not self.device:
                self.cond_mask = self.cond_mask.to(self.device)
            ## condition on fist few frames
            x_noisy = x_start * self.cond_mask + (1.0 - self.cond_mask) * x_noisy
        model_output = self.apply_model(x_noisy, t, cond, **kwargs)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.diffusion_scheduler.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        if self.frame_cond:
            ## [b,c,t,h,w]: only care about the predicted part (avoid disturbance)
            model_output = model_output[:, :, self.frame_cond :, :, :]
            target = target[:, :, self.frame_cond :, :, :]

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])

        if torch.isnan(loss_simple).any():
            print(f"loss_simple exists nan: {loss_simple}")
            # import pdb; pdb.set_trace()
            for i in range(loss_simple.shape[0]):
                if torch.isnan(loss_simple[i]).any():
                    loss_simple[i] = torch.zeros_like(loss_simple[i])

        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        if self.diffusion_scheduler.logvar.device is not self.device:
            self.diffusion_scheduler.logvar = self.diffusion_scheduler.logvar.to(
                self.device
            )
        logvar_t = self.diffusion_scheduler.logvar[t]
        # logvar_t = self.logvar[t.item()].to(self.device) # device conflict when ddp shared
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.diffusion_scheduler.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.diffusion_scheduler.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        if self.original_elbo_weight > 0:
            loss_vlb = self.get_loss(model_output, target, mean=False).mean(
                dim=(1, 2, 3, 4)
            )
            loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
            loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
            loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(
            batch, random_uncond=self.classifier_free_guidance
        )
        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )
        # self.log("epoch/global_step", self.global_step.float(), prog_bar=True, logger=True, on_step=True, on_epoch=False)
        """
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        """
        if (batch_idx + 1) % self.log_every_t == 0:
            mainlogger.info(
                f"batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}"
            )
        return loss

    def _get_denoise_row_from_list(self, samples, desc=""):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_log_timesteps = len(denoise_row)

        denoise_row = torch.stack(denoise_row)  # n_log_timesteps, b, C, H, W

        if denoise_row.dim() == 5:
            # img, num_imgs= n_log_timesteps * bs, grid_size=[bs,n_log_timesteps]
            # batch:col, different samples,
            # n:rows, different steps for one sample
            denoise_grid = rearrange(denoise_row, "n b c h w -> b n c h w")
            denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
            denoise_grid = make_grid(denoise_grid, nrow=n_log_timesteps)
        elif denoise_row.dim() == 6:
            # video, grid_size=[n_log_timesteps*bs, t]
            video_length = denoise_row.shape[3]
            denoise_grid = rearrange(denoise_row, "n b c t h w -> b n c t h w")
            denoise_grid = rearrange(denoise_grid, "b n c t h w -> (b n) c t h w")
            denoise_grid = rearrange(denoise_grid, "n c t h w -> (n t) c h w")
            denoise_grid = make_grid(denoise_grid, nrow=video_length)
        else:
            raise ValueError

        return denoise_grid

    @torch.no_grad()
    def log_images(
        self,
        batch,
        sample=True,
        ddim_steps=200,
        ddim_eta=1.0,
        plot_denoise_rows=False,
        unconditional_guidance_scale=1.0,
        **kwargs,
    ):
        """log images for LatentDiffusion"""
        ## TBD: currently, classifier_free_guidance sampling is only supported by DDIM
        use_ddim = ddim_steps is not None
        log = dict()
        z, c, x, xrec, xc = self.get_batch_input(
            batch,
            random_uncond=False,
            return_first_stage_outputs=True,
            return_original_cond=True,
        )
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
                        c_cat = None  # set default value is None
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
                samples, z_denoise_row = self.sample_log(
                    cond=c,
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=uc,
                    mask=self.cond_mask,
                    x0=z,
                    **kwargs,
                )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log

    @torch.no_grad()
    def p_sample_loop(
        self,
        cond,
        shape,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        callback=None,
        timesteps=None,
        mask=None,
        x0=None,
        img_callback=None,
        start_T=None,
        log_every_t=None,
        **kwargs,
    ):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.diffusion_scheduler.betas.device
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

        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Sampling t", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.diffusion_scheduler.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.diffusion_scheduler.q_sample(
                    x_start=cond, t=tc, noise=torch.randn_like(cond)
                )

            img = self.diffusion_scheduler.p_sample(
                img, cond, ts, clip_denoised=self.clip_denoised, **kwargs
            )
            if mask is not None:
                img_orig = self.diffusion_scheduler.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(
        self,
        cond,
        batch_size=16,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        timesteps=None,
        mask=None,
        x0=None,
        shape=None,
        decode=True,
        **kwargs,
    ):
        if shape is None:
            shape = (batch_size, self.channels, self.temporal_length, *self.image_size)

        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: (
                        cond[key][:batch_size]
                        if not isinstance(cond[key], list)
                        else list(map(lambda x: x[:batch_size], cond[key]))
                    )
                    for key in cond
                }
            else:
                cond = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )

        samples = self.p_sample_loop(
            cond,
            shape,
            return_intermediates=return_intermediates,
            x_T=x_T,
            verbose=verbose,
            timesteps=timesteps,
            mask=mask,
            x0=x0,
            **kwargs,
        )
        if decode:
            samples = self.decode_first_stage(samples)
        return samples

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.temporal_length, *self.image_size)
            # kwargs.update({"clean_cond": True})
            samples, intermediates = ddim_sampler.sample(
                ddim_steps, batch_size, shape, cond, verbose=False, **kwargs
            )

        else:
            samples, intermediates = self.sample(
                cond=cond, batch_size=batch_size, return_intermediates=True, **kwargs
            )

        return samples, intermediates

    def configure_optimizers(self):
        """configure_optimizers for LatentDiffusion"""
        lr = self.learning_rate
        if self.empty_params_only and hasattr(self, "empty_paras"):
            params = [
                p for n, p in self.model.named_parameters() if n in self.empty_paras
            ]
            print("self.empty_paras", len(self.empty_paras))
            for n, p in self.model.named_parameters():
                if n not in self.empty_paras:
                    p.requires_grad = False
            mainlogger.info(f"@Training [{len(params)}] Empty Paramters ONLY.")
        elif len(self.lora_args) > 0:
            # if there is lora_args, but haven't injected lora, it would also work.
            # but the trainable params will be significantly more than the lora_params
            # filter out the non lora parameters.
            params = [
                p
                for n, p in self.model.named_parameters()
                if p.requires_grad and "lora" in n
            ]
            mainlogger.info(f"@Training [{len(params)}] Lora Paramters.")
        else:
            params = list(self.model.parameters())
            mainlogger.info(f"@Training [{len(params)}] Full Paramters.")

        if self.diffusion_scheduler.learn_logvar:
            mainlogger.info("Diffusion model optimizing logvar")
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)

        ## optimizer
        optimizer = torch.optim.AdamW(params, lr=lr)
        ## lr scheduler
        if self.use_scheduler:
            mainlogger.info("Setting up LambdaLR scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]

        return optimizer

    def configure_schedulers(self, optimizer):
        assert "target" in self.scheduler_config
        scheduler_name = self.scheduler_config.target.split(".")[-1]
        interval = self.scheduler_config.interval
        frequency = self.scheduler_config.frequency
        if scheduler_name == "LambdaLRScheduler":
            scheduler = instantiate_from_config(self.scheduler_config)
            scheduler.start_step = self.global_step
            lr_scheduler = {
                "scheduler": LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                "interval": interval,
                "frequency": frequency,
            }
        elif scheduler_name == "CosineAnnealingLRScheduler":
            scheduler = instantiate_from_config(self.scheduler_config)
            decay_steps = scheduler.decay_steps
            last_step = -1 if self.global_step == 0 else scheduler.start_step
            lr_scheduler = {
                "scheduler": CosineAnnealingLR(
                    optimizer, T_max=decay_steps, last_epoch=last_step
                ),
                "interval": interval,
                "frequency": frequency,
            }
        else:
            raise NotImplementedError
        return lr_scheduler


class RewardLVDMTrainer(LVDMFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Reward Gradient Training Using LoRA as a default
        # TODO: use config and getattr to set default values
        # sampling configs for DDIM
        self.ddim_eta = 1.0
        self.ddim_steps = 20  # reduce some steps to speed up sampling process
        self.n_samples = 1
        self.fps = 24  # default 24 following VADER
        # rlhf configs
        self.backprop_mode = (
            "last"  # m"backpropagation mode: 'last', 'rand', 'specific'"
        )
        self.decode_frame = (
            -1
        )  # it could also be any number str like '3', '10'. alt: alternate frames, fml: first, middle, last frames, all: all frames. '-1': random frame
        self.reward_loss_type = "aesthetic"
        # self.configure_reward_loss()

    def configure_reward_loss(self, loss_type=None):
        if loss_type is None:
            loss_type = self.reward_loss_type

        if loss_type == "aesthetic":
            self.loss_fn = aesthetic_loss_fn(
                grad_scale=0.1,
                aesthetic_target=10,
                torch_dtype=self.model.dtype,
                device=self.device,
            )
        else:
            raise NotImplementedError(f"loss type {loss_type} not implemented")

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=None):
        # the reason why configure here is to wait the model transfered to target device
        # otherwise, the loss_fn will be on cpu if configure in __init__
        self.configure_reward_loss()

    def training_step(self, batch, batch_idx):
        """training_step for Reward Model Feedback"""
        # in reward model training, we just need shape of video frames
        # default cond is text prompt
        prompts = batch[self.cond_stage_key]
        # print(prompts) #  Elon mask is talking
        x, c = self.get_batch_input(
            batch, random_uncond=self.classifier_free_guidance, is_imgbatch=False
        )  # x is latent image ; c is text embedding(tensor)
        kwargs = {}
        batch_size = x.shape[0]
        noise_shape = (
            batch_size,
            self.channels,
            self.temporal_length // 4,
            *self.image_size,
        )  # (1, 4, 4, 40, 64)
        # print("noise shape",noise_shape)
        fps = torch.tensor([self.fps] * batch_size).to(self.device).long()
        cond = {"c_crossattn": [c], "fps": fps}
        # Notice: VADER has modified ddim for training
        # input cond = {"c_crossattn": [text_emb], "fps": fps}
        batch_samples = batch_ddim_sampling(
            self,
            cond,
            noise_shape,
            self.n_samples,
            self.ddim_steps,
            self.ddim_eta,
            self.classifier_free_guidance,
            None,
            backprop_mode=self.backprop_mode,
            decode_frame=self.decode_frame,
            **kwargs,
        )

        video_frames_ = batch_samples.permute(
            1, 0, 3, 2, 4, 5
        )  # batch,samples,channels,frames,height,width >> s,b,f,c,h,w
        # print("video_frames shape",batch_samples.shape,video_frames_.requires_grad)
        s_, bs, nf, c_, h_, w_ = video_frames_.shape
        assert s_ == 1  # samples should only be on single sample in training mode
        video_frames_ = video_frames_.squeeze(0)  # s,b,f,c,h,w >> b,f,c,h,w
        assert nf == 1  # reward should only be on single frame
        video_frames_ = video_frames_.squeeze(1)  # b,f,c,h,w >> b,c,h,w
        video_frames_ = video_frames_.to(x.dtype)

        # some reward fn may require prompts as input.
        loss, rewards = self.loss_fn(video_frames_)  # rewards is for logging only.
        loss_dict = {
            "reward_train_loss": loss.detach().item(),
            "step_reward": rewards.detach().item(),
        }
        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        if (batch_idx + 1) % self.log_every_t == 0:
            mainlogger.info(
                f"batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss} reward={rewards}"
            )
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        # the training steps count is short no need for scheduler as default
        if self.use_scheduler:
            lr_scheduler = self.configure_schedulers(opt)
            return [opt], [lr_scheduler]
        return opt


class LatentVisualDiffusionFlow(LVDMFlow):
    def __init__(
        self,
        img_cond_stage_config,
        finegrained=False,  # vc1-i2v
        image_proj_stage_config=None,  # dc
        freeze_embedder=True,  # dc
        image_proj_model_trainable=True,  # dc
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.image_proj_model_trainable = image_proj_model_trainable
        self._init_embedder(img_cond_stage_config, freeze_embedder)
        if image_proj_stage_config is not None:
            self._init_img_ctx_projector(
                image_proj_stage_config, image_proj_model_trainable
            )
        else:
            self.init_projector(
                use_finegrained=finegrained,
                num_tokens=16 if finegrained else 4,
                input_dim=1024,
                cross_attention_dim=1024,
                dim=1280,
            )

    def _init_img_ctx_projector(self, config, trainable):
        self.image_proj_model = instantiate_from_config(config)
        if not trainable:
            self.image_proj_model.eval()
            self.image_proj_model.train = disabled_train
            for param in self.image_proj_model.parameters():
                param.requires_grad = False

    def _init_embedder(self, config, freeze=True):
        embedder = instantiate_from_config(config)
        if freeze:
            self.embedder = embedder.eval()
            self.embedder.train = disabled_train
            for param in self.embedder.parameters():
                param.requires_grad = False

    def init_projector(
        self, use_finegrained, num_tokens, input_dim, cross_attention_dim, dim
    ):
        if not use_finegrained:
            image_proj_model = ImageProjModel(
                clip_extra_context_tokens=num_tokens,
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=input_dim,
            )
        else:
            image_proj_model = Resampler(
                dim=input_dim,
                depth=4,
                dim_head=64,
                heads=12,
                num_queries=num_tokens,
                embedding_dim=dim,
                output_dim=cross_attention_dim,
                ff_mult=4,
            )
        self.image_proj_model = image_proj_model

    ## Never delete this func: it is used in log_images() and inference stage
    def get_image_embeds(self, batch_imgs):
        ## img: b c h w
        img_token = self.embedder(batch_imgs)
        img_emb = self.image_proj_model(img_token)
        return img_emb

    def shared_step(self, batch, random_uncond, **kwargs):
        x, c, fs = self.get_batch_input(
            batch, random_uncond=random_uncond, return_fs=True
        )
        kwargs.update({"fs": fs.long()})
        loss, loss_dict = self(x, c, **kwargs)
        return loss, loss_dict

    def get_batch_input(
        self,
        batch,
        random_uncond,
        return_first_stage_outputs=False,
        return_original_cond=False,
        return_fs=False,
        return_cond_frame=False,
        return_original_input=False,
        **kwargs,
    ):
        ## x: b c t h w
        x = super().get_input(batch, self.first_stage_key)
        ## encode video frames x to z via a 2D encoder
        z = self.encode_first_stage(x)

        ## get caption condition
        cond_input = batch[self.cond_stage_key]

        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))

        cond = {}
        ## to support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(
                x.size(0), device=x.device
            )  ## by doning so, we can get text embedding and complete img emb for inference
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, "n -> n 1 1")
        input_mask = 1 - rearrange(
            (random_num >= self.uncond_prob).float()
            * (random_num < 3 * self.uncond_prob).float(),
            "n -> n 1 1 1",
        )

        null_prompt = self.get_learned_conditioning([""])
        prompt_imb = torch.where(prompt_mask, null_prompt, cond_emb.detach())

        ## get conditioning frame
        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(
                0, self.model.diffusion_model.temporal_length - 1
            )

        img = x[:, :, cond_frame_index, ...]
        img = input_mask * img
        ## img: b c h w
        img_emb = self.embedder(img)  ## b l c
        img_emb = self.image_proj_model(img_emb)

        if self.model.conditioning_key == "hybrid":
            if self.interp_mode:
                ## starting frame + (L-2 empty frames) + ending frame
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
                img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
            else:
                ## simply repeat the cond_frame to match the seq_len of z
                img_cat_cond = z[:, :, cond_frame_index, :, :]
                img_cat_cond = img_cat_cond.unsqueeze(2)
                img_cat_cond = repeat(
                    img_cat_cond, "b c t h w -> b c (repeat t) h w", repeat=z.shape[2]
                )
            cond["c_concat"] = [img_cat_cond]  # b c t h w
        cond["c_crossattn"] = [
            torch.cat([prompt_imb, img_emb], dim=1)
        ]  ## concat in the seq_len dim

        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])

        if return_original_cond:
            out.append(cond_input)
        if return_fs:
            if self.fps_condition_type == "fs":
                fs = super().get_input(batch, "frame_stride")
            elif self.fps_condition_type == "fps":
                fs = super().get_input(batch, "fps")
            out.append(fs)
        if return_cond_frame:
            out.append(x[:, :, cond_frame_index, ...].unsqueeze(2))
        if return_original_input:
            out.append(x)

        return out

    @torch.no_grad()
    def log_images(
        self,
        batch,
        sample=True,
        ddim_steps=50,
        ddim_eta=1.0,
        plot_denoise_rows=False,
        unconditional_guidance_scale=1.0,
        mask=None,
        **kwargs,
    ):
        """log images for LatentVisualDiffusion"""
        ##### sampled_img_num: control sampled imgae for logging, larger value may cause OOM
        sampled_img_num = 1
        for key in batch.keys():
            batch[key] = batch[key][:sampled_img_num]

        ## TBD: currently, classifier_free_guidance sampling is only supported by DDIM
        use_ddim = ddim_steps is not None
        log = dict()

        z, c, xrec, xc, fs, cond_x = self.get_batch_input(
            batch,
            random_uncond=False,
            return_first_stage_outputs=True,
            return_original_cond=True,
            return_fs=True,
            return_cond_frame=True,
        )

        N = xrec.shape[0]
        log["image_condition"] = cond_x
        log["reconst"] = xrec
        xc_with_fs = []
        for idx, content in enumerate(xc):
            xc_with_fs.append(content + "_fs=" + str(fs[idx].item()))
        log["condition"] = xc_with_fs
        kwargs.update({"fs": fs.long()})

        c_cat = None
        if sample:
            # get uncond embedding for classifier-free guidance sampling
            if unconditional_guidance_scale != 1.0:
                if isinstance(c, dict):
                    c_emb = c["c_crossattn"][0]
                    if "c_concat" in c.keys():
                        c_cat = c["c_concat"][0]
                else:
                    c_emb = c

                if self.uncond_type == "empty_seq":
                    prompts = N * [""]
                    uc_prompt = self.get_learned_conditioning(prompts)
                elif self.uncond_type == "zero_embed":
                    uc_prompt = torch.zeros_like(c_emb)

                img = torch.zeros_like(xrec[:, :, 0])  ## b c h w
                ## img: b c h w
                img_emb = self.embedder(img)  ## b l c
                uc_img = self.image_proj_model(img_emb)

                uc = torch.cat([uc_prompt, uc_img], dim=1)
                ## hybrid case
                if isinstance(c, dict):
                    uc_hybrid = {"c_concat": [c_cat], "c_crossattn": [uc]}
                    uc = uc_hybrid
            else:
                uc = None

            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(
                    cond=c,
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=uc,
                    x0=z,
                    **kwargs,
                )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log

    def configure_optimizers(self):
        """configure_optimizers for LatentDiffusion"""
        lr = self.learning_rate

        params = list(self.model.parameters())
        mainlogger.info(f"@Training [{len(params)}] Full Paramters.")

        if self.cond_stage_trainable:
            params_cond_stage = [
                p for p in self.cond_stage_model.parameters() if p.requires_grad is True
            ]
            mainlogger.info(
                f"@Training [{len(params_cond_stage)}] Paramters for Cond_stage_model."
            )
            params.extend(params_cond_stage)

        if self.image_proj_model_trainable:
            mainlogger.info(
                f"@Training [{len(list(self.image_proj_model.parameters()))}] Paramters for Image_proj_model."
            )
            params.extend(list(self.image_proj_model.parameters()))

        if self.diffusion_scheduler.learn_logvar:
            mainlogger.info("Diffusion model optimizing logvar")
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)

        ## optimizer
        optimizer = torch.optim.AdamW(params, lr=lr)

        ## lr scheduler
        if self.use_scheduler:
            mainlogger.info("Setting up scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]

        return optimizer


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        # print('diff model config: ', diff_model_config)
        # self.precision = diff_model_config.pop('precision', None)
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key

    def forward(
        self,
        x,
        t,
        c_concat: list = None,
        c_crossattn: list = None,
        c_crossattn_stdit: list = None,
        mask: list = None,
        c_adm=None,
        s=None,
        **kwargs,
    ):
        # temporal_context = fps is foNone
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == "concat":
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, **kwargs)
        elif self.conditioning_key == "crossattn":
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, **kwargs)
        elif self.conditioning_key == "crossattn_stdit":
            cc = torch.cat(c_crossattn_stdit, 1)  # [b, 77, 1024]
            mask = torch.cat(mask, 1)
            # TODO fix precision
            # if self.precision is not None and self.precision == 'bf16':
            # print('Convert datatype')
            cc = cc.to(torch.bfloat16)
            self.diffusion_model = self.diffusion_model.to(torch.bfloat16)

            out = self.diffusion_model(x, t, y=cc, mask=mask)
            # def forward(self, x, timestep, y, mask=None, x_mask=None, fps=None, height=None, width=None, **kwargs):
        elif self.conditioning_key == "hybrid":
            ## it is just right [b,c,t,h,w]: concatenate in channel dim
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, **kwargs)
        elif self.conditioning_key == "resblockcond":
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == "adm":
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        elif self.conditioning_key == "hybrid-adm":
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, y=c_adm, **kwargs)
        elif self.conditioning_key == "hybrid-time":
            assert s is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, s=s)
        elif self.conditioning_key == "concat-time-mask":
            # assert s is not None
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, context=None, s=s, mask=mask)
        elif self.conditioning_key == "concat-adm-mask":
            # assert s is not None
            if c_concat is not None:
                xc = torch.cat([x] + c_concat, dim=1)
            else:
                xc = x
            out = self.diffusion_model(xc, t, context=None, y=s, mask=mask)
        elif self.conditioning_key == "hybrid-adm-mask":
            cc = torch.cat(c_crossattn, 1)
            if c_concat is not None:
                xc = torch.cat([x] + c_concat, dim=1)
            else:
                xc = x
            out = self.diffusion_model(xc, t, context=cc, y=s, mask=mask)
        elif (
            self.conditioning_key == "hybrid-time-adm"
        ):  # adm means y, e.g., class index
            # assert s is not None
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, s=s, y=c_adm)
        elif self.conditioning_key == "crossattn-adm":
            assert c_adm is not None
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, y=c_adm)
        else:
            raise NotImplementedError()

        return out
