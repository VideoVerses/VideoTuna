import enum
import logging
import math
import os
import random
from contextlib import contextmanager
from functools import partial

import numpy as np
from einops import rearrange, repeat
from omegaconf.listconfig import ListConfig
from tqdm import tqdm

mainlogger = logging.getLogger("mainlogger")

import torch
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_only
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torchvision.utils import make_grid

from videotuna.base.ddim import DDIMSampler
from videotuna.base.ddpm3d import DDPMFlow
from videotuna.base.diffusion_schedulers import DDPMScheduler
from videotuna.base.distributions import DiagonalGaussianDistribution, normal_kl
from videotuna.base.utils_diffusion import (
    discretized_gaussian_log_likelihood,
    make_beta_schedule,
    rescale_zero_terminal_snr,
)
from videotuna.lvdm.modules.utils import (
    default,
    disabled_train,
    exists,
    extract_into_tensor,
    noise_like,
)
from videotuna.utils.common_utils import instantiate_from_config


def mean_flat(tensor: torch.Tensor, mask=None) -> torch.Tensor:
    """
    Take the mean over all non-batch dimensions.
    """
    if mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    else:
        assert tensor.dim() == 5
        assert tensor.shape[2] == mask.shape[1]
        tensor = rearrange(tensor, "b c t h w -> b t (c h w)")
        denom = mask.sum(dim=1) * tensor.shape[-1]
        loss = (tensor * mask.unsqueeze(2)).sum(dim=1).sum(dim=1) / denom
        return loss


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(
        beta_start, beta_end, warmup_time, dtype=np.float64
    )
    return betas


def get_beta_schedule(
    beta_schedule: str, *, beta_start, beta_end, num_diffusion_timesteps: int
) -> np.ndarray:
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(
    schedule_name: str, num_diffusion_timesteps: int
) -> np.ndarray:
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(
    num_diffusion_timesteps, alpha_bar, max_beta=0.999
) -> np.ndarray:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class IDDPMScheduler(DDPMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def register_schedule(self, given_betas, *args, **kwargs):
        if exists(given_betas):
            if isinstance(given_betas, list) or isinstance(given_betas, ListConfig):
                betas = np.array(given_betas, dtype=np.float32)
            elif isinstance(given_betas, np.ndarray):
                betas = given_betas
            else:
                raise TypeError("given_betas type error")
        else:
            raise ValueError("given_betas must be provided")

        betas = np.array(betas, dtype=np.float32)

        timesteps = betas.shape[0]
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.0)
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, device=self.device)

        self.register_buffer("betas", to_torch(betas, dtype=torch.float32))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
        self.register_buffer("alphas_cumprod_next", to_torch(alphas_cumprod_next))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer(
            "posterior_log_variance_clipped",
            (
                to_torch(
                    np.log(np.append(posterior_variance[1], posterior_variance[1:]))
                )
                if len(self.posterior_variance) > 1
                else torch.DoubleTensor([])
            ),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    @torch.no_grad()
    def p_sample(
        self,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        mask=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        if mask is not None:
            if mask.shape[0] != x.shape[0]:
                mask = mask.repeat(2, 1)  # HACK
            mask_t = (mask * len(self.betas)).to(torch.int)

            # x0: copy unchanged x values
            # x_noise: add noise to x values
            x0 = x.clone()
            x_noise = x0 * extract_into_tensor(
                self.sqrt_alphas_cumprod, t, x.shape
            ) + torch.randn_like(x) * extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x.shape
            )

            # active noise addition
            mask_t_equall = (mask_t == t.unsqueeze(1))[:, None, :, None, None]
            x = torch.where(mask_t_equall, x_noise, x0)

            # create x_mask
            mask_t_upper = (mask_t > t.unsqueeze(1))[:, None, :, None, None]
            batch_size = x.shape[0]
            model_kwargs["x_mask"] = mask_t_upper.reshape(batch_size, -1).to(torch.bool)

        model_mean, _, model_log_variance = self.p_mean_variance(
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        # TODO: check cond_fn
        # if cond_fn is not None:
        #     model_mean = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

        if mask is not None:
            mask_t_lower = (mask_t < t.unsqueeze(1))[:, None, :, None, None]
            sample = torch.where(mask_t_lower, x0, sample)

        return sample

    def predict_start_from_noise(self, x_t, t, noise):
        assert noise.shape == x_t.shape
        return super().predict_start_from_noise(x_t, t, noise)

    def predict_start_from_prev(self, x_t, t, x_prev):
        assert x_prev.shape == x_t.shape
        return (  # (x_prev - coef2 * x_t) / coef1
            extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * x_prev
            - extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def p_mean_variance(
        self, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = self.model(x, t, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            min_log = extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = extract_into_tensor(torch.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    torch.cat(self.posterior_variance[1].unsqueeze(0), self.betas[1:]),
                    torch.log(
                        torch.cat(
                            self.posterior_variance[1].unsqueeze(0), self.betas[1:]
                        )
                    ),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            model_mean = model_output
            pred_xstart = process_xstart(
                self.predict_start_from_prev(x_t=x, t=t, x_prev=model_output)
            )
        elif self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
            model_mean, _, _ = self.q_posterior(x_start=pred_xstart, x_t=x, t=t)
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(
                self.predict_start_from_noise(x_t=x, t=t, eps=model_output)
            )
            model_mean, _, _ = self.q_posterior(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return model_mean, model_variance, model_log_variance


class OpenSoraScheduler(IDDPMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def p_mean_variance(
        self,
        x,
        c,
        t,
        clip_denoised: bool,
        return_x0=False,
        score_corrector=None,
        corrector_kwargs=None,
        model_kwargs=None,
        **kwargs,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        t_in = t
        model_output = self.apply_model(x, t_in, c, **kwargs)
        B, C = x.shape[:2]
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            min_log = extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = extract_into_tensor(torch.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    torch.cat(self.posterior_variance[1].unsqueeze(0), self.betas[1:]),
                    torch.log(
                        torch.cat(
                            self.posterior_variance[1].unsqueeze(0), self.betas[1:]
                        )
                    ),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = extract_into_tensor(model_log_variance, t, x.shape)

        if self.model_mean_type == ModelMeanType.START_X:
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_output)
        else:
            x_recon = model_output
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_output = score_corrector.modify_score(
                self, model_output, x, t, c, **corrector_kwargs
            )

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        # posterior_variance = extract_into_tensor(posterior_variance, t, x.shape)
        # posterior_log_variance = extract_into_tensor(posterior_log_variance, t, x.shape)
        if return_x0:
            return model_mean, model_log_variance, model_log_variance, x_recon
        else:
            return model_mean, model_log_variance, model_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x,
        c,
        t,
        clip_denoised=False,
        repeat_noise=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        **kwargs,
    ):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(
            model=self.model,
            x=x,
            c=c,
            t=t,
            clip_denoised=clip_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            **kwargs,
        )
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return (
                model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,
                x0,
            )
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def apply_model(self, x_noisy, t, cond, **kwargs):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = (
                "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            )
            cond = {key: cond}

        # If model is provided as a keyword argument, use it to train the variance.
        if "model" not in kwargs:
            x_recon = self.model(x_noisy, t, **cond, **kwargs)
        else:
            x_recon = kwargs["model"](x_noisy, t)

        if isinstance(x_recon, tuple):
            return x_recon[0]
        else:
            return x_recon


class IDDPM(DDPMFlow):
    def __init__(
        self,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.LEARNED_RANGE,
        loss_type=LossType.MSE,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        # Add the model mean and variance types to the scheduler.
        self.diffusion_scheduler.model_mean_type = model_mean_type
        self.diffusion_scheduler.model_var_type = model_var_type

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        return_intermediates=False,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        mask=None,
    ):
        """
        Generate samples from the model.
        :param shape: the shape of the samples, (N, C, H, W).
        :param return_intermediates: if True, return all intermediate samples.
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        if device is None:
            device = next(self.model.parameters()).device
        assert isinstance(shape, (tuple, list))

        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.diffusion_scheduler.num_timesteps))[::-1]
        b = shape[0]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        intermediates = [img]
        for i in indices:
            t = torch.full((b,), i, device=device)
            out = self.diffusion_scheduler.p_sample(
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                mask=mask,
            )
            if (
                i % self.log_every_t == 0
                or i == self.diffusion_scheduler.num_timesteps - 1
            ):
                intermediates.append(img)

        if return_intermediates:
            return img, intermediates
        else:
            return img

    def _vb_terms_bpd(
        self, x_start, x_t, t, clip_denoised=True, model_kwargs=None, mask=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.diffusion_scheduler.q_posterior(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.diffusion_scheduler.p_mean_variance(
            x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl, mask=mask) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll, mask=mask) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return output

    def p_losses(
        self,
        x_start,
        t,
        noise=None,
        model_kwargs=None,
        mask=None,
        weights=None,
    ):
        """
        Compute the losses for the model.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: the timestep tensor.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param mask: if not None, a mask tensor to apply to the losses.
        :return: a dict containing the following keys:
                 - 'loss': the loss value.
                 - 'denoised': the denoised signal.
        """
        if model_kwargs is None:
            model_kwargs = {}
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.diffusion_scheduler.q_sample(x_start, t, noise=noise)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.diffusion_scheduler.q_sample(x_start, t0, noise=noise)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        terms = {}
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            assert mask is None, "mask not supported for KL loss"
            terms["loss"] = self._vb_terms_bpd(
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.diffusion_scheduler.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = self.model(x_t, t, **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                # TODO: refactor this protect mean prediction
                # frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                    mask=mask,
                )
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.diffusion_scheduler.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.diffusion_scheduler.q_posterior(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            if weights is None:
                terms["mse"] = mean_flat((target - model_output) ** 2, mask=mask)
            else:
                weight = extract_into_tensor(weights, t, target.shape)
                terms["mse"] = mean_flat(
                    weight * (target - model_output) ** 2, mask=mask
                )
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        loss = terms["loss"].mean()

        loss_dict = {}
        log_prefix = "train" if self.training else "val"

        loss_dict.update({f"{log_prefix}/loss": loss})
        loss_dict.update({f"{log_prefix}/loss_mse": terms["mse"].mean()})
        if "vb" in terms:
            loss_dict.update({f"{log_prefix}/loss_vb": terms["vb"].mean()})

        return loss, loss_dict


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(IDDPM):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["given_betas"])

        kwargs = self.add_given_betas_to_config(
            kwargs["given_betas"], kwargs
        )  # add the 'given_betas' into the 'diffusion_scheduler_config'
        base_diffusion = IDDPM(**kwargs)  # pylint: disable=missing-kwoa

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(
            base_diffusion.diffusion_scheduler.alphas_cumprod
        ):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["given_betas"] = torch.FloatTensor(new_betas)
        kwargs = self.add_given_betas_to_config(new_betas, kwargs)
        super().__init__(**kwargs)
        self.map_tensor = torch.tensor(self.timestep_map)  # TODO: get device

    def add_given_betas_to_config(self, given_betas, kwargs):
        given_betas_list = list(given_betas)
        given_betas_list = [float(x) for x in given_betas_list]
        kwargs["diffusion_scheduler_config"]["params"]["given_betas"] = given_betas_list
        return kwargs

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.map_tensor, self.original_num_steps)

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, map_tensor, original_num_steps):
        self.model = model
        self.map_tensor = map_tensor
        # self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        new_ts = self.map_tensor[ts].to(device=ts.device, dtype=ts.dtype)
        # if self.rescale_timesteps:
        #     new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


class LatentDiffusion(SpacedDiffusion):
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
        # Added for LVDM
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
        num_sampling_steps=None,  # Added for SpacedDiffusion
        timestep_respacing=None,  # Added for SpacedDiffusion
        *args,
        **kwargs,
    ):
        self.scale_by_std = scale_by_std
        # for backwards compatibility after implementation of DiffusionWrapper
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        conditioning_key = default(conditioning_key, "crossattn")

        given_betas = get_named_beta_schedule(
            "linear", diffusion_scheduler_config.params.get("timesteps", 1000)
        )
        if num_sampling_steps is not None:
            assert timestep_respacing is None
            timestep_respacing = str(num_sampling_steps)
        if timestep_respacing is None or timestep_respacing == "":
            timestep_respacing = [
                diffusion_scheduler_config.params.get("timesteps", 1000)
            ]

        super().__init__(
            use_timesteps=space_timesteps(
                diffusion_scheduler_config.params.get("timesteps", 1000),
                timestep_respacing,
            ),
            given_betas=given_betas,
            conditioning_key=conditioning_key,
            diffusion_scheduler_config=diffusion_scheduler_config,
            *args,
            **kwargs,
        )

        # add support for auto gradient checkpointing
        from videotuna.opensora.acceleration.checkpoint import set_grad_checkpoint

        set_grad_checkpoint(self.model)

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
        except:
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

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(
            given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s
        )

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

    @torch.no_grad()
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
            try:
                cond = {
                    key: [cond["c_crossattn"][0]["y"]],
                    "mask": [cond["c_crossattn"][0]["mask"]],
                }
            except:
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

        # If model is provided as a keyword argument, use it to train the variance.
        if "model" not in kwargs:
            x_recon = self.model(x_noisy, t, **cond, **kwargs)
        else:
            x_recon = kwargs["model"](x_noisy, t)

        if isinstance(x_recon, tuple):
            return x_recon[0]
        else:
            return x_recon

    def p_losses(
        self,
        x_start,
        cond,
        t,
        noise=None,
        model_kwargs=None,
        mask=None,
        weights=None,
        **kwargs,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.diffusion_scheduler.q_sample(x_start=x_start, t=t, noise=noise)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.diffusion_scheduler.q_sample(x_start, t0, noise=noise)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        model_output = self.apply_model(x_t, t, cond, **kwargs)

        terms = {}
        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    x_start=x_start,
                    x_t=x_t,
                    c=cond,
                    t=t,
                    clip_denoised=False,
                    mask=mask,
                    model=lambda *args, r=frozen_out: r,
                    **kwargs,
                )
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.diffusion_scheduler.q_posterior(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            if weights is None:
                terms["mse"] = mean_flat((target - model_output) ** 2, mask=mask)
            else:
                weight = extract_into_tensor(weights, t, target.shape)
                terms["mse"] = mean_flat(
                    weight * (target - model_output) ** 2, mask=mask
                )
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        loss = terms["loss"].mean()

        loss_dict = {}
        log_prefix = "train" if self.training else "val"

        loss_dict.update({f"{log_prefix}/loss": loss})
        loss_dict.update({f"{log_prefix}/loss_mse": terms["mse"].mean()})
        if "vb" in terms:
            loss_dict.update({f"{log_prefix}/loss_vb": terms["vb"].mean()})

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

    def _vb_terms_bpd(
        self,
        x_start,
        x_t,
        c,
        t,
        clip_denoised=True,
        model_kwargs=None,
        mask=None,
        **kwargs,
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.diffusion_scheduler.q_posterior(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.diffusion_scheduler.p_mean_variance(
            x_t, c, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs, **kwargs
        )
        model_mean, posterior_variance, posterior_log_variance = out
        kl = normal_kl(
            true_mean, true_log_variance_clipped, model_mean, posterior_log_variance
        )
        kl = mean_flat(kl, mask=mask) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=model_mean, log_scales=0.5 * posterior_log_variance
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll, mask=mask) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return output

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
        return self.p_sample_loop(
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
