"""Load Flux components and inject PEFT LoRA adapters."""

from __future__ import annotations

from typing import Any, cast

import torch
from diffusers import AutoencoderKL, FluxTransformer2DModel
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

FLUX_LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]


def load_flux_training_models(
    pretrained_model_name_or_path: str,
    lora_rank: int,
    mixed_precision: str = "bf16",
    gradient_checkpointing: bool = True,
):
    weight_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16

    tokenizer_one = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer"
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer_2"
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    )
    text_encoder_two = T5EncoderModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        torch_dtype=weight_dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=FLUX_LORA_TARGET_MODULES,
    )
    transformer = get_peft_model(cast(Any, transformer), lora_config)

    if gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    elif hasattr(transformer, "disable_gradient_checkpointing"):
        transformer.disable_gradient_checkpointing()

    return {
        "tokenizer_one": tokenizer_one,
        "tokenizer_two": tokenizer_two,
        "text_encoder_one": text_encoder_one,
        "text_encoder_two": text_encoder_two,
        "vae": vae,
        "transformer": transformer,
        "weight_dtype": weight_dtype,
    }
