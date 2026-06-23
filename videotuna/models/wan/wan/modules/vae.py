# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Shim so YAML configs can target ``modules.vae.WanVAE_`` (implemented in vae2_1)."""

from .vae2_1 import Wan2_1_VAE, WanVAE_

__all__ = ["Wan2_1_VAE", "WanVAE_"]
