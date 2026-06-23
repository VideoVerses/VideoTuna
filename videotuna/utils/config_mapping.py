"""Shared OmegaConf mapping helpers for YAML config sections."""

from __future__ import annotations

import re
from typing import Any, Self

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError, model_validator

_DOT_PATH = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")


class ConfigMappingError(ValueError):
    """Raised when a config section mapping is invalid or references missing paths."""


class ConfigPathMappings(BaseModel):
    root: dict[str, str]

    @model_validator(mode="after")
    def validate_dot_paths(self) -> Self:
        for source, target in self.root.items():
            if not _DOT_PATH.match(source):
                raise ValueError(f"invalid source path {source!r}")
            if not _DOT_PATH.match(target):
                raise ValueError(f"invalid target path {target!r}")
        return self


def config_path_exists(cfg: DictConfig, path: str) -> bool:
    """Return True when every dot segment of ``path`` exists in ``cfg``."""
    if not path:
        return False

    node: object = cfg
    for segment in path.split("."):
        if not OmegaConf.is_config(node):
            return False
        if segment not in node:
            return False
        node = node[segment]
    return True


def get_config_path(cfg: DictConfig, path: str) -> Any:
    """Return the value at ``path`` or raise ``ConfigMappingError`` if missing."""
    if not config_path_exists(cfg, path):
        raise ConfigMappingError(f"config path {path!r} does not exist")
    return OmegaConf.select(cfg, path)


def apply_config_mappings(cfg: DictConfig, *, section: str = "train") -> None:
    """Validate and apply ``{section}.mapping`` entries (source -> target)."""
    mapping = OmegaConf.select(cfg, f"{section}.mapping")
    if mapping is None:
        return

    if not OmegaConf.is_dict(mapping):
        raise ConfigMappingError(
            f"{section}.mapping must be a mapping of source paths to target paths, "
            f"got {type(mapping).__name__}"
        )

    raw_mapping = OmegaConf.to_container(mapping, resolve=False)
    if not isinstance(raw_mapping, dict):
        raise ConfigMappingError(
            f"{section}.mapping must be a mapping of source paths to target paths, "
            f"got {type(raw_mapping).__name__}"
        )

    try:
        validated = ConfigPathMappings(root=raw_mapping)
    except ValidationError as exc:
        raise ConfigMappingError(
            f"{section}.mapping contains invalid dot paths: {exc}"
        ) from exc

    for source_path, target_path in validated.root.items():
        if not config_path_exists(cfg, source_path):
            raise ConfigMappingError(
                f"{section}.mapping source path {source_path!r} does not exist "
                f"(entry {source_path!r} -> {target_path!r})"
            )
        if not config_path_exists(cfg, target_path):
            raise ConfigMappingError(
                f"{section}.mapping target path {target_path!r} does not exist "
                f"(entry {source_path!r} -> {target_path!r})"
            )

        value = OmegaConf.select(cfg, source_path)
        if value is not None:
            OmegaConf.update(cfg, target_path, value)
            logger.info(f"update {target_path} by {source_path} value: {value}")
