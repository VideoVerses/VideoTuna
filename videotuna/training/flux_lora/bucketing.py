"""SimpleTuner-style pixel_area aspect bucketing for Flux LoRA training."""

from __future__ import annotations

import math


def round_aspect_ratio(width: int, height: int, rounding: int) -> float:
    if height <= 0:
        raise ValueError(f"Invalid image height: {height}")
    return round(width / height, rounding)


def target_pixel_area(resolution: int) -> int:
    return resolution * resolution


def bucket_dimensions(
    aspect: float,
    target_area: int,
    *,
    align: int = 64,
) -> tuple[int, int]:
    if aspect <= 0:
        raise ValueError(f"Aspect ratio must be positive, got {aspect}")
    height = math.sqrt(target_area / aspect)
    width = height * aspect
    width = max(align, round(width / align) * align)
    height = max(align, round(height / align) * align)
    return int(width), int(height)


def bucket_dimensions_for_image(
    width: int,
    height: int,
    resolution: int,
    resolution_type: str,
    aspect_bucket_rounding: int,
    *,
    align: int = 64,
) -> tuple[int, int]:
    if resolution_type != "pixel_area":
        raise ValueError(
            "Unsupported resolution_type="
            f"{resolution_type!r}; only 'pixel_area' is supported"
        )
    aspect = round_aspect_ratio(width, height, aspect_bucket_rounding)
    area = target_pixel_area(resolution)
    return bucket_dimensions(aspect, area, align=align)


def meets_minimum_size(
    width: int,
    height: int,
    minimum_image_size: int,
    resolution_type: str,
) -> bool:
    if minimum_image_size <= 0:
        return True
    if resolution_type != "pixel_area":
        raise ValueError(
            "Unsupported resolution_type="
            f"{resolution_type!r}; only 'pixel_area' is supported"
        )
    min_area = minimum_image_size * minimum_image_size
    return width * height >= min_area
