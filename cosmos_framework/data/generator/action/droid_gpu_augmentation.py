# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Deferred CUDA image augmentation for the DROID concat-view policy recipe.

The regular DROID path composes and augments decoded floating-point frames in
DataLoader workers.  The optional deferred path keeps the three camera views as
one compact ``uint8`` tensor and records the already-sampled random parameters.
The model applies the same spatial/color operations after the batch has moved to
its device, immediately before the existing video normalization step.  The
transport is quantized to uint8 first, so arbitrary floating-point decoder
outputs are numerically near-equivalent rather than bit-identical to the CPU
path.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as transforms_F
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as transforms_v2_F

DROID_DEFERRED_AUGMENTATION_KEY = "droid_deferred_gpu_augmentation"
DROID_DEFERRED_CROP_KEY = "droid_deferred_crop_params"
DROID_DEFERRED_COLOR_ORDER_KEY = "droid_deferred_color_order"
DROID_DEFERRED_COLOR_FACTORS_KEY = "droid_deferred_color_factors"
DROID_DEFERRED_LOGICAL_SIZE_KEY = "droid_deferred_logical_size"
DROID_DEFERRED_FRAME_CHUNK_KEY = "droid_deferred_frame_chunk"

DROID_DEFERRED_METADATA_KEYS = (
    DROID_DEFERRED_AUGMENTATION_KEY,
    DROID_DEFERRED_CROP_KEY,
    DROID_DEFERRED_COLOR_ORDER_KEY,
    DROID_DEFERRED_COLOR_FACTORS_KEY,
    DROID_DEFERRED_LOGICAL_SIZE_KEY,
    DROID_DEFERRED_FRAME_CHUNK_KEY,
)

_DROID_COLOR_JITTER = T.ColorJitter(
    brightness=0.3,
    contrast=0.4,
    saturation=0.5,
    hue=0.08,
)


def prepare_deferred_droid_augmentation(
    wrist: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    frame_chunk_size: int = 8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pack three decoded DROID views and sample the worker-side RNG state.

    Args:
        wrist: Wrist camera video in ``[T, 3, H, W]`` float ``[0, 1]`` format.
        left: Left exterior camera video with the same shape and dtype.
        right: Right exterior camera video with the same shape and dtype.

    Returns:
        A ``[9, T, H, W]`` uint8 tensor plus flat tensor metadata sufficient
        to reproduce crop -> resize -> ColorJitter on the model device.
    """
    views = (wrist, left, right)
    if frame_chunk_size <= 0:
        raise ValueError(f"Deferred DROID augmentation frame_chunk_size must be positive, got {frame_chunk_size}.")
    expected_shape = tuple(wrist.shape)
    if wrist.ndim != 4 or wrist.shape[1] != 3:
        raise ValueError(f"Deferred DROID augmentation expects wrist video shape [T,3,H,W], got {tuple(wrist.shape)}.")
    for name, view in zip(("wrist", "left", "right"), views, strict=True):
        if tuple(view.shape) != expected_shape:
            raise ValueError(
                "Deferred DROID augmentation requires equal camera shapes: "
                f"wrist={expected_shape}, {name}={tuple(view.shape)}."
            )
        if not torch.is_floating_point(view):
            raise TypeError(
                f"Deferred DROID augmentation expects floating-point decoded {name} frames, got {view.dtype}."
            )

    _, _, h, w = wrist.shape
    crop_h, crop_w = int(h * 0.95), int(w * 0.95)
    crop_params = T.RandomCrop((crop_h, crop_w)).make_params([wrist])
    color_params = _DROID_COLOR_JITTER.make_params([wrist])

    # Match BaseActionLeRobotDataset._convert_video exactly: multiply, clamp,
    # truncate to uint8, then convert TCHW -> CTHW.  Convert each view before
    # concatenation so workers never allocate a 3-view floating-point copy.
    formatted_views = [
        (view * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3).contiguous() for view in views
    ]
    packed_video = torch.cat(formatted_views, dim=0)  # [9,T,H,W]

    metadata = {
        DROID_DEFERRED_AUGMENTATION_KEY: torch.tensor(True),
        DROID_DEFERRED_CROP_KEY: torch.tensor(
            [
                crop_params["top"],
                crop_params["left"],
                crop_params["height"],
                crop_params["width"],
                h,
                w,
            ],
            dtype=torch.int64,
        ),
        DROID_DEFERRED_COLOR_ORDER_KEY: color_params["fn_idx"].to(dtype=torch.int64),
        DROID_DEFERRED_COLOR_FACTORS_KEY: torch.tensor(
            [
                color_params["brightness_factor"],
                color_params["contrast_factor"],
                color_params["saturation_factor"],
                color_params["hue_factor"],
            ],
            dtype=torch.float32,
        ),
        # Logical post-composition dimensions, not the temporary 9-channel
        # transport tensor dimensions.
        DROID_DEFERRED_LOGICAL_SIZE_KEY: torch.tensor([h + h // 2, w], dtype=torch.int64),
        DROID_DEFERRED_FRAME_CHUNK_KEY: torch.tensor(frame_chunk_size, dtype=torch.int64),
    }
    return packed_video, metadata


def apply_deferred_droid_augmentation(
    packed_video: torch.Tensor,
    *,
    crop_params: torch.Tensor,
    color_order: torch.Tensor,
    color_factors: torch.Tensor,
    image_size: torch.Tensor,
    frame_chunk_size: int = 8,
) -> torch.Tensor:
    """Apply the deferred DROID pipeline on ``packed_video.device``.

    The spatial/color operation order matches the existing worker path, with
    the documented uint8 transport quantization occurring before these steps:

    ``crop -> bilinear resize -> ColorJitter -> exterior-view downsample ->
    compose -> uint8 quantization -> bicubic resize -> reflection/edge pad``.

    The returned video deliberately remains uint8 so the model's existing
    normalization path remains the single owner of conversion to ``[-1, 1]``.
    """
    if packed_video.ndim != 4 or packed_video.shape[0] != 9:
        raise ValueError(
            f"Deferred DROID augmentation expects packed video shape [9,T,H,W], got {tuple(packed_video.shape)}."
        )
    if packed_video.dtype != torch.uint8:
        raise TypeError(f"Deferred DROID augmentation expects uint8 video, got {packed_video.dtype}.")
    if frame_chunk_size <= 0:
        raise ValueError(f"Deferred DROID augmentation frame_chunk_size must be positive, got {frame_chunk_size}.")

    crop_values = _metadata_values(crop_params, "crop_params", expected=6, cast=int)
    color_order_values = _metadata_values(color_order, "color_order", expected=4, cast=int)
    color_factor_values = _metadata_values(color_factors, "color_factors", expected=4, cast=float)
    image_size_values = _metadata_values(image_size, "image_size", expected=4, cast=int)

    top, left, crop_h, crop_w, resize_h, resize_w = crop_values
    target_h, target_w, content_h, content_w = image_size_values
    if sorted(color_order_values) != [0, 1, 2, 3]:
        raise ValueError(
            f"Deferred DROID ColorJitter order must be a permutation of [0,1,2,3], got {color_order_values}."
        )

    _, t, h, w = packed_video.shape
    if resize_h != h or resize_w != w:
        raise ValueError(
            "Deferred DROID resize metadata does not match packed video: "
            f"metadata={(resize_h, resize_w)}, video={(h, w)}."
        )
    if top < 0 or left < 0 or top + crop_h > h or left + crop_w > w:
        raise ValueError(f"Deferred DROID crop {(top, left, crop_h, crop_w)} is outside video spatial size {(h, w)}.")

    brightness, contrast, saturation, hue = color_factor_values
    half_h, half_w = resize_h // 2, resize_w // 2
    composite = torch.empty(
        (t, 3, resize_h + half_h, resize_w),
        dtype=torch.float32,
        device=packed_video.device,
    )

    # torchvision's hue conversion has a large temporary footprint when all
    # 3*T frames are processed at once.  Chunk only the execution—not the RNG:
    # crop/order/factors were sampled once above and therefore remain identical
    # for every camera and frame, matching the original concatenated transform.
    for view_index in range(3):
        raw_view = packed_video[view_index * 3 : (view_index + 1) * 3].permute(1, 0, 2, 3)
        for frame_start in range(0, t, frame_chunk_size):
            frame_end = min(frame_start + frame_chunk_size, t)
            view_chunk = raw_view[frame_start:frame_end].to(dtype=torch.float32).div_(255.0)
            view_chunk = view_chunk[..., top : top + crop_h, left : left + crop_w]
            view_chunk = transforms_v2_F.resize(
                view_chunk,
                size=[resize_h, resize_w],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            for operation in color_order_values:
                if operation == 0:
                    view_chunk = transforms_v2_F.adjust_brightness(
                        view_chunk,
                        brightness_factor=brightness,
                    )
                elif operation == 1:
                    view_chunk = transforms_v2_F.adjust_contrast(
                        view_chunk,
                        contrast_factor=contrast,
                    )
                elif operation == 2:
                    view_chunk = transforms_v2_F.adjust_saturation(
                        view_chunk,
                        saturation_factor=saturation,
                    )
                else:
                    view_chunk = transforms_v2_F.adjust_hue(view_chunk, hue_factor=hue)

            if view_index == 0:
                composite[frame_start:frame_end, :, :resize_h, :] = view_chunk
            else:
                view_chunk = F.interpolate(
                    view_chunk,
                    size=(half_h, half_w),
                    mode="bilinear",
                    align_corners=False,
                )
                horizontal_start = 0 if view_index == 1 else half_w
                composite[
                    frame_start:frame_end,
                    :,
                    resize_h:,
                    horizontal_start : horizontal_start + half_w,
                ] = view_chunk

    formatted_video = (
        (composite * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3).contiguous()
    )  # [3,T,H,W]

    logical_h, logical_w = formatted_video.shape[-2:]
    if content_h > target_h or content_w > target_w:
        raise ValueError(f"Deferred DROID content size {(content_h, content_w)} exceeds target {(target_h, target_w)}.")
    if content_h != logical_h or content_w != logical_w:
        formatted_video = transforms_F.resize(
            formatted_video,
            size=[content_h, content_w],
            interpolation=transforms_F.InterpolationMode.BICUBIC,
            antialias=True,
        )

    padding_right = target_w - content_w
    padding_bottom = target_h - content_h
    if padding_right or padding_bottom:
        padding_mode = "edge" if padding_right >= content_w or padding_bottom >= content_h else "reflect"
        formatted_video = transforms_F.pad(
            formatted_video,
            [0, 0, padding_right, padding_bottom],
            padding_mode=padding_mode,
        )

    return formatted_video.contiguous()


def _metadata_values(
    value: torch.Tensor,
    name: str,
    *,
    expected: int,
    cast: type[int] | type[float],
) -> list[Any]:
    """Validate and copy one small metadata tensor to Python once."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Deferred DROID {name} must be a tensor, got {type(value).__name__}.")
    flat = value.detach().reshape(-1)
    if flat.numel() != expected:
        raise ValueError(f"Deferred DROID {name} must contain {expected} values, got shape {tuple(value.shape)}.")
    return [cast(item) for item in flat.cpu().tolist()]
