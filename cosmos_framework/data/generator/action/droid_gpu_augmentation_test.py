# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from cosmos_framework.data.generator.action.droid_gpu_augmentation import (
    DROID_DEFERRED_AUGMENTATION_KEY,
    DROID_DEFERRED_COLOR_FACTORS_KEY,
    DROID_DEFERRED_COLOR_ORDER_KEY,
    DROID_DEFERRED_CROP_KEY,
    DROID_DEFERRED_FRAME_CHUNK_KEY,
    DROID_DEFERRED_LOGICAL_SIZE_KEY,
    apply_deferred_droid_augmentation,
    prepare_deferred_droid_augmentation,
)
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.joint_dataloader import JointDataLoader


def _make_discrete_float_views(t: int, h: int, w: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return [
        torch.randint(0, 256, (t, 3, h, w), dtype=torch.uint8, generator=generator).float() / 255.0 for _ in range(3)
    ]


def _cpu_reference(
    views: list[torch.Tensor],
    *,
    seed: int,
) -> torch.Tensor:
    wrist, left, right = views
    t, _, h, w = wrist.shape
    torch.manual_seed(seed)
    augmentor = T.Compose(
        [
            T.RandomCrop((int(h * 0.95), int(w * 0.95))),
            T.Resize((h, w), antialias=True),
            T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
        ]
    )
    combined = augmentor(torch.cat([wrist, left, right], dim=0))
    wrist, left, right = combined[:t], combined[t : 2 * t], combined[2 * t :]
    left = F.interpolate(left, size=(h // 2, w // 2), mode="bilinear", align_corners=False)
    right = F.interpolate(right, size=(h // 2, w // 2), mode="bilinear", align_corners=False)
    composite = torch.cat([wrist, torch.cat([left, right], dim=-1)], dim=-2)
    return (composite * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)


def test_chunked_deferred_augmentation_matches_worker_path_for_discrete_inputs() -> None:
    t, h, w = 5, 64, 80
    seed = 123
    views = _make_discrete_float_views(t, h, w)
    reference = _cpu_reference(views, seed=seed)

    torch.manual_seed(seed)
    packed_video, metadata = prepare_deferred_droid_augmentation(
        *views,
        frame_chunk_size=2,
    )
    actual = apply_deferred_droid_augmentation(
        packed_video,
        crop_params=metadata[DROID_DEFERRED_CROP_KEY],
        color_order=metadata[DROID_DEFERRED_COLOR_ORDER_KEY],
        color_factors=metadata[DROID_DEFERRED_COLOR_FACTORS_KEY],
        image_size=torch.tensor([h + h // 2, w, h + h // 2, w]),
        frame_chunk_size=int(metadata[DROID_DEFERRED_FRAME_CHUNK_KEY].item()),
    )

    assert packed_video.shape == (9, t, h, w)
    assert packed_video.dtype == torch.uint8
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_action_transform_uses_logical_composite_size_without_worker_resize() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=64,
        format_prompt_as_json=True,
    )
    raw_transport = torch.zeros(9, 33, 360, 640, dtype=torch.uint8)
    sample = {
        "ai_caption": "Move the robot.",
        "video": raw_transport,
        "action": torch.zeros(32, 8),
        "conditioning_fps": torch.tensor(15),
        "mode": "wam",
        "domain_id": torch.tensor(0),
        "viewpoint": "concat_view",
        DROID_DEFERRED_AUGMENTATION_KEY: torch.tensor(True),
        DROID_DEFERRED_CROP_KEY: torch.tensor([1, 2, 342, 608, 360, 640]),
        DROID_DEFERRED_COLOR_ORDER_KEY: torch.tensor([0, 1, 2, 3]),
        DROID_DEFERRED_COLOR_FACTORS_KEY: torch.tensor([1.0, 1.0, 1.0, 0.0]),
        DROID_DEFERRED_LOGICAL_SIZE_KEY: torch.tensor([540, 640]),
        DROID_DEFERRED_FRAME_CHUNK_KEY: torch.tensor(8),
    }

    result = pipeline(sample, resolution="480")

    assert result["video"] is raw_transport
    assert result["video"].shape == (9, 33, 360, 640)
    torch.testing.assert_close(
        result["image_size"],
        torch.tensor([544.0, 736.0, 540.0, 640.0]),
    )
    assert result["sequence_plan"].condition_frame_indexes_vision == [0]
    assert result["ai_caption"]["resolution"] == {"H": 544, "W": 736}


def test_packer_counts_deferred_logical_target_shape() -> None:
    loader = object.__new__(JointDataLoader)
    loader.tokenizer_spatial_compression_factor = 16
    loader.tokenizer_temporal_compression_factor = 4
    loader.patch_spatial = 2
    loader.uniae_chunk_frames = None
    loader.uniae_pad_frames = None
    loader.sound_latent_fps = 0
    loader.audio_sample_rate = 48000

    sample = {
        "video": [torch.zeros(9, 33, 360, 640, dtype=torch.uint8)],
        "image_size": torch.tensor([544.0, 736.0, 540.0, 640.0]),
        "action": [torch.zeros(32, 64)],
        DROID_DEFERRED_AUGMENTATION_KEY: torch.tensor([True]),
    }

    # H/W must come from target 544x736:
    # ceil((544/16)/2) * ceil((736/16)/2) * (1+(33-1)/4)
    # = 17 * 23 * 9 vision tokens, plus 32 action tokens.
    assert loader._compute_num_tokens_per_sample(sample) == 17 * 23 * 9 + 32


def test_model_consumes_packed_metadata_before_normalization() -> None:
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    t, h, w = 3, 32, 48
    views = _make_discrete_float_views(t, h, w)
    packed_video, metadata = prepare_deferred_droid_augmentation(
        *views,
        frame_chunk_size=2,
    )
    logical_h = h + h // 2
    data_batch = {
        "video": [[packed_video]],
        "image_size": [torch.tensor([logical_h, w, logical_h, w])],
        **{key: [value.unsqueeze(0)] for key, value in metadata.items()},
    }

    model = object.__new__(OmniMoTModel)
    model.input_video_key = "video"
    model.tensor_kwargs_fp32 = {"device": "cpu", "dtype": torch.float32}
    model._apply_deferred_droid_augmentation_inplace(data_batch)
    model._normalize_video_databatch_inplace(data_batch)

    assert data_batch["video"][0].shape == (1, 3, t, logical_h, w)
    assert data_batch["video"][0].dtype == torch.float32
    assert data_batch["is_preprocessed"] is True
    assert all(key not in data_batch for key in metadata)
