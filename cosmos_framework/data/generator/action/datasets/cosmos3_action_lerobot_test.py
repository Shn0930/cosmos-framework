# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import datasets as hf_datasets
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot as lerobot_adapter
from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    ProjectedLeRobotDataset,
    _load_projected_nested_dataset,
    build_projected_data_columns,
)
from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import (
    DROIDLeRobotDataset,
    _build_droid_delta_timestamps,
)
from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset_config import (
    _GRIPPER_STATE_FEATURE,
    _JOINT_ACTION_FEATURE,
    _JOINT_STATE_FEATURE,
)

_STATE_FEATURE = "observation.state.cartesian_position"
_ACTION_FEATURE = "action.gripper_position"
_IMAGE_FEATURES = {
    "wrist": "observation.image.wrist",
    "left": "observation.image.left",
    "right": "observation.image.right",
}
_CORE_COLUMNS = ("timestamp", "episode_index", "index", "task_index")
_JOINT_COLUMNS = (
    *_CORE_COLUMNS,
    _ACTION_FEATURE,
    _JOINT_ACTION_FEATURE,
    _JOINT_STATE_FEATURE,
    _GRIPPER_STATE_FEATURE,
)


def _write_synthetic_parquet(root: Path) -> tuple[Path, hf_datasets.Features]:
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    path = data_dir / "file-000.parquet"
    table = pa.table(
        {
            "timestamp": pa.array([0.0, 0.1, 0.2, 0.0, 0.1, 0.2], type=pa.float32()),
            "episode_index": pa.array([0, 0, 0, 1, 1, 1], type=pa.int64()),
            "index": pa.array(range(6), type=pa.int64()),
            "task_index": pa.array([0, 0, 0, 1, 1, 1], type=pa.int64()),
            _ACTION_FEATURE: pa.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], type=pa.float32()),
            _JOINT_ACTION_FEATURE: pa.array(
                [[0.0, 1.0], [0.1, 1.1], [0.2, 1.2], [0.3, 1.3], [0.4, 1.4], [0.5, 1.5]],
                type=pa.list_(pa.float32()),
            ),
            _JOINT_STATE_FEATURE: pa.array(
                [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4], [1.5, 2.5]],
                type=pa.list_(pa.float32()),
            ),
            _GRIPPER_STATE_FEATURE: pa.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.0], type=pa.float32()),
            "unused.velocity": pa.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], type=pa.float32()),
        }
    )
    pq.write_table(table, path)
    features = hf_datasets.Features(
        {
            "timestamp": hf_datasets.Value("float32"),
            "episode_index": hf_datasets.Value("int64"),
            "index": hf_datasets.Value("int64"),
            "task_index": hf_datasets.Value("int64"),
            _ACTION_FEATURE: hf_datasets.Value("float32"),
            _JOINT_ACTION_FEATURE: hf_datasets.Sequence(hf_datasets.Value("float32"), length=2),
            _JOINT_STATE_FEATURE: hf_datasets.Sequence(hf_datasets.Value("float32"), length=2),
            _GRIPPER_STATE_FEATURE: hf_datasets.Value("float32"),
            "unused.velocity": hf_datasets.Value("float32"),
        }
    )
    return path, features


def test_projected_parquet_pushes_columns_and_matches_full_data(tmp_path: Path, monkeypatch) -> None:
    path, all_features = _write_synthetic_parquet(tmp_path)
    full = hf_datasets.Dataset.from_parquet(
        str(path),
        features=all_features,
        cache_dir=str(tmp_path / "full-cache"),
    )
    projected_features = hf_datasets.Features({column: all_features[column] for column in _JOINT_COLUMNS})

    calls: list[list[str] | None] = []
    original_from_parquet = hf_datasets.Dataset.from_parquet

    def tracked_from_parquet(*args, **kwargs):
        calls.append(kwargs.get("columns"))
        return original_from_parquet(*args, **kwargs)

    monkeypatch.setattr(hf_datasets.Dataset, "from_parquet", staticmethod(tracked_from_parquet))
    projected = _load_projected_nested_dataset(
        tmp_path / "data",
        columns=_JOINT_COLUMNS,
        features=projected_features,
        cache_dir=tmp_path / "projected-cache",
    )

    assert calls == [list(_JOINT_COLUMNS)]
    assert projected.column_names == list(_JOINT_COLUMNS)
    assert projected.to_dict() == full.select_columns(list(_JOINT_COLUMNS)).to_dict()
    assert "unused.velocity" not in projected.column_names


def test_projected_parquet_preserves_episode_filter(tmp_path: Path) -> None:
    _, all_features = _write_synthetic_parquet(tmp_path)
    projected_features = hf_datasets.Features({column: all_features[column] for column in _JOINT_COLUMNS})

    projected = _load_projected_nested_dataset(
        tmp_path / "data",
        columns=_JOINT_COLUMNS,
        features=projected_features,
        episodes=[1],
    )

    assert projected.column_names == list(_JOINT_COLUMNS)
    assert projected["episode_index"] == [1, 1, 1]
    assert projected["index"] == [3, 4, 5]


def test_projected_parquet_reports_missing_columns(tmp_path: Path) -> None:
    _write_synthetic_parquet(tmp_path)
    requested = (*_JOINT_COLUMNS, "missing.action")

    with pytest.raises(
        ValueError,
        match=r"Projected LeRobot parquet columns missing.*file-000\.parquet.*missing\.action",
    ):
        _load_projected_nested_dataset(
            tmp_path / "data",
            columns=requested,
        )


def test_projected_lerobot_override_uses_metadata_projection(tmp_path: Path, monkeypatch) -> None:
    _, all_features = _write_synthetic_parquet(tmp_path)
    lerobot_features = {
        name: {
            "dtype": getattr(feature, "feature", feature).dtype,
            "shape": (feature.length,) if hasattr(feature, "length") else (1,),
        }
        for name, feature in all_features.items()
    }
    projected = object.__new__(ProjectedLeRobotDataset)
    projected._data_columns = _JOINT_COLUMNS
    projected.root = tmp_path
    projected.episodes = None
    projected.meta = SimpleNamespace(features=lerobot_features)

    original_loader = _load_projected_nested_dataset

    def load_with_test_cache(*args, **kwargs):
        kwargs["cache_dir"] = tmp_path / "override-cache"
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(lerobot_adapter, "_load_projected_nested_dataset", load_with_test_cache)
    hf_dataset = projected.load_hf_dataset()
    item = hf_dataset[0]

    assert hf_dataset.column_names == list(_JOINT_COLUMNS)
    assert set(item) == set(_JOINT_COLUMNS)
    assert "unused.velocity" not in item


def test_joint_policy_projects_only_joint_and_gripper_columns() -> None:
    delta_timestamps = _build_droid_delta_timestamps(
        action_space="joint_pos",
        state_feature=_STATE_FEATURE,
        action_feature=_ACTION_FEATURE,
        image_features=_IMAGE_FEATURES,
        viewpoint="concat_view",
        use_state=True,
        max_num_history_actions=0,
        dt=1 / 15,
        chunk_length=32,
    )
    columns = build_projected_data_columns(
        delta_timestamps,
        video_keys=tuple(_IMAGE_FEATURES.values()),
    )

    assert columns == _JOINT_COLUMNS
    assert _STATE_FEATURE not in delta_timestamps
    assert all(video_key not in columns for video_key in _IMAGE_FEATURES.values())
    assert len(delta_timestamps[_JOINT_ACTION_FEATURE]) == 32
    assert len(delta_timestamps[_JOINT_STATE_FEATURE]) == 33


def test_cartesian_policy_projects_only_cartesian_and_gripper_columns() -> None:
    delta_timestamps = _build_droid_delta_timestamps(
        action_space="midtrain",
        state_feature=_STATE_FEATURE,
        action_feature=_ACTION_FEATURE,
        image_features=_IMAGE_FEATURES,
        viewpoint="concat_view",
        use_state=True,
        max_num_history_actions=2,
        dt=1 / 15,
        chunk_length=32,
    )
    columns = build_projected_data_columns(
        delta_timestamps,
        video_keys=tuple(_IMAGE_FEATURES.values()),
    )

    assert columns == (*_CORE_COLUMNS, _STATE_FEATURE, _ACTION_FEATURE, _GRIPPER_STATE_FEATURE)
    assert _JOINT_ACTION_FEATURE not in delta_timestamps
    assert all(video_key not in columns for video_key in _IMAGE_FEATURES.values())
    assert len(delta_timestamps[_STATE_FEATURE]) == 35
    assert len(delta_timestamps[_ACTION_FEATURE]) == 34
    assert len(delta_timestamps[_GRIPPER_STATE_FEATURE]) == 33


def test_val_temp_seg_reads_each_record_from_its_own_shard() -> None:
    class SliceDataset:
        def __init__(self, gripper: list[float]) -> None:
            self._data = {
                _ACTION_FEATURE: [torch.tensor(value, dtype=torch.float32) for value in gripper],
                _STATE_FEATURE: [torch.zeros(6, dtype=torch.float32) for _ in gripper],
            }

        def __getitem__(self, index: slice) -> dict[str, list]:
            assert isinstance(index, slice)
            return {key: values[index] for key, values in self._data.items()}

    dataset = object.__new__(DROIDLeRobotDataset)
    dataset._chunk_length = 2
    dataset._action_features = _ACTION_FEATURE
    dataset._state_features = _STATE_FEATURE
    dataset._is_flat_action = False
    dataset._episode_records = [
        (0, 0, 1, 10),
        (0, 0, 1, 11),
        (1, 0, 1, 20),
    ]

    shards = {
        0: SimpleNamespace(hf_dataset=SliceDataset([1.0, 1.0, 1.0])),
        1: SimpleNamespace(hf_dataset=SliceDataset([0.0, 1.0, 1.0])),
    }
    loaded_shards: list[int] = []

    def get_dataset(ds_idx: int):
        loaded_shards.append(ds_idx)
        return shards[ds_idx]

    dataset._get_dataset = get_dataset
    dataset._apply_temp_seg_filter()

    assert loaded_shards == [0, 1]
    assert dataset._episode_records == [(1, 0, 1, 20)]
    assert dataset._episode_cum_ends == [1]
    assert dataset._num_valid_indices == 1
