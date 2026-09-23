# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from itertools import islice

import numpy as np
import pytest

from nemo_automodel.components.datasets.llm.npy_token_dataset import NpyTokenDataset, NpyTokenDatasetConfig


def _write_shard(path, values: list[int]) -> str:
    np.save(path, np.asarray(values, dtype=np.uint32))
    return str(path)


def test_num_val_samples_caps_iteration(tmp_path):
    shard = _write_shard(tmp_path / "tokens.npy", list(range(16)))
    dataset = NpyTokenDataset(file_pattern=[shard], seq_len=4, num_val_samples=3)

    samples = list(dataset)

    assert len(samples) == 3
    assert samples[0]["input_ids"] == [0, 1, 2, 3]
    assert samples[0]["labels"] == [1, 2, 3, -100]
    assert samples[-1]["input_ids"] == [8, 9, 10, 11]


def test_dataset_repeats_without_validation_cap(tmp_path):
    shard = _write_shard(tmp_path / "tokens.npy", list(range(8)))
    dataset = NpyTokenDataset(file_pattern=[shard], seq_len=4)

    samples = list(islice(iter(dataset), 3))

    assert len(samples) == 3
    assert samples[0]["input_ids"] == [0, 1, 2, 3]
    assert samples[1]["input_ids"] == [4, 5, 6, 7]
    assert samples[2]["input_ids"] == [0, 1, 2, 3]


def test_config_build_forwards_num_val_samples(tmp_path):
    shard = _write_shard(tmp_path / "tokens.npy", list(range(16)))
    config = NpyTokenDatasetConfig(file_pattern=[shard], seq_len=4, num_val_samples=2)

    dataset = config.build()

    assert isinstance(dataset, NpyTokenDataset)
    assert dataset.num_val_samples == 2


def test_negative_num_val_samples_rejected(tmp_path):
    shard = _write_shard(tmp_path / "tokens.npy", list(range(8)))

    with pytest.raises(ValueError, match="num_val_samples"):
        NpyTokenDataset(file_pattern=[shard], seq_len=4, num_val_samples=-1)