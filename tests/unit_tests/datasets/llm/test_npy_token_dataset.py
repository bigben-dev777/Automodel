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

import numpy as np
import pytest

from nemo_automodel.components.datasets.llm.npy_token_dataset import (
    NpyTokenDataset,
    load_npy_shard,
)


def test_load_npy_shard_returns_flat_integer_array(tmp_path):
    shard = tmp_path / "tokens.npy"
    np.save(shard, np.array([[1, 2], [3, 4]], dtype=np.int32))

    tokens = load_npy_shard(shard)

    assert tokens.shape == (4,)
    assert tokens.tolist() == [1, 2, 3, 4]


def test_npy_token_dataset_iterates_each_multi_file_shard_once(tmp_path):
    shard_a = tmp_path / "a.npy"
    shard_b = tmp_path / "b.npy"
    np.save(shard_a, np.arange(8, dtype=np.int32))
    np.save(shard_b, np.arange(8, 16, dtype=np.int32))

    dataset = NpyTokenDataset([shard_a, shard_b], seq_len=4)

    samples = list(dataset)

    assert len(dataset) == 4
    assert len(samples) == 4
    assert samples[0] == {"input_ids": [0, 1, 2, 3], "labels": [1, 2, 3, -100]}
    assert samples[-1] == {"input_ids": [12, 13, 14, 15], "labels": [13, 14, 15, -100]}


def test_npy_token_dataset_rejects_non_divisible_shards(tmp_path):
    shard = tmp_path / "tokens.npy"
    np.save(shard, np.arange(5, dtype=np.int32))

    dataset = NpyTokenDataset([shard], seq_len=4)

    with pytest.raises(ValueError, match="not divisible by seq_len"):
        len(dataset)
