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

import tempfile
from pathlib import Path

import numpy as np

from nemo_automodel.components.datasets.llm.npy_token_dataset import (
    NpyTokenDataset,
    load_npy_shard,
)


def _make_npy_shard(tmpdir: Path, name: str, tokens: np.ndarray) -> Path:
    path = tmpdir / name
    np.save(path, tokens.astype(np.uint32), allow_pickle=False)
    return path


def test_load_npy_shard_returns_flat_integer_array():
    with tempfile.TemporaryDirectory() as tmp:
        shard = _make_npy_shard(
            Path(tmp), "shard.npy", np.arange(8, dtype=np.uint32).reshape(2, 4)
        )
        loaded = load_npy_shard(shard)
        assert loaded.shape == (8,)
        assert loaded.dtype == np.uint32


def test_npy_token_dataset_iteration_masks_last_label():
    with tempfile.TemporaryDirectory() as tmp:
        shard = _make_npy_shard(
            Path(tmp),
            "shard.npy",
            np.array([10, 11, 12, 13, 20, 21, 22, 23], dtype=np.uint32),
        )
        ds = NpyTokenDataset(str(shard), seq_len=4, shuffle_files=False)

        samples = []
        for i, sample in enumerate(ds):
            if i >= 2:
                break
            samples.append(sample)

        assert samples[0]["input_ids"] == [10, 11, 12, 13]
        assert samples[0]["labels"] == [11, 12, 13, -100]
        assert samples[1]["input_ids"] == [20, 21, 22, 23]
        assert samples[1]["labels"] == [21, 22, 23, -100]


def test_npy_token_dataset_raises_for_nondivisible_shard():
    with tempfile.TemporaryDirectory() as tmp:
        shard = _make_npy_shard(
            Path(tmp), "bad.npy", np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        )
        ds = NpyTokenDataset(str(shard), seq_len=4)
        iterator = iter(ds)
        try:
            next(iterator)
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "not divisible by seq_len" in str(exc)
