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

import itertools
import json

import numpy as np
import pytest
import torch.distributed as dist

from nemo_automodel.components.datasets.llm.jsonl_token_dataset import JsonlTokenDataset, JsonlTokenDatasetConfig
from nemo_automodel.components.datasets.llm.npy_token_dataset import NpyTokenDataset, NpyTokenDatasetConfig


def test_npy_token_dataset_is_infinite_by_default(tmp_path):
    shard = tmp_path / "tokens.npy"
    np.save(shard, np.arange(8, dtype=np.int32))

    dataset = NpyTokenDataset(shard, seq_len=4)

    samples = list(itertools.islice(dataset, 3))
    assert len(samples) == 3
    assert samples[0]["input_ids"] == [0, 1, 2]
    assert samples[1]["input_ids"] == [4, 5, 6]
    assert samples[2]["input_ids"] == [0, 1, 2]


def test_npy_token_dataset_num_validation_samples_is_finite(tmp_path):
    shard = tmp_path / "tokens.npy"
    np.save(shard, np.arange(12, dtype=np.int32))

    dataset = NpyTokenDatasetConfig(file_path=shard, seq_len=4, num_validation_samples=2).build()

    samples = list(dataset)
    assert len(samples) == 2
    assert len(dataset) == 2
    assert samples[0]["input_ids"] == [0, 1, 2]
    assert samples[1]["input_ids"] == [4, 5, 6]


def test_jsonl_token_dataset_is_infinite_by_default(tmp_path):
    shard = tmp_path / "tokens.jsonl"
    shard.write_text(
        "\n".join(
            [
                json.dumps({"tokens": [10, 11, 12, 13], "loss": 1.5}),
                json.dumps({"input_ids": [20, 21, 22, 23], "loss": 2.5}),
            ]
        ),
        encoding="utf-8",
    )

    dataset = JsonlTokenDataset(shard, seq_len=4)

    samples = list(itertools.islice(dataset, 3))
    assert len(samples) == 3
    assert samples[0]["input_ids"] == [10, 11, 12]
    assert samples[0]["loss"] == 1.5
    assert samples[1]["input_ids"] == [20, 21, 22]
    assert samples[2]["input_ids"] == [10, 11, 12]


def test_jsonl_token_dataset_num_validation_samples_is_finite(tmp_path):
    shard = tmp_path / "tokens.jsonl"
    shard.write_text(
        "\n".join(
            [
                json.dumps({"tokens": [10, 11, 12, 13], "loss": 1.5}),
                json.dumps({"input_ids": [20, 21, 22, 23], "loss": 2.5}),
                json.dumps({"tokens": [30, 31, 32, 33], "loss": 3.5}),
            ]
        ),
        encoding="utf-8",
    )

    dataset = JsonlTokenDatasetConfig(file_path=shard, seq_len=4, num_validation_samples=2).build()

    samples = list(dataset)
    assert len(samples) == 2
    assert len(dataset) == 2
    assert samples[0]["input_ids"] == [10, 11, 12]
    assert samples[1]["input_ids"] == [20, 21, 22]


@pytest.mark.parametrize(
    ("dataset_cls", "factory", "expected_inputs"),
    [
        (
            NpyTokenDataset,
            lambda shard: np.save(shard, np.arange(16, dtype=np.int32)),
            ([8, 9, 10], [12, 13, 14]),
        ),
        (
            JsonlTokenDataset,
            lambda shard: shard.write_text(
                "\n".join(
                    [
                        json.dumps({"tokens": [0, 1, 2, 3], "loss": 0.1}),
                        json.dumps({"tokens": [4, 5, 6, 7], "loss": 0.2}),
                        json.dumps({"tokens": [8, 9, 10, 11], "loss": 0.3}),
                        json.dumps({"tokens": [12, 13, 14, 15], "loss": 0.4}),
                    ]
                ),
                encoding="utf-8",
            ),
            ([8, 9, 10], [12, 13, 14]),
        ),
    ],
)
def test_token_iterable_dataset_len_and_getitem_are_rank_local(
    tmp_path, monkeypatch, dataset_cls, factory, expected_inputs
):
    shard = tmp_path / ("tokens.npy" if dataset_cls is NpyTokenDataset else "tokens.jsonl")
    factory(shard)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)

    dataset = dataset_cls(shard, seq_len=4)

    assert len(dataset) == 2
    assert dataset[0]["input_ids"] == list(expected_inputs[0])
    assert dataset[1]["input_ids"] == list(expected_inputs[1])
    with pytest.raises(IndexError, match="local"):
        _ = dataset[2]


@pytest.mark.parametrize("dataset_cls", [NpyTokenDataset, JsonlTokenDataset])
def test_validation_sample_limit_must_be_positive(dataset_cls, tmp_path):
    if dataset_cls is NpyTokenDataset:
        shard = tmp_path / "tokens.npy"
        np.save(shard, np.arange(8, dtype=np.int32))
    else:
        shard = tmp_path / "tokens.jsonl"
        shard.write_text(json.dumps({"tokens": [1, 2, 3, 4], "loss": 0.5}), encoding="utf-8")

    with pytest.raises(ValueError, match="num_validation_samples must be positive"):
        dataset_cls(shard, seq_len=4, num_validation_samples=0)