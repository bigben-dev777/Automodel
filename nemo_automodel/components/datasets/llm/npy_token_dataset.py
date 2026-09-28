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

"""IterableDataset for pretokenized ``.npy`` token shards.

This loader targets a fixed-length packed-sequence single ``.npy`` file.
The file stores a flat token array whose length is an integer multiple of ``seq_len``.
The dataset emits one independent sequence block at a time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from nemo_automodel.components.datasets.llm.nanogpt_dataset import (
    _get_start_end_pos_single_file,
    _get_worker_id_and_total_workers,
)

__all__ = ["NpyTokenDataset", "NpyTokenDatasetConfig", "load_npy_shard"]


def load_npy_shard(file_path: str | os.PathLike) -> np.ndarray:
    """Load a single ``.npy`` token shard as a flat NumPy array."""
    tokens = np.load(file_path, mmap_mode="r", allow_pickle=False)
    return tokens.reshape(-1)


def _peek_num_tokens(file_path: str | os.PathLike) -> int:
    """Peek the number of tokens in a single ``.npy`` token shard without loading the entire array into memory."""
    tokens = np.load(file_path, mmap_mode="r", allow_pickle=False)
    return tokens.size


@dataclass
class NpyTokenDatasetConfig:
    """Construction-time configuration for :class:`NpyTokenDataset`."""

    file_path: str | os.PathLike
    """Path to the single ``.npy`` shard file."""
    seq_len: int
    """Length of each packed training sequence in tokens."""
    num_validation_samples: int | None = None
    """Optional finite cap used for validation instead of the default infinite stream."""

    def build(self) -> "NpyTokenDataset":
        return NpyTokenDataset(
            file_path=self.file_path,
            seq_len=self.seq_len,
            num_validation_samples=self.num_validation_samples,
        )


class NpyTokenDataset(IterableDataset):
    """Stream independent next-token samples from a single fixed-length ``.npy`` shard."""

    def __init__(
        self,
        file_path: str | os.PathLike,
        seq_len: int,
        num_validation_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.file_path = str(file_path)
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        self.seq_len = int(seq_len)
        self.sequences = _peek_num_tokens(self.file_path) // self.seq_len
        if num_validation_samples is not None and num_validation_samples <= 0:
            raise ValueError("num_validation_samples must be positive when provided")
        self.num_validation_samples = num_validation_samples

    def _setup_worker_context(
        self,
    ) -> tuple[int, int | None]:
        worker = get_worker_info()
        global_worker_id, total_workers = _get_worker_id_and_total_workers(worker)

        total_sequences = self.sequences
        if self.num_validation_samples is not None:
            total_sequences = min(total_sequences, self.num_validation_samples)
        start_seq, end_seq = _get_start_end_pos_single_file(
            total_sequences, total_workers, global_worker_id
        )

        file_start_pos = start_seq * self.seq_len
        file_end_pos = end_seq * self.seq_len

        return file_start_pos, file_end_pos

    def _process_file_tokens(
        self,
        file_start_pos: int,
        file_end_pos: int | None,
    ) -> Iterator[dict]:
        tokens = load_npy_shard(self.file_path)
        if tokens.size % self.seq_len != 0:
            raise ValueError(
                f"Token shard {self.file_path} has {tokens.size} tokens, which is not divisible by seq_len={self.seq_len}"
            )

        pos = file_start_pos
        max_pos = (
            min(file_end_pos, tokens.size)
            if file_end_pos is not None
            else tokens.size
        )

        while pos + self.seq_len <= max_pos:
            buf = tokens[pos : pos + self.seq_len]
            input_ids = buf[:-1].astype(np.int32, copy=False).tolist()
            labels = buf[1:].astype(np.int64, copy=False).tolist()
            yield {"input_ids": input_ids, "labels": labels}
            pos += self.seq_len

    def _get_file_iterator(
        self,
        file_start_pos: int,
        file_end_pos: int | None,
    ) -> Iterator[dict]:
        yield from self._process_file_tokens(
            file_start_pos,
            file_end_pos,
        )

    def __iter__(self) -> Iterator[dict]:
        file_start_pos, file_end_pos = self._setup_worker_context()
        if self.num_validation_samples is not None:
            yield from self._get_file_iterator(
                file_start_pos,
                file_end_pos,
            )
            return

        while True:
            yield from self._get_file_iterator(
                file_start_pos,
                file_end_pos,
            )

    def __len__(self) -> int:
        if self.num_validation_samples is not None:
            return min(self.sequences, self.num_validation_samples)
        return self.sequences

    def __getitem__(self, index: int):
        if index < 0 or index >= self.sequences:
            raise IndexError(f"Index {index} out of range for dataset with {self.sequences} sequences.")
        file_start_pos = index * self.seq_len
        file_end_pos = file_start_pos + self.seq_len
        return next(self._process_file_tokens(file_start_pos, file_end_pos))