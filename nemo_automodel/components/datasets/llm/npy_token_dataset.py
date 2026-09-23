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

This loader targets fixed-length packed-sequence shards such as those produced by
``datasets/build_npy_files.py`` in the surrounding workspace. Each shard stores a
flat token array whose length is an integer multiple of ``seq_len``. The dataset
emits one independent sequence block at a time and masks the last label position
with ``-100`` so no cross-block next-token supervision leaks across packed
sequence boundaries.
"""

from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from nemo_automodel.components.datasets.llm.nanogpt_dataset import (
    _get_start_end_pos_single_file,
    _get_worker_id_and_total_workers,
)

__all__ = ["NpyTokenDataset", "NpyTokenDatasetConfig", "load_npy_shard"]


def load_npy_shard(path: str | os.PathLike) -> np.ndarray:
    """Load a token shard as a read-only flattened NumPy memmap/array."""
    tokens = np.load(path, mmap_mode="r", allow_pickle=False)
    if not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError(f"Expected integer token dtype in {path}, got {tokens.dtype}")
    return tokens.reshape(-1)


def _peek_num_tokens(path: str | os.PathLike) -> int:
    return int(load_npy_shard(path).size)


def _count_sequences_in_shard(path: str | os.PathLike, seq_len: int) -> int:
    total_tokens = _peek_num_tokens(path)
    if total_tokens % seq_len != 0:
        raise ValueError(
            f"Token shard {path} has {total_tokens} tokens, which is not divisible by seq_len={seq_len}"
        )
    return total_tokens // seq_len


@dataclass
class NpyTokenDatasetConfig:
    """Construction-time configuration for :class:`NpyTokenDataset`."""

    file_pattern: str | Sequence[str]
    """Glob pattern or explicit list of ``.npy`` shard file paths."""
    seq_len: int
    """Length of each packed training sequence in tokens."""
    shuffle_files: bool = False
    """Shuffle shard order between epochs."""
    num_val_samples: int | None = None
    """Optional cap on emitted samples, primarily for validation datasets."""

    def build(self) -> "NpyTokenDataset":
        return NpyTokenDataset(
            file_pattern=self.file_pattern,
            seq_len=self.seq_len,
            shuffle_files=self.shuffle_files,
            num_val_samples=self.num_val_samples,
        )


class NpyTokenDataset(IterableDataset):
    """Stream independent next-token samples from fixed-length ``.npy`` shards."""

    def __init__(
        self,
        file_pattern: str | Sequence[str],
        seq_len: int,
        *,
        shuffle_files: bool = False,
        num_val_samples: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(file_pattern, (str, Path)):
            self.files: List[str] = sorted(glob.glob(str(file_pattern)))
        else:
            self.files = list(map(str, file_pattern))
        if not self.files:
            raise FileNotFoundError(f"No files matched pattern {file_pattern}")
        self.seq_len = int(seq_len)
        self.shuffle_files = shuffle_files
        if num_val_samples is not None and int(num_val_samples) < 0:
            raise ValueError("num_val_samples must be non-negative when provided")
        self.num_val_samples = None if num_val_samples is None else int(num_val_samples)

    def _setup_worker_context(
        self,
    ) -> tuple[List[str], bool, int, int | None]:
        worker = get_worker_info()
        rng = random.Random()
        if worker is not None:
            rng.seed(worker.id + 12345)
        else:
            rng.seed(os.getpid())

        global_worker_id, total_workers = _get_worker_id_and_total_workers(worker)
        worker_files = self.files[global_worker_id::total_workers].copy()
        if not worker_files:
            worker_files = self.files.copy()

        split_single_file = len(worker_files) == 1 and total_workers > 1
        file_start_pos = 0
        file_end_pos = None
        if split_single_file:
            total_tokens = _peek_num_tokens(worker_files[0])
            total_sequences = total_tokens // self.seq_len
            start_seq, end_seq = _get_start_end_pos_single_file(
                total_sequences, total_workers, global_worker_id
            )
            file_start_pos = start_seq * self.seq_len
            file_end_pos = end_seq * self.seq_len

        if self.shuffle_files:
            rng.shuffle(worker_files)

        return worker_files, split_single_file, file_start_pos, file_end_pos

    def _process_file_tokens(
        self,
        file: str,
        split_single_file: bool,
        file_start_pos: int,
        file_end_pos: int | None,
    ) -> Iterator[dict]:
        tokens = load_npy_shard(file)
        if tokens.size % self.seq_len != 0:
            raise ValueError(
                f"Token shard {file} has {tokens.size} tokens, which is not divisible by seq_len={self.seq_len}"
            )

        if split_single_file:
            pos = file_start_pos
            max_pos = (
                min(file_end_pos, tokens.size)
                if file_end_pos is not None
                else tokens.size
            )
        else:
            pos = 0
            max_pos = tokens.size

        while pos + self.seq_len <= max_pos:
            buf = tokens[pos : pos + self.seq_len]
            input_ids = buf.astype(np.int32, copy=False).tolist()
            labels = buf[1:].astype(np.int64, copy=False).tolist()
            labels.append(-100)
            yield {"input_ids": input_ids, "labels": labels}
            pos += self.seq_len

    def _get_file_iterator(
        self,
        worker_files: List[str],
        split_single_file: bool,
        file_start_pos: int,
        file_end_pos: int | None,
    ) -> Iterator[dict]:
        for file in worker_files:
            yield from self._process_file_tokens(
                file,
                split_single_file,
                file_start_pos,
                file_end_pos,
            )

    def __iter__(self) -> Iterator[dict]:
        worker_files, split_single_file, file_start_pos, file_end_pos = (
            self._setup_worker_context()
        )
        yield from self._get_file_iterator(
            worker_files,
            split_single_file,
            file_start_pos,
            file_end_pos,
        )

    def __len__(self) -> int:  # type: ignore[override]
        return sum(_count_sequences_in_shard(file, self.seq_len) for file in self.files)

    def __getitem__(self, index: int):
        raise NotImplementedError("__getitem__ is not implemented for NpyTokenDataset.")
