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

"""IterableDataset for .npy shards produced by the Teutonic-II build_datasets.py pipeline.

Shard format (written by build_datasets.py)::

    np.save(path, arr)          # arr: np.ndarray, dtype=uint32, shape=(N * seq_len,)

Every consecutive block of ``seq_len`` tokens in the flat array is one packed
training sequence (documents are concatenated with EOS tokens; tails shorter
than ``seq_len`` are discarded).  This dataset slides a window of
``seq_len + 1`` tokens across each shard (step = ``seq_len``) and returns:

    input_ids : tokens[pos : pos + seq_len]          (int32)
    labels    : tokens[pos + 1 : pos + seq_len + 1]  (int64, shifted by 1)

Shard distribution across distributed ranks and DataLoader workers mirrors
the strategy used by NanogptDataset: each global worker (rank × num_workers)
receives a non-overlapping subset of shards.
"""

from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


class NpyShardDataset(IterableDataset):
    """IterableDataset over pre-tokenized .npy shards.

    Args:
        file_pattern: Glob pattern or explicit list of ``.npy`` shard paths.
        seq_len: Tokens per training sample (labels are ``input_ids`` shifted by 1).
        shuffle_files: Shuffle shard order each pass.
        loop: If ``True`` (default), cycle through shards forever — use for
            training so ``max_steps`` controls termination.  Set ``False`` for
            validation to do a single finite pass.
    """

    def __init__(
        self,
        file_pattern: str | Sequence[str],
        seq_len: int,
        *,
        shuffle_files: bool = True,
        loop: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(file_pattern, (str, Path)):
            self.files = sorted(glob.glob(str(file_pattern)))
        else:
            self.files = list(map(str, file_pattern))
        if not self.files:
            raise FileNotFoundError(f"No .npy shards matched: {file_pattern!r}")
        self.seq_len = int(seq_len)
        self.shuffle_files = shuffle_files
        self.loop = loop

    # ------------------------------------------------------------------
    # Worker / rank helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _global_worker_info() -> tuple[int, int]:
        """Return (global_worker_id, total_workers) across ranks and DL workers."""
        try:
            import torch.distributed as dist
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            world_size, rank = 1, 0

        wi = get_worker_info()
        n_dl = wi.num_workers if wi is not None else 1
        dl_id = wi.id if wi is not None else 0

        return rank * n_dl + dl_id, world_size * n_dl

    # ------------------------------------------------------------------
    # Shard iteration
    # ------------------------------------------------------------------

    def _iter_shard(self, path: str) -> Iterator[dict]:
        """Yield (input_ids, labels) dicts from a single .npy shard."""
        arr = np.load(path, mmap_mode="r")
        # Cast to int64 once to avoid repeated per-sample casting.
        tokens = torch.from_numpy(arr.astype(np.int64))
        n = len(tokens)
        pos = 0
        while pos + self.seq_len + 1 <= n:
            chunk = tokens[pos : pos + self.seq_len + 1]
            yield {
                "input_ids": chunk[:-1].to(torch.int32).tolist(),
                "labels": chunk[1:].to(torch.int64).tolist(),
            }
            pos += self.seq_len

    def __iter__(self) -> Iterator[dict]:
        global_id, total = self._global_worker_info()

        # Each global worker owns a strided slice of shards.
        worker_files = self.files[global_id::total] or self.files

        rng = random.Random(global_id + 31337)
        if self.shuffle_files:
            rng.shuffle(worker_files)

        while True:
            for path in worker_files:
                yield from self._iter_shard(path)
            if not self.loop:
                return
            # Reshuffle at the start of each new epoch.
            if self.shuffle_files:
                rng.shuffle(worker_files)
