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

"""IterableDataset for pretokenized JSONL token shards.

This loader targets a fixed-length packed-sequence single JSONL file.
Each line is a JSON object with a token list (key ``"tokens"`` or ``"input_ids"``)
whose length is an integer multiple of ``seq_len`` (or exactly ``seq_len``).
The dataset emits one independent sequence block at a time.
"""

from __future__ import annotations

from bisect import bisect_right
import json
import logging
import os
from dataclasses import dataclass
from typing import Iterator

from torch.utils.data import IterableDataset, get_worker_info

from nemo_automodel.components.datasets.llm.nanogpt_dataset import (
    _get_start_end_pos_single_file,
)

__all__ = ["JsonlTokenDataset", "JsonlTokenDatasetConfig", "load_jsonl_shard"]


logger = logging.getLogger(__name__)


def _extract_tokens(obj: dict) -> list[int]:
    """Extract the token list from a JSON object."""
    if "tokens" in obj:
        return obj["tokens"]
    if "input_ids" in obj:
        return obj["input_ids"]
    raise KeyError(
        f"JSONL line must contain 'tokens' or 'input_ids'. Found keys: {list(obj.keys())}"
    )


def extract_loss(obj: dict) -> float:
    """Extract the loss value from a JSON object."""
    if "loss" in obj:
        return float(obj["loss"])
    if "king_loss" in obj:
        return float(obj["king_loss"])
    raise KeyError(
        f"JSONL line must contain 'loss' or 'king_loss'. Found keys: {list(obj.keys())}"
    )


def load_jsonl_shard(file_path: str | os.PathLike) -> tuple[list[list[int]], list[float]]:
    """Load a single JSONL token shard as a list of token sequences."""
    sequences: list[list[int]] = []
    losses: list[float] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                tokens = _extract_tokens(obj)
                loss = extract_loss(obj)
                sequences.append(tokens)
                losses.append(loss)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                raise ValueError(
                    f"Failed to parse line {line_num} in {file_path}: {e}"
                ) from e
    return sequences, losses


def _peek_num_sequences(file_path: str | os.PathLike) -> int:
    """Count the number of non-empty lines (sequences) without fully parsing tokens."""
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _peek_blocks_per_sequence(file_path: str | os.PathLike, seq_len: int) -> list[int]:
    """Count how many fixed-length training blocks each JSONL record yields."""
    blocks_per_sequence: list[int] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                tokens = _extract_tokens(obj)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                raise ValueError(
                    f"Failed to parse line {line_num} in {file_path}: {e}"
                ) from e

            if len(tokens) < seq_len:
                raise ValueError(
                    f"Sequence {line_num} in {file_path} has length {len(tokens)}, "
                    f"which is shorter than seq_len={seq_len}"
                )
            blocks_per_sequence.append(len(tokens) // seq_len)
    return blocks_per_sequence


@dataclass
class JsonlTokenDatasetConfig:
    """Construction-time configuration for :class:`JsonlTokenDataset`."""

    file_path: str | os.PathLike
    """Path to the single JSONL shard file."""
    seq_len: int
    """Length of each packed training sequence in tokens."""
    num_validation_samples: int | None = None
    """Optional finite cap used for validation instead of the default infinite stream."""

    def build(self) -> "JsonlTokenDataset":
        return JsonlTokenDataset(
            file_path=self.file_path,
            seq_len=self.seq_len,
            num_validation_samples=self.num_validation_samples,
        )


class JsonlTokenDataset(IterableDataset):
    """Stream independent next-token samples from a single fixed-length JSONL shard."""

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
        self.sequences = _peek_num_sequences(self.file_path)
        self.blocks_per_sequence = _peek_blocks_per_sequence(self.file_path, self.seq_len)
        self.total_samples = sum(self.blocks_per_sequence)
        self.sample_offsets = [0]
        for blocks in self.blocks_per_sequence:
            self.sample_offsets.append(self.sample_offsets[-1] + blocks)
        if num_validation_samples is not None and num_validation_samples <= 0:
            raise ValueError("num_validation_samples must be positive when provided")
        self.num_validation_samples = num_validation_samples

    def _rank_sample_range(self) -> tuple[int, int]:
        total_samples = self.total_samples
        if self.num_validation_samples is not None:
            total_samples = min(total_samples, self.num_validation_samples)

        try:
            import torch.distributed as dist

            world_size = dist.get_world_size() if dist.is_initialized() else 1
            rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            world_size = 1
            rank = 0

        return _get_start_end_pos_single_file(total_samples, world_size, rank)

    def _setup_worker_context(
        self,
    ) -> tuple[int, int | None]:
        rank_start, rank_end = self._rank_sample_range()
        worker = get_worker_info()

        if worker is None:
            return rank_start, rank_end

        local_total_samples = rank_end - rank_start
        worker_start, worker_end = _get_start_end_pos_single_file(
            local_total_samples,
            worker.num_workers,
            worker.id,
        )
        return rank_start + worker_start, rank_start + worker_end

    def _process_file_tokens(
        self,
        start_sample: int,
        end_sample: int | None,
    ) -> Iterator[dict]:
        sequences, losses = load_jsonl_shard(self.file_path)

        if losses is None or len(losses) != len(sequences):
            logger.warning(
                "Losses are missing or do not match the number of sequences in %s; "
                "Mu_hat and LCB metrics will be skipped.",
                self.file_path,
            )
            losses = [None] * len(sequences)

        max_sample = (
            min(end_sample, self.total_samples)
            if end_sample is not None
            else self.total_samples
        )

        sample_idx = start_sample
        seq_idx = bisect_right(self.sample_offsets, start_sample) - 1

        while sample_idx < max_sample and seq_idx < len(sequences):
            tokens = sequences[seq_idx]
            loss = losses[seq_idx]
            seq_sample_start = self.sample_offsets[seq_idx]
            seq_sample_end = self.sample_offsets[seq_idx + 1]
            block_idx = sample_idx - seq_sample_start

            while sample_idx < max_sample and (seq_sample_start + block_idx) < seq_sample_end:
                pos = block_idx * self.seq_len
                buf = tokens[pos : pos + self.seq_len]
                input_ids = list(map(int, buf[:-1]))
                labels = list(map(int, buf[1:]))
                yield {"input_ids": input_ids, "labels": labels, "loss": loss}
                block_idx += 1
                sample_idx += 1

            seq_idx += 1

    def _get_file_iterator(
        self,
        start_sample: int,
        end_sample: int | None,
    ) -> Iterator[dict]:
        yield from self._process_file_tokens(start_sample, end_sample)

    def __iter__(self) -> Iterator[dict]:
        start_sample, end_sample = self._setup_worker_context()
        if self.num_validation_samples is not None:
            yield from self._get_file_iterator(start_sample, end_sample)
            return

        while True:  # infinite stream (same as original)
            yield from self._get_file_iterator(start_sample, end_sample)

    def __len__(self) -> int:
        rank_start, rank_end = self._rank_sample_range()
        return rank_end - rank_start

    def __getitem__(self, index: int):
        rank_start, rank_end = self._rank_sample_range()
        local_samples = rank_end - rank_start
        if index < 0 or index >= local_samples:
            raise IndexError(
                f"Index {index} out of range for dataset with {local_samples} local samples."
            )
        global_index = rank_start + index
        return next(self._process_file_tokens(global_index, global_index + 1))