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

import json
import os
from dataclasses import dataclass
from typing import Iterator

from torch.utils.data import IterableDataset, get_worker_info

from nemo_automodel.components.datasets.llm.nanogpt_dataset import (
    _get_start_end_pos_single_file,
    _get_worker_id_and_total_workers,
)

__all__ = ["JsonlTokenDataset", "JsonlTokenDatasetConfig", "load_jsonl_shard"]


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


@dataclass
class JsonlTokenDatasetConfig:
    """Construction-time configuration for :class:`JsonlTokenDataset`."""

    file_path: str | os.PathLike
    """Path to the single JSONL shard file."""
    seq_len: int
    """Length of each packed training sequence in tokens."""

    def build(self) -> "JsonlTokenDataset":
        return JsonlTokenDataset(
            file_path=self.file_path,
            seq_len=self.seq_len,
        )


class JsonlTokenDataset(IterableDataset):
    """Stream independent next-token samples from a single fixed-length JSONL shard."""

    def __init__(
        self,
        file_path: str | os.PathLike,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.file_path = str(file_path)
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        self.seq_len = int(seq_len)
        self.sequences = _peek_num_sequences(self.file_path)

    def _setup_worker_context(
        self,
    ) -> tuple[int, int | None]:
        worker = get_worker_info()
        global_worker_id, total_workers = _get_worker_id_and_total_workers(worker)

        total_sequences = self.sequences
        start_seq, end_seq = _get_start_end_pos_single_file(
            total_sequences, total_workers, global_worker_id
        )
        # For JSONL we work in sequence indices, not token positions
        return start_seq, end_seq

    def _process_file_tokens(
        self,
        start_seq: int,
        end_seq: int | None,
    ) -> Iterator[dict]:
        sequences, losses = load_jsonl_shard(self.file_path)

        if losses is None or len(losses) != len(sequences):
            print(f"Warning: Losses are missing or do not match the number of sequences in {self.file_path}, Not calculate Mu_hat and LCB")
            losses = [None] * len(sequences)

        max_seq = (
            min(end_seq, len(sequences))
            if end_seq is not None
            else len(sequences)
        )

        for seq_idx in range(start_seq, max_seq):
            tokens = sequences[seq_idx]
            loss = losses[seq_idx]
            if len(tokens) < self.seq_len:
                # Skip or raise – here we require exact / multiple length
                raise ValueError(
                    f"Sequence {seq_idx} in {self.file_path} has length {len(tokens)}, "
                    f"which is shorter than seq_len={self.seq_len}"
                )

            # Pack fixed-length blocks the same way as the original npy loader
            pos = 0
            while pos + self.seq_len <= len(tokens):
                buf = tokens[pos : pos + self.seq_len]
                input_ids = list(map(int, buf[:-1]))
                labels = list(map(int, buf[1:]))
                yield {"input_ids": input_ids, "labels": labels, "loss": loss}
                pos += self.seq_len

    def _get_file_iterator(
        self,
        start_seq: int,
        end_seq: int | None,
    ) -> Iterator[dict]:
        yield from self._process_file_tokens(start_seq, end_seq)

    def __iter__(self) -> Iterator[dict]:
        start_seq, end_seq = self._setup_worker_context()
        while True:  # infinite stream (same as original)
            yield from self._get_file_iterator(start_seq, end_seq)

    def __len__(self) -> int:
        return self.sequences

    def __getitem__(self, index: int):
        if index < 0 or index >= self.sequences:
            raise IndexError(
                f"Index {index} out of range for dataset with {self.sequences} sequences."
            )
        return next(self._process_file_tokens(index, index + 1))