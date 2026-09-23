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

"""State dict adapter for MiMoV2 (MiMo-V2.5-Pro and Teutonic-II).

Differences vs MiMoV2FlashStateDictAdapter:
  • Attention keys are identical in HF and Automodel (both use qkv_proj) —
    no renaming needed.
  • Supports both FP8 (MiMo-V2.5-Pro, has *_scale_inv companions) and plain
    BF16 (Teutonic-II, no scale_inv keys) checkpoints.
  • Shared expert weights (mlp.shared_experts.*) pass through as plain tensors;
    only routed experts (mlp.experts.{E}.*) are regrouped.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import (
    create_scale_inv_for_weight,
    dequantize_from_fp8,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

logger = logging.getLogger(__name__)

# Keys that are never FP8-quantized, even in quantized checkpoints.
_NON_QUANTIZED_PATTERNS = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "norm.weight",
    "lm_head.weight",
    "embed_tokens.weight",
    "mlp.gate.weight",
    "self_attn.o_proj.weight",
    # shared expert weights are stored in BF16 even in FP8 checkpoints
    "mlp.shared_experts.",
    "mlp.gate.e_score_correction_bias",
]


def _should_quantize_key(key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    return not any(pat in key for pat in _NON_QUANTIZED_PATTERNS)


class MiMoV2StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Convert MiMoV2 HF checkpoints to Automodel's grouped MoE layout.

    Handles both:
      - FP8 quantized checkpoints (MiMo-V2.5-Pro): dequantizes weights before
        regrouping experts.
      - Plain BF16 checkpoints (Teutonic-II): skips dequantization, shared
        expert weights flow through unchanged.

    HF format stores routed experts as per-expert split projections:
        mlp.experts.{E}.{gate,up,down}_proj.weight
    Automodel groups these into gate_and_up_projs / down_projs tensors for EP.
    """

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

    # ------------------------------------------------------------------
    # HF → Automodel
    # ------------------------------------------------------------------

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        del kwargs
        # Detect prefix from expert keys.
        for key in hf_state_dict:
            if ".mlp.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = key.startswith("model.")
                break
        # Dequantize FP8 if scale_inv companions are present.
        hf_state_dict = self._dequantize_if_fp8(hf_state_dict)
        return self._from_hf_w_merged_experts(hf_state_dict, device_mesh)

    def _dequantize_if_fp8(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Dequantize FP8 weights when *_scale_inv keys are found."""
        scale_keys_to_remove: list[str] = []
        dequantized = 0
        for key in list(state_dict):
            if not key.endswith(".weight"):
                continue
            scale_key = key + "_scale_inv"
            if scale_key not in state_dict:
                continue
            state_dict[key] = dequantize_from_fp8(
                state_dict[key],
                state_dict[scale_key],
                dtype=self.dtype,
                name=key,
            )
            scale_keys_to_remove.append(scale_key)
            dequantized += 1
        for k in scale_keys_to_remove:
            state_dict.pop(k, None)
        if dequantized:
            logger.debug("[MiMoV2 FP8 Dequant] Dequantized %d weights", dequantized)
        return state_dict

    # ------------------------------------------------------------------
    # Automodel → HF
    # ------------------------------------------------------------------

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert Automodel state dict back to the HF MiMoV2 layout (BF16)."""
        hf_state_dict: dict[str, Any] = {}
        for fqn, tensor in state_dict.items():
            for key, value in self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
            ):
                hf_state_dict[key] = value
        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        result = expert_result if expert_result is not None else [(fqn, tensor)]
        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]
        return result