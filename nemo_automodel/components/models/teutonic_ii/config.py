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

from nemo_automodel.components.models.mimo_v25.config import MiMoV2Config


class TeutonicIIConfig(MiMoV2Config):
    """Configuration for Teutonic-II checkpoints derived from MiMo v2.5."""

    model_type = "teutonic_ii"

    def __init__(self, *args, **kwargs):
        attention_projection_layout = kwargs.get("attention_projection_layout", "split")
        n_routed_experts = kwargs.get("n_routed_experts")
        n_shared_experts = kwargs.get("n_shared_experts")
        moe_intermediate_size = kwargs.get("moe_intermediate_size")
        num_experts_per_tok = kwargs.get("num_experts_per_tok")
        routed_scaling_factor = kwargs.get("routed_scaling_factor")
        scoring_func = kwargs.get("scoring_func", "sigmoid")
        topk_method = kwargs.get("topk_method", "noaux_tc")
        n_group = kwargs.get("n_group")
        topk_group = kwargs.get("topk_group")
        norm_topk_prob = kwargs.get("norm_topk_prob", True)
        moe_layer_freq = kwargs.get("moe_layer_freq")

        super().__init__(*args, **kwargs)

        self.attention_projection_layout = attention_projection_layout
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = (
            moe_intermediate_size
            if moe_intermediate_size is not None
            else self.intermediate_size
        )
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.moe_layer_freq = (
            moe_layer_freq
            if moe_layer_freq is not None
            else getattr(self, "moe_layer_freq", None)
        )


__all__ = ["TeutonicIIConfig"]