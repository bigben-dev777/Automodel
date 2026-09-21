# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Portions copyright 2026 Xiaomi Corporation.
# Portions copyright 2026 The HuggingFace Inc. team.
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

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mimo_v25.model import MiMoV2ForCausalLM, MiMoV2Model
from nemo_automodel.components.models.teutonic_ii.config import TeutonicIIConfig
from nemo_automodel.components.models.teutonic_ii.state_dict_adapter import TeutonicIIStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class TeutonicIIModel(MiMoV2Model):
    """Teutonic-II backbone reusing the MiMo v2.5 decoder implementation."""


class TeutonicIIForCausalLM(MiMoV2ForCausalLM):
    """Teutonic-II causal LM wrapper with its own package entrypoint."""

    @classmethod
    def from_config(
        cls,
        config: TeutonicIIConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ) -> "TeutonicIIForCausalLM":
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ) -> "TeutonicIIForCausalLM":
        config = TeutonicIIConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: TeutonicIIConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__(config, moe_config=moe_config, backend=backend, **kwargs)
        resolved_moe_config = moe_config
        if resolved_moe_config is None:
            adapter = getattr(self, "state_dict_adapter", None)
            resolved_moe_config = getattr(adapter, "moe_config", None)
        if resolved_moe_config is None:
            dtype = get_dtype(config.torch_dtype, torch.bfloat16)
            resolved_moe_config = MoEConfig(
                dim=config.hidden_size,
                inter_dim=config.intermediate_size,
                moe_inter_dim=config.moe_intermediate_size,
                n_routed_experts=int(config.n_routed_experts or 0),
                n_shared_experts=int(config.n_shared_experts or 0),
                n_activated_experts=int(config.num_experts_per_tok or 0),
                n_expert_groups=int(config.n_group or 0),
                n_limited_groups=int(config.topk_group or 0),
                train_gate=True,
                gate_bias_update_factor=0.0,
                score_func=(
                    "sigmoid_with_bias"
                    if config.scoring_func == "sigmoid"
                    else config.scoring_func
                ),
                route_scale=(
                    config.routed_scaling_factor
                    if config.routed_scaling_factor is not None
                    else 1.0
                ),
                aux_loss_coeff=0.0,
                norm_topk_prob=config.norm_topk_prob,
                router_bias=False,
                expert_bias=False,
                expert_activation="swiglu",
                softmax_before_topk=False,
                force_e_score_correction_bias=True,
                dtype=dtype,
            )

        self.model = TeutonicIIModel(config, resolved_moe_config, self.backend)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = TeutonicIIStateDictAdapter(
                self.config,
                resolved_moe_config,
                self.backend,
                dtype=get_dtype(config.torch_dtype, torch.bfloat16),
            )


ModelClass = TeutonicIIForCausalLM

__all__ = ["TeutonicIIForCausalLM", "TeutonicIIModel", "ModelClass"]