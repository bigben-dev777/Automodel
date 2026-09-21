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

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mimo_v25.config import TeutonicIIConfig
from nemo_automodel.components.models.mimo_v25.model import TeutonicIIForCausalLM
from nemo_automodel.components.moe.layers import MoE


def test_teutonic_ii_builds_shared_experts_from_mimo_runtime():
    config = TeutonicIIConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        v_head_dim=8,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=8,
        swa_v_head_dim=8,
        max_position_embeddings=64,
        layernorm_epsilon=1e-6,
        rope_theta=10000.0,
        swa_rope_theta=10000.0,
        attention_projection_layout="fused_qkv",
        partial_rotary_factor=0.5,
        sliding_window=4,
        sliding_window_size=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        moe_layer_freq=[0, 1],
        hybrid_layer_pattern=[0, 1],
        torch_dtype="float32",
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )

    model = TeutonicIIForCausalLM(config, backend=backend)

    assert "rotary_emb" in model._keep_in_fp32_modules_strict
    assert isinstance(model.model.layers["1"].mlp, MoE)
    assert model.model.layers["1"].mlp.shared_experts is not None
