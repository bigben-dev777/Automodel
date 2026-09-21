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

from nemo_automodel.components.models.mimo_v25.config import MiMoV2Config
from nemo_automodel.components.models.teutonic_ii.config import TeutonicIIConfig


class TestTeutonicIIConfig:
    def test_model_type(self):
        cfg = TeutonicIIConfig()
        assert cfg.model_type == "teutonic_ii"

    def test_shared_expert_fields_match_teutonic_shape(self):
        cfg = TeutonicIIConfig(
            hidden_size=6144,
            intermediate_size=16384,
            moe_intermediate_size=1024,
            num_hidden_layers=45,
            num_attention_heads=48,
            num_key_value_heads=4,
            head_dim=192,
            v_head_dim=128,
            swa_num_attention_heads=48,
            swa_num_key_value_heads=8,
            swa_head_dim=192,
            swa_v_head_dim=128,
            n_routed_experts=256,
            n_shared_experts=1,
            num_experts_per_tok=8,
            n_group=1,
            topk_group=1,
            moe_layer_freq=[0] + [1] * 44,
            hybrid_layer_pattern=[0, 1, 1, 1, 1] * 9,
            attention_projection_layout="fused_qkv",
            partial_rotary_factor=0.334,
        )

        assert cfg.num_local_experts == 256
        assert cfg.n_shared_experts == 1
        assert cfg.moe_intermediate_size == 1024
        assert cfg.attention_projection_layout == "fused_qkv"

    def test_base_mimo_config_still_available(self):
        cfg = MiMoV2Config(num_hidden_layers=2)
        assert cfg.model_type == "mimo_v2"