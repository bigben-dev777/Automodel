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

"""Config for XiaomiMiMo/MiMo-V2.5-Pro and Teutonic-II (both use model_type=mimo_v2).

Teutonic-II is a scaled-down MiMoV2-style MoE with one shared expert per MoE
layer, BF16 precision, and fused QKV attention — all captured by this config.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class MiMoV2Config(PretrainedConfig):
    """Configuration shared by MiMo-V2.5-Pro and Teutonic-II.

    Both models declare model_type="mimo_v2" and MiMoV2ForCausalLM in their
    config.json.  The main Teutonic-II extension is n_shared_experts=1 (always
    0/None for MiMo-V2.5-Pro).
    """

    model_type = "mimo_v2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}

    def __init__(
        self,
        vocab_size: int = 152576,
        hidden_size: int = 4096,
        intermediate_size: int = 16384,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 70,
        num_attention_heads: int = 128,
        num_key_value_heads: int = 8,
        head_dim: int = 192,
        v_head_dim: int = 128,
        swa_num_attention_heads: int | None = None,
        swa_num_key_value_heads: int | None = None,
        swa_head_dim: int | None = None,
        swa_v_head_dim: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        initializer_range: float = 0.02,
        layernorm_epsilon: float = 1e-5,
        rms_norm_eps: float | None = None,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10_000_000.0,
        swa_rope_theta: float = 10_000.0,
        rope_scaling: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        attention_value_scale: float | None = None,
        attention_projection_layout: str = "fused_qkv",
        add_full_attention_sink_bias: bool = False,
        add_swa_attention_sink_bias: bool = True,
        hybrid_block_size: int | None = None,
        hybrid_layer_pattern: list[int] | None = None,
        partial_rotary_factor: float = 0.334,
        sliding_window: int | None = 128,
        sliding_window_size: int | None = 128,
        attention_chunk_size: int | None = 128,
        n_routed_experts: int | None = 256,
        n_shared_experts: int | None = None,
        shared_expert_inter_dim: int | None = None,
        num_experts_per_tok: int = 8,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float | None = None,
        moe_layer_freq: list[int] | None = None,
        # Accept both "torch_dtype" (MiMo-V2.5-Pro) and "dtype" (Teutonic-II).
        torch_dtype: str = "bfloat16",
        dtype: str | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.swa_num_attention_heads = swa_num_attention_heads if swa_num_attention_heads is not None else num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads if swa_num_key_value_heads is not None else self.num_key_value_heads
        self.swa_head_dim = swa_head_dim if swa_head_dim is not None else head_dim
        self.swa_v_head_dim = swa_v_head_dim if swa_v_head_dim is not None else v_head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layernorm_epsilon = layernorm_epsilon
        self.rms_norm_eps = layernorm_epsilon if rms_norm_eps is None else rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.swa_rope_theta = swa_rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attention_value_scale = attention_value_scale
        self.attention_projection_layout = attention_projection_layout

        self.add_full_attention_sink_bias = add_full_attention_sink_bias
        self.add_swa_attention_sink_bias = add_swa_attention_sink_bias

        if hybrid_block_size is not None and hybrid_layer_pattern is None:
            hybrid_layer_pattern = [0 if ((i + 1) % hybrid_block_size == 0) else 1 for i in range(num_hidden_layers)]
        if hybrid_layer_pattern is None:
            hybrid_layer_pattern = [0 if i % 6 == 0 or i == num_hidden_layers - 1 else 1 for i in range(num_hidden_layers)]
        self.hybrid_block_size = hybrid_block_size
        self.hybrid_layer_pattern = hybrid_layer_pattern
        self.layer_types = [
            "sliding_attention" if hybrid_layer_pattern[i] == 1 else "full_attention"
            for i in range(num_hidden_layers)
        ]

        self.partial_rotary_factor = partial_rotary_factor
        self.sliding_window = sliding_window
        self.sliding_window_size = sliding_window_size
        self.attention_chunk_size = attention_chunk_size

        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.shared_expert_inter_dim = shared_expert_inter_dim
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_layer_freq = moe_layer_freq if moe_layer_freq is not None else [0] + [1] * (num_hidden_layers - 1)

        # "dtype" (Teutonic-II) takes precedence over "torch_dtype" (MiMo-V2.5-Pro).
        self.torch_dtype = dtype if dtype is not None else torch_dtype

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            **kwargs,
        )
