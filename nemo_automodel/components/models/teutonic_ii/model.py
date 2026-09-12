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

"""MiMoV2ForCausalLM — covers MiMo-V2.5-Pro and Teutonic-II.

Key differences from MiMoV2FlashForCausalLM:
  • Fused QKV projection (qkv_proj) instead of separate q/k/v projections.
  • n_shared_experts support in each MoE layer (Teutonic-II adds one shared expert).
  • Works with both BF16 (Teutonic-II) and FP8 (MiMo-V2.5-Pro) checkpoints.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import (
    _has_dtensor_params,
    cast_model_to_dtype,
    compute_lm_head_logits,
)
# Re-use shared utilities from mimo_v2_flash — rotary, norm, mask helpers,
# eager attention kernel, and MLP/MoE builders are identical.
from nemo_automodel.components.models.mimo_v2_flash.model import (
    MiMoV2FlashRotaryEmbedding,
    MiMoV2RMSNorm,
    _apply_rotary_pos_emb,
    _eager_attention_forward,
    _ensure_additive_mask,
    _fallback_additive_mask,
    _derive_padding_mask,
    _convert_bool_4d_mask_to_additive,
)
from nemo_automodel.components.models.teutonic_ii.config import MiMoV2Config
from nemo_automodel.components.models.teutonic_ii.state_dict_adapter import MiMoV2StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MLP, MoE
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class TeutonicIIMoE(MoE):
    """Teutonic-II MoE: routed dispatch + one unconditional shared expert.

    Teutonic-II's key architectural modification over MiMo-V2.5-Pro is adding
    one shared expert per MoE layer that runs on EVERY token without routing:
        output = top_k_routed_experts(x) + shared_experts(x)

    The shared expert is created explicitly here rather than being buried in
    MoEConfig so that the architectural intent is visible at the model level.
    MoE.forward() handles the combination: y = routed + z when shared_experts
    is not None (standard non-MoK dispatch path).
    """

    def __init__(self, moe_config: MoEConfig, backend: "BackendConfig"):
        n_shared = int(moe_config.n_shared_experts or 0)
        # Initialise the parent with n_shared_experts=0 so we own the creation.
        routed_config = dataclasses.replace(moe_config, n_shared_experts=0, shared_expert_inter_dim=None)
        super().__init__(routed_config, backend)
        # Teutonic-II shared expert: processes all tokens, no routing/gating.
        if n_shared > 0:
            shared_inter = n_shared * (moe_config.shared_expert_inter_dim or moe_config.moe_inter_dim)
            self.shared_experts = MLP(
                moe_config.dim,
                shared_inter,
                backend.linear,
                dtype=moe_config.dtype,
                activation="swiglu",
                bias=False,
            )

try:
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
except ImportError:
    from transformers.modeling_attn_mask_utils import (
        AttentionMaskConverter as _AMC,
    )

    def create_causal_mask(**kw):  # type: ignore[misc]
        return _AMC.make_causal_mask(**kw)

    def create_sliding_window_causal_mask(**kw):  # type: ignore[misc]
        return _AMC.make_sliding_window_causal_mask(**kw)


class MiMoV2Attention(nn.Module):
    """MiMoV2 attention with fused QKV projection (qkv_proj)."""

    def __init__(self, config: MiMoV2Config, backend: BackendConfig, is_swa: bool, layer_idx: int):
        super().__init__()
        self.config = config
        self.backend = backend
        self.layer_idx = layer_idx
        self.is_swa = is_swa

        if is_swa:
            self.head_dim = config.swa_head_dim
            self.v_head_dim = config.swa_v_head_dim
            self.num_attention_heads = config.swa_num_attention_heads
            self.num_key_value_heads = config.swa_num_key_value_heads
        else:
            self.head_dim = config.head_dim
            self.v_head_dim = config.v_head_dim
            self.num_attention_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads

        self.rope_dim = int(self.head_dim * config.partial_rotary_factor)
        self.rope_dim = self.rope_dim - (self.rope_dim % 2)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_dropout = float(config.attention_dropout or 0.0)
        self.scaling = self.head_dim**-0.5
        self.v_scale = getattr(config, "attention_value_scale", None)

        self.q_size = self.num_attention_heads * self.head_dim
        self.k_size = self.num_key_value_heads * self.head_dim
        self.v_size = self.num_key_value_heads * self.v_head_dim

        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        # Single fused projection for Q, K, V — matches HF checkpoint layout.
        self.qkv_proj = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            self.q_size + self.k_size + self.v_size,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.o_proj = initialize_linear_module(
            backend.linear,
            self.num_attention_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            dtype=dtype,
        )

        has_sink = (config.add_full_attention_sink_bias and not is_swa) or (
            config.add_swa_attention_sink_bias and is_swa
        )
        if has_sink:
            self.register_buffer("attention_sink_bias", torch.empty(self.num_attention_heads, dtype=torch.float32))
        else:
            self.attention_sink_bias = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        batch, seq_len = hidden_states.shape[:2]

        qkv = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)

        query_states = query_states.view(batch, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch, seq_len, self.num_key_value_heads, self.v_head_dim).transpose(1, 2)

        if self.v_scale is not None:
            value_states = value_states * self.v_scale

        cos, sin = position_embeddings
        query_rope, query_nope = query_states.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        key_rope, key_nope = key_states.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        query_rope, key_rope = _apply_rotary_pos_emb(query_rope, key_rope, cos, sin)
        query_states = torch.cat([query_rope, query_nope], dim=-1)
        key_states = torch.cat([key_rope, key_nope], dim=-1)

        attn_output, attn_weights = _eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )
        attn_output = attn_output.reshape(batch, seq_len, -1).contiguous()
        return self.o_proj(attn_output), attn_weights

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        del buffer_device
        for linear in (self.qkv_proj, self.o_proj):
            nn.init.normal_(linear.weight, mean=0.0, std=init_std)
            if getattr(linear, "bias", None) is not None:
                nn.init.zeros_(linear.bias)
        if self.attention_sink_bias is not None:
            nn.init.zeros_(self.attention_sink_bias)


class MiMoV2Block(nn.Module):
    """Decoder block using fused-QKV attention and routed + shared MoE."""

    def __init__(self, layer_idx: int, config: MiMoV2Config, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        is_swa = config.hybrid_layer_pattern[layer_idx] == 1
        self.attention_type = "sliding_attention" if is_swa else "full_attention"
        self.self_attn = MiMoV2Attention(config, backend, is_swa=is_swa, layer_idx=layer_idx)

        is_moe_layer = (
            getattr(config, "n_routed_experts", None) is not None and bool(config.moe_layer_freq[layer_idx])
        )
        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        if is_moe_layer:
            # TeutonicIIMoE makes the shared-expert modification explicit;
            # when n_shared_experts=0 it behaves identically to plain MoE.
            self.mlp = TeutonicIIMoE(moe_config, backend)
        else:
            self.mlp = MLP(
                config.hidden_size,
                config.intermediate_size,
                backend.linear,
                dtype=dtype,
                activation="swiglu",
                bias=False,
            )

        self.input_layernorm = MiMoV2RMSNorm(config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype)
        self.post_attention_layernorm = MiMoV2RMSNorm(config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        padding_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, TeutonicIIMoE):
            hidden_states = self.mlp(hidden_states, padding_mask)
        else:
            hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def init_weights(self, buffer_device: torch.device) -> None:
        for norm in (self.input_layernorm, self.post_attention_layernorm):
            norm.reset_parameters()
        self.self_attn.init_weights(buffer_device, init_std=0.02)
        self.mlp.init_weights(buffer_device)


class MiMoV2Model(nn.Module):
    """Backbone transformer for MiMoV2 (MiMo-V2.5-Pro / Teutonic-II)."""

    def __init__(
        self,
        config: MiMoV2Config,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict | None = None,
    ):
        super().__init__()
        self.config = config
        self.backend = backend

        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides.")

        # Keep gate routing in fp32 for numerical stability.
        if self.backend.gate_precision is None:
            self.backend.gate_precision = torch.float32

        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=int(config.n_routed_experts or 0),
            n_shared_experts=int(config.n_shared_experts or 0),
            shared_expert_inter_dim=config.shared_expert_inter_dim,
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=config.n_group,
            n_limited_groups=config.topk_group,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func="sigmoid_with_bias" if config.scoring_func == "sigmoid" else config.scoring_func,
            route_scale=float(config.routed_scaling_factor or 1.0),
            aux_loss_coeff=0.0,
            norm_topk_prob=config.norm_topk_prob,
            router_bias=False,
            expert_bias=False,
            expert_activation="swiglu",
            softmax_before_topk=False,
            # noaux_tc topk requires an e_score_correction_bias buffer.
            force_e_score_correction_bias=True,
            dtype=get_dtype(config.torch_dtype, torch.bfloat16),
        )
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        self.moe_config = moe_config or MoEConfig(**moe_defaults)

        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.layers = nn.ModuleDict(
            {
                str(layer_id): MiMoV2Block(layer_id, config, self.moe_config, backend)
                for layer_id in range(config.num_hidden_layers)
            }
        )
        self.norm = MiMoV2RMSNorm(config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype)

        self.rotary_emb = MiMoV2FlashRotaryEmbedding(
            rope_theta=float(config.rope_theta),
            head_dim=int(config.head_dim),
            partial_rotary_factor=float(config.partial_rotary_factor),
            dtype=dtype,
        )
        self.swa_rotary_emb = MiMoV2FlashRotaryEmbedding(
            rope_theta=float(config.swa_rope_theta),
            head_dim=int(config.swa_head_dim),
            partial_rotary_factor=float(config.partial_rotary_factor),
            dtype=dtype,
        )
        self.has_sliding_layers = any(p == 1 for p in config.hybrid_layer_pattern)

    def _build_causal_mask_mapping(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len = inputs_embeds.shape[:2]
        if isinstance(attention_mask, dict):
            full = attention_mask.get("full_attention")
            sliding = attention_mask.get("sliding_attention") or attention_mask.get("sliding_window_attention")
            return {
                "full_attention": _ensure_additive_mask(
                    full,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                    attention_mask=None,
                    sliding_window=None,
                ),
                "sliding_attention": _ensure_additive_mask(
                    sliding,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                    attention_mask=None,
                    sliding_window=self.config.sliding_window,
                ),
            }

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        full = create_causal_mask(**mask_kwargs)
        sliding = create_sliding_window_causal_mask(**mask_kwargs) if self.has_sliding_layers else None
        pad_mask = attention_mask if isinstance(attention_mask, torch.Tensor) else None
        return {
            "full_attention": _ensure_additive_mask(
                full,
                batch_size=batch_size,
                seq_len=seq_len,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                attention_mask=pad_mask,
                sliding_window=None,
            ),
            "sliding_attention": _ensure_additive_mask(
                sliding,
                batch_size=batch_size,
                seq_len=seq_len,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                attention_mask=pad_mask,
                sliding_window=self.config.sliding_window,
            ),
        }

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        padding_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if padding_mask is None and isinstance(attention_mask, torch.Tensor):
            padding_mask = _derive_padding_mask(attention_mask)

        causal_mask_mapping = self._build_causal_mask_mapping(
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        swa_position_embeddings = self.swa_rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers.values():
            layer_pos_emb = (
                swa_position_embeddings
                if decoder_layer.attention_type == "sliding_attention"
                else position_embeddings
            )
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=layer_pos_emb,
                padding_mask=padding_mask,
            )

        return self.norm(hidden_states) if self.norm is not None else hidden_states

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()
        for layer in self.layers.values():
            layer.init_weights(buffer_device)


class MiMoV2ForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Causal LM for MiMo-V2.5-Pro and Teutonic-II.

    Covers both FP8 (MiMo-V2.5-Pro) and BF16 (Teutonic-II) checkpoints via
    the MiMoV2StateDictAdapter, and supports n_shared_experts for Teutonic-II.
    """

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY

    _keep_in_fp32_modules_strict = ["mlp.gate.e_score_correction_bias", "attention_sink_bias", "rotary_emb"]
    _pp_keep_self_forward = True
    _skip_init_weights_on_load = True

    @dataclass(frozen=True)
    class ModelCapabilities:
        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = True
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls,
        config: MiMoV2Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        config = MiMoV2Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: MiMoV2Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        reject_unsupported_tie_word_embeddings(type(self), config)
        self.backend = backend or BackendConfig()
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model = MiMoV2Model(
            config,
            backend=self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=get_dtype(config.torch_dtype, torch.bfloat16),
        )
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = MiMoV2StateDictAdapter(
                self.config,
                self.model.moe_config,
                self.backend,
                dtype=get_dtype(config.torch_dtype, torch.bfloat16),
            )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        padding_mask: torch.Tensor | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )
        hidden = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **kwargs,
        )
        if self.lm_head is None:
            return CausalLMOutputWithPast(
                logits=hidden,
                hidden_states=hidden if output_hidden_states else None,
            )
        return compute_lm_head_logits(self.lm_head, hidden, logits_to_keep, output_hidden_states=output_hidden_states)

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module | None = None,
    ) -> list[list[str]]:
        """Keep the SWA rotary embedding on every PP stage."""
        text_model = text_model or self.model
        stage_modules = [list(modules) for modules in module_names_per_stage]
        if getattr(text_model, "swa_rotary_emb", None) is not None:
            fqn = f"{layers_prefix}swa_rotary_emb"
            for modules in stage_modules:
                if fqn not in modules:
                    modules.append(fqn)
        return stage_modules

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            self.model.init_weights(buffer_device)
            final_out_std = self.config.hidden_size**-0.5
            cutoff = 3
            if self.lm_head is not None:
                nn.init.trunc_normal_(
                    self.lm_head.weight,
                    mean=0.0,
                    std=final_out_std,
                    a=-cutoff * final_out_std,
                    b=cutoff * final_out_std,
                )
        if _has_dtensor_params(self):
            return
        cast_model_to_dtype(self, dtype)


ModelClass = MiMoV2ForCausalLM
