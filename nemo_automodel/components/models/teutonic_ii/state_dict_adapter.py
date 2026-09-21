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

from nemo_automodel.components.models.mimo_v25.state_dict_adapter import MiMoV2StateDictAdapter


class TeutonicIIStateDictAdapter(MiMoV2StateDictAdapter):
    """HF checkpoint adapter for Teutonic-II.

    Teutonic-II uses the same HF weight layout as MiMo-V2.5-Pro for the pieces
    Automodel consumes, including routed/shared experts and fused-QKV FP8
    dequantization.
    """


__all__ = ["TeutonicIIStateDictAdapter"]