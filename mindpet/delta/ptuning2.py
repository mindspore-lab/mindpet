# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
p-tuning-v2
Reference: https://arxiv.org/pdf/2110.07602.pdf
"""

import mindspore as ms
import mindspore.nn as nn
import numpy as np

from mindspore import dtype as mstype
from mindspore.ops import operations as P

from mindpet.utils.version_control import get_dropout

try:
    from mindspore._checkparam import Validator, Rel

    INC_LEFT = Rel.INC_LEFT
except:
    import mindspore._checkparam as Validator

    INC_LEFT = Validator.INC_LEFT


class PrefixEncoder(nn.Cell):
    """
    The cell to encode the prefix
    Input : batch_size
    Output shape: layers * (2, bs, num_heads, pre_len, kv_channels)
    """

    def __init__(self, pre_seq_len, num_layers, num_heads, kv_channels, prefix_projection,
                 projection_dim, dropout_prob):
        """
        Args:
            pre_seq_len: prefix的序列长度
            num_layers: 原模型transformer层数
            num_heads: 原模型transformer多头注意力头数
            kv_channels: 原模型transformer kv维度
            prefix_projection 是否使用MLP表征
            projection_dim: MLP维度
            dropout_prob: 丢弃率
        """
        super().__init__()
        self.pre_seq_len = Validator.check_positive_int(pre_seq_len, "pre_seq_len")
        self.num_layers = Validator.check_positive_int(num_layers, "num_layers")
        self.num_heads = Validator.check_positive_int(num_heads, "num_heads")
        self.kv_channels = Validator.check_positive_int(kv_channels, "kv_channels")

        dropout_prob = Validator.check_float_range(dropout_prob, 0.0, 1.0, INC_LEFT)
        self.dropout = get_dropout(dropout_prob)

        self.prefix_projection = prefix_projection

        self.tk_delta_ptuning2_prefix = ms.Parameter(np.arange(self.pre_seq_len),
                                                     requires_grad=False)

        out_embed_dim = self.num_layers * self.kv_channels * self.num_heads * 2
        self.tk_delta_ptuning2_embedding = nn.Embedding(self.pre_seq_len, out_embed_dim)

        if self.prefix_projection:
            self.projection_dim = Validator.check_positive_int(projection_dim, "projection_dim")
            # two-layer MLP to encode the prefix
            self.tk_delta_ptuning2_trans = nn.SequentialCell(
                nn.Dense(out_embed_dim, self.projection_dim),
                nn.Tanh(),
                nn.Dense(self.projection_dim, out_embed_dim)
            )

        self.expand_dims = P.ExpandDims()
        self.tile = P.Tile()
        self.cast = P.Cast()

    def construct(self, batch_size, dtype=mstype.half):
        prefix_tokens = self.expand_dims(self.tk_delta_ptuning2_prefix, 0)
        prefix_tokens = self.tile(prefix_tokens, (batch_size, 1))

        # (bs, pre_len) -> (bs, pre_len, 2 * layers * num_heads * kv_channels)
        past_key_values = self.tk_delta_ptuning2_embedding(prefix_tokens)

        if self.prefix_projection:
            past_key_values = self.tk_delta_ptuning2_trans(past_key_values)

        past_key_values = self.cast(past_key_values, dtype)

        # (bs, pre_len, 2 * layers * num_heads * kv_channels) -> (bs, pre_len, 2 * layers, num_heads, kv_channels)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.num_layers * 2,
            self.num_heads,
            self.kv_channels
        )

        past_key_values = self.dropout(past_key_values)

        # (bs, pre_len, 2 * layers, num_heads, kv_channels) -> layers * (2, bs, num_heads, pre_len, kv_channels)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return past_key_values
