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
import numpy as np
from mindspore import dtype as mstype
from mindspore import nn
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P

try:
    from mindspore._checkparam import Rel, Validator

    INC_LEFT = Rel.INC_LEFT
except ImportError:
    import mindspore._checkparam as Validator

    INC_LEFT = Validator.INC_LEFT

from mindpet.utils.version_control import get_dropout


class PrefixEmbedding(nn.Cell):
    """
    The embedding with parallel_config.
    """

    def __init__(self, vocab_size, embedding_size, data_parallel=1, model_parallel=1, vocab_emb_dp=True,
                 param_init='normal'):
        """
        vocab_size: vocab size
        embedding_size: embedding size
        data_parallel: data parallel config
        model_parallel: data parallel config
        vocab_emb_dp: embedding dp config
        param_init: parameter init method
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding_table = ms.Parameter(initializer(param_init, [self.vocab_size, self.embedding_size]),
                                            name='embedding_table', parallel_optimizer=False)
        if vocab_emb_dp:
            self.gather = P.Gather().shard(((1, 1), (data_parallel, 1)))
        else:
            if self.vocab_size % model_parallel != 0:
                raise ValueError(f"The vocab size of the embedding {self.vocab_size} must be a "
                                 f"multiple of model_parallel {model_parallel}.")
            self.gather = P.Gather().shard(((model_parallel, 1), (data_parallel, 1)))

    def construct(self, input_ids):
        """
        embedding inputs
        """
        output = self.gather(self.embedding_table, input_ids, 0)
        return output


class PrefixEncoder(nn.Cell):
    """
    The cell to encode the prefix
    Input : batch_size
    Output shape: layers * (2, bs, num_heads, pre_len, kv_channels)
    """

    def __init__(
            self,
            pre_seq_len,
            num_layers,
            num_heads,
            kv_channels,
            prefix_projection,
            projection_dim,
            dropout_prob,
            parallel_config=None,
            out_perm=(2, 0, 3, 1, 4),
    ):
        """
        Args:
            pre_seq_len: prefix的序列长度
            num_layers: 原模型transformer层数
            num_heads: 原模型transformer多头注意力头数
            kv_channels: 原模型transformer kv维度
            prefix_projection 是否使用MLP表征
            projection_dim: MLP维度
            dropout_prob: 丢弃率
            parallel_config: 并行参数
            out_perm: 输出的维度
        """
        super().__init__()
        self.pre_seq_len = Validator.check_positive_int(pre_seq_len, "pre_seq_len")
        self.num_layers = Validator.check_positive_int(num_layers, "num_layers")
        self.num_heads = Validator.check_positive_int(num_heads, "num_heads")
        self.kv_channels = Validator.check_positive_int(kv_channels, "kv_channels")

        dropout_prob = Validator.check_float_range(dropout_prob, 0.0, 1.0, INC_LEFT)
        self.dropout = get_dropout(dropout_prob)

        self.prefix_projection = prefix_projection

        self.mindpet_delta_ptuning2_prefix = ms.Parameter(
            np.arange(self.pre_seq_len), requires_grad=False
        )

        out_embed_dim = self.num_layers * self.kv_channels * self.num_heads * 2
        if parallel_config:
            data_parallel = Validator.check_positive_int(parallel_config.data_parallel, "data_parallel")
            model_parallel = Validator.check_positive_int(parallel_config.model_parallel, "model_parallel")
            vocab_emb_dp = parallel_config.vocab_emb_dp
        else:
            data_parallel = 1
            model_parallel = 1
            vocab_emb_dp = True

        self.mindpet_delta_ptuning2_embedding = PrefixEmbedding(vocab_size=self.pre_seq_len,
                                                                embedding_size=out_embed_dim,
                                                                data_parallel=data_parallel,
                                                                model_parallel=model_parallel,
                                                                vocab_emb_dp=vocab_emb_dp)

        if self.prefix_projection:
            self.projection_dim = Validator.check_positive_int(
                projection_dim, "projection_dim"
            )
            # two-layer MLP to encode the prefix
            self.mindpet_delta_ptuning2_dense_in = nn.Dense(out_embed_dim, self.projection_dim)
            self.mindpet_delta_ptuning2_tanh = nn.Tanh()
            self.mindpet_delta_ptuning2_dense_out = nn.Dense(self.projection_dim, out_embed_dim)
            self.mindpet_delta_ptuning2_trans = nn.SequentialCell(
                self.mindpet_delta_ptuning2_dense_in,
                self.mindpet_delta_ptuning2_tanh,
                self.mindpet_delta_ptuning2_dense_out,
            )

        self.out_perm = out_perm
        self.expand_dims = P.ExpandDims().shard(((1,),))
        self.tile = P.Tile().shard(((1, 1),))
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.spilt_layers = P.Split(axis=0, output_num=self.num_layers)

    def construct(self, batch_size, dtype=mstype.half):
        """
        new prefix
        """
        prefix_tokens = self.expand_dims(self.mindpet_delta_ptuning2_prefix, 0)
        prefix_tokens = self.tile(prefix_tokens, (batch_size, 1))

        # (bs, pre_len) -> (bs, pre_len, 2 * layers * num_heads * kv_channels)
        past_key_values = self.mindpet_delta_ptuning2_embedding(prefix_tokens)

        if self.prefix_projection:
            past_key_values = self.mindpet_delta_ptuning2_trans(past_key_values)

        past_key_values = self.cast(past_key_values, dtype)

        # (bs, pre_len, 2 * layers * num_heads * kv_channels) -> (bs, pre_len, 2 * layers, num_heads, kv_channels)
        past_key_values = self.reshape(past_key_values, (batch_size,
                                                         self.pre_seq_len,
                                                         self.num_layers * 2,
                                                         self.num_heads,
                                                         self.kv_channels,))

        past_key_values = self.dropout(past_key_values)

        # (bs, pre_len, 2 * layers, num_heads, kv_channels) -> (2 * layers, bs, num_heads, pre_len, kv_channels)
        past_key_values = self.transpose(past_key_values, self.out_perm)

        # (2 * layers, bs, num_heads, pre_len, kv_channels) -> , layers * (2, bs, num_heads, pre_len, kv_channels)
        past_key_values = self.spilt_layers(past_key_values)

        return past_key_values

    def shard(self, data_parallel, model_parallel):
        """
        set shard strategy
        """

        if self.prefix_projection:
            # (bs, pre_len, embedding_dim)
            self.mindpet_delta_ptuning2_dense_in.matmul.shard(((data_parallel, 1), (model_parallel, 1)))
            self.mindpet_delta_ptuning2_dense_in.bias_add.shard(((data_parallel, 1), (1,)))
            self.mindpet_delta_ptuning2_tanh.tanh.shard(((data_parallel, 1, 1),))
            self.mindpet_delta_ptuning2_dense_out.matmul.shard(((data_parallel, 1), (model_parallel, 1)))
            self.mindpet_delta_ptuning2_dense_out.bias_add.shard(((data_parallel, 1), (1,)))

        # (bs, pre_len, 2 * layers * num_heads * kv_channels)
        self.cast.shard(((data_parallel, 1, 1),))  # (dp, 1, 1)

        # (bs, pre_len, 2 * layers, num_heads, kv_channels)
        self.dropout.dropout.shard(((data_parallel, 1, 1, 1, 1),))  # (dp, 1, 1, 1, 1)
        self.transpose.shard(((data_parallel, 1, 1, 1, 1),))  # (dp, 1, 1, 1, 1)

        # (2 * layers, bs, num_heads, pre_len, kv_channels)
        self.spilt_layers.shard(((1, data_parallel, 1, 1, 1),))  # (1, dp, 1, 1, 1)
