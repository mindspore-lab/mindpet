#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2010-2022. All rights reserved.
"""Prefix Layer Cell"""
import mindspore as ms
from mindspore import nn
try:
    from mindspore._checkparam import Validator, Rel
    INC_LEFT = Rel.INC_LEFT
except ImportError:
    import mindspore._checkparam as Validator
    INC_LEFT = Validator.INC_LEFT
from mindpet.utils.version_control import get_dropout


def check_multiple(param_dividend, value_dividend, param_divisor, value_divisor):
    if value_dividend % value_divisor != 0:
        raise ValueError(f"param {param_dividend} must be a multiple of {param_divisor}")
    return value_dividend / value_divisor


class PrefixLayer(nn.Cell):
    """A layer of prefix tuning module."""
    def __init__(self,
                 prefix_token_num,
                 batch_size,
                 num_heads,
                 hidden_dim,
                 embed_dim,
                 mid_dim=512,
                 dropout_rate=0.1):
        """

        Args:
            prefix_token_num: prefix的长度
            batch_size: 单次传递给程序用以训练的数据个数，与原模型参数一致
            num_heads: 多头注意力头数，与原模型参数一致
            hidden_dim: 模型隐藏维度， 与原模型参数一致
            embed_dim: embedding的维度， 与原模型参数一致
            mid_dim: prefix中间维度
            dropout_rate: 丢弃率
        """
        super().__init__()
        self.prefix_token_num = Validator.check_positive_int(prefix_token_num, int, "prefix_token_num")
        self.batch_size = Validator.check_positive_int(batch_size, int, "batch_size")
        self.num_heads = Validator.check_positive_int(num_heads, int, "num_heads")
        self.hidden_dim = Validator.check_positive_int(hidden_dim, int, "hidden_dim")
        self.embed_dim = Validator.check_positive_int(embed_dim, int, "embed_dim")
        self.mid_dim = Validator.check_positive_int(mid_dim, int, "mid_dim")
        self.dropout_rate = Validator.check_float_range(dropout_rate, 0.0, 1.0, INC_LEFT)
        try:
            check_multiple("prefix_token_num", prefix_token_num, "batch_size", batch_size)
        except ValueError as ex:
            raise ValueError(f"Invalid param [prefix_token_num] when initializing"
                             f"PrefixLayer, error message:{str(ex)}") from ex
        self.dropout = get_dropout(dropout_rate)
        self.past_value_reparam = None
        self.past_key_reparam = None
        self.__define_network()
        self.__allocate_parameter()

    def __define_network(self) -> None:
        """the network structure of prefix"""
        self.input_tokens = ms.Parameter(ms.numpy.arange(0, self.prefix_token_num, 1),
                                                name="mindpet_delta_prefixtuning_input_tokens", requires_grad=False)
        self.mindpet_delta_prefixtuning_wte = nn.Embedding(self.prefix_token_num, self.embed_dim)
        self.mindpet_delta_prefixtuning_control_trans = nn.SequentialCell(
            nn.Dense(self.embed_dim, self.mid_dim),
            nn.Tanh(),
            nn.Dense(self.mid_dim, self.hidden_dim)
        )

    def __allocate_parameter(self):
        """the value of prefix matrix"""
        input_tokens = self.input_tokens
        temp_control = self.mindpet_delta_prefixtuning_wte(input_tokens)
        past_key_values = self.mindpet_delta_prefixtuning_control_trans(temp_control)
        seq_len, _ = past_key_values.shape
        past_key_values = past_key_values.view(seq_len, -1, self.hidden_dim)
        past_key_values = self.dropout(past_key_values)
        past_key_values = ms.ops.transpose(past_key_values, (1, 0, 2)).split(2)
        self.past_key_reparam = past_key_values[0][0]
        self.past_value_reparam = past_key_values[0][0]
