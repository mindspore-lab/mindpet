#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023, All rights reserved.
"""LoRA Cell"""
import math
from mindspore import nn
from mindspore import ops
from mindspore import Parameter
from mindspore.common.tensor import Tensor
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
try:
    from mindspore._checkparam import Validator, Rel
    INC_LEFT = Rel.INC_LEFT
except ImportError:
    import mindspore._checkparam as Validator
    INC_LEFT = Validator.INC_LEFT
from mindspore.common.initializer import initializer, HeUniform
from mindspore.nn.cell import Cell
from mindspore.ops.primitive import Primitive
from mindpet.delta.delta_constants import VALID_TENSOR_DATATYPE
from mindpet.utils.version_control import get_dropout, get_activation
from mindpet.layers.activation import LeakyReLU, LogSigmoid, LogSoftmax

def check_dense_input_shape(x, prim_name=None):
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if len(x) < 2:
        raise ValueError(f"{msg_prefix} dimension of 'x' should not be less than 2, but got {len(x)}.")


class LoRADense(nn.Dense):
    """Define a dense layer with LoRA structure.

    Attributes:
        lora_in_channels (int): The number of channels in the input space.
        lora_out_channels (int): The number of channels in the output space.
        lora_rank(int): The number of rows(columns) in LoRA matrices.
        lora_alpha(float): A constant in lora_rank.
        lora_dropout(float): The dropout rate, greater equal than 0 and less than 1.
        param_init_type(:class:`mindspore.dtype`): The type of data in initialized tensor.
        compute_dtype(:class:`mindspore.dtype`): The compute type of data.

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            lora_rank: int,
            lora_alpha: float,
            lora_dropout: float = 0.,
            lora_a_init=HeUniform(negative_slope=math.sqrt(5)),
            lora_b_init='zeros',
            param_init_type=mstype.float32,
            compute_dtype=mstype.float16,
            **kwargs
    ):
        super().__init__(in_channels, out_channels, **kwargs)

        # Check params
        self._check_num(lora_rank, lora_alpha, lora_dropout)
        self._check_init(lora_a_init, lora_b_init, lora_rank)
        self._check_type_of_data(param_init_type, compute_dtype)

        # Define and initialize params
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = get_dropout(lora_dropout)
        self.mindpet_delta_lora_a = Parameter(
            initializer(lora_a_init, [lora_rank, in_channels], param_init_type),
            name='mindpet_delta_lora_A')
        self.mindpet_delta_lora_b = Parameter(initializer(lora_b_init, [out_channels, lora_rank], param_init_type),
                                         name='mindpet_delta_lora_B')
        self.scaling = self.lora_alpha / self.lora_rank

        # Calculation utils
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.cast = ops.Cast()
        self.dtype = compute_dtype
        self.lora_a_matmul = P.MatMul(transpose_b=True)
        self.lora_b_matmul = P.MatMul(transpose_b=True)

        activation = kwargs.pop("activation", None)
        self.activation = get_activation(activation) if isinstance(
            activation, str) else activation
        if activation is not None and not isinstance(self.activation, (Cell, Primitive)):
            raise TypeError(f"For '{self.cls_name}', the 'activation' must be str or Cell or Primitive, but got "
                            f"{type(activation).__name__}.")
        self.activation_flag = self.activation is not None

    def construct(self, input_tensor):
        """Foward"""
        # Data type operation
        ori_dtype = F.dtype(input_tensor)
        input_tensor = self.cast(input_tensor, self.dtype)
        weight = self.cast(self.weight, self.dtype)
        lora_a = self.cast(self.mindpet_delta_lora_a, self.dtype)
        lora_b = self.cast(self.mindpet_delta_lora_b, self.dtype)
        scaling = self.cast(self.scaling, self.dtype)

        # Shape operations
        x_shape = self.shape_op(input_tensor)
        check_dense_input_shape(x_shape, self.cls_name)
        input_tensor = self.reshape(input_tensor, (-1, x_shape[-1]))

        # Dense result
        dense_result = self.matmul(input_tensor, weight)
        if self.has_bias:
            bias = self.cast(self.bias, self.dtype)
            dense_result = self.bias_add(dense_result, bias)

        # LoRA result
        input_tensor = self.lora_dropout(input_tensor)
        input_tensor = self.lora_a_matmul(input_tensor, lora_a)
        input_tensor = self.lora_b_matmul(input_tensor, lora_b)
        input_tensor = self.mul(input_tensor, scaling)

        # Result addition and activation
        dense_result = self.add(dense_result, input_tensor)
        if self.activation_flag:
            dense_result = self.activation(dense_result)

        # Shape restore
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (-1,)
            dense_result = self.reshape(dense_result, out_shape)
        dense_result = self.cast(dense_result, ori_dtype)

        return dense_result

    def shard(self, strategy_org_dense_matmul=None,
              strategy_org_bias_add=None,
              strategy_lora_dropout=None,
              strategy_lora_a_matmul=None,
              strategy_lora_b_matmul=None,
              strategy_lora_mul=None,
              strategy_lora_add=None,
              strategy_activation=None):
        r"""
        Set the shard for the LoraDense. All the strategy size should be equal to the inputs.

        Note:
            It is valid only in semi auto parallel or auto parallel mode.
            In other parallel modes, strategies set here will be ignored.

        Args:
            strategy_org_dense_matmul (tuple): The strategy for the origin dense matmul.
            strategy_org_bias_add (tuple): The strategy for the origin dense bias_add.
            strategy_lora_dropout (tuple): The strategy for the LoraDense dropout.
            strategy_lora_a_matmul (tuple): The strategy for the LoraDense lora_a matmul.
            strategy_lora_b_matmul (tuple): The strategy for the LoraDense lora_b matmul.
            strategy_lora_mul (tuple): The strategy for the LoraDense mul.
            strategy_lora_add (tuple): The strategy for the LoraDense add.
            strategy_activation (tuple): The strategy for the strategy_activation.
        """
        try:
            self.matmul.shard(strategy_org_dense_matmul)
            if self.has_bias:
                self.bias_add.shard(strategy_org_bias_add)
            self.lora_dropout.dropout.shard(strategy_lora_dropout)
            self.lora_a_matmul.shard(strategy_lora_a_matmul)
            self.lora_b_matmul.shard(strategy_lora_b_matmul)
            self.mul.shard(strategy_lora_mul)
            self.add.shard(strategy_lora_add)

            if self.activation_flag:
                if isinstance(self.activation, LeakyReLU):
                    self.activation.select_op.shard((strategy_activation[0], strategy_activation[0]))
                elif isinstance(self.activation, LogSigmoid):
                    self.activation.mul.shard((strategy_activation[0], ()))
                    self.activation.exp.shard(strategy_activation)
                    self.activation.add.shard((strategy_activation[0], ()))
                    self.activation.rec.shard(strategy_activation)
                    self.activation.log.shard(strategy_activation)
                elif isinstance(self.activation, LogSoftmax):
                    raise ValueError("The 'LogSoftmax' function is not supported in semi auto parallel "
                                     "or auto parallel mode.")
                else:
                    getattr(self.activation, self.act_name).shard(strategy_activation)
        except Exception as ex:
            # pylint: disable=W0719
            raise Exception(f"Exception occurred when set the shard for LoRADense, error message: {str(ex)}") from ex

    def _check_num(self, lora_rank, lora_alpha, lora_dropout):
        Validator.check_positive_int(arg_value=lora_rank, arg_name="lora_rank", prim_name=self.cls_name)

        Validator.check_value_type(arg_value=lora_alpha, arg_name='lora_alpha',
                                   valid_types=[int, float], prim_name=self.cls_name)
        if lora_alpha == 0:
            raise ValueError(f"For {self.cls_name}, the 'lora_alpha' cannot be zero, "
                             f"but got '{lora_alpha}' with type {type(lora_alpha)}.")

        Validator.check_float_range(arg_value=lora_dropout, lower_limit=0.0, upper_limit=1.0,
                                    rel=INC_LEFT, arg_name='lora_dropout', prim_name=self.cls_name)

    def _check_init(self, lora_a_init, lora_b_init, lora_rank):
        """check init arguments"""
        if isinstance(lora_a_init, Tensor):
            if lora_a_init.ndim != 2 or lora_a_init.shape[0] != lora_rank or \
                    lora_a_init.shape[1] != self.in_channels:
                raise ValueError(f"For '{self.cls_name}', [lora_a_init] shape error. "
                                 f"The ndim of [lora_a_init] must be equal to 2, "
                                 f"the first dim must be equal to 'lora_rank', "
                                 f"and the second dim must be equal to 'in_channels'. ")
        if isinstance(lora_b_init, Tensor):
            if lora_b_init.ndim != 2 or lora_b_init.shape[0] != self.out_channels or \
                    lora_b_init.shape[1] != lora_rank:
                raise ValueError(f"For '{self.cls_name}', [lora_b_init] shape error. "
                                 f"The ndim of [lora_b_init] must be equal to 2, "
                                 f"the first dim must be equal to 'out_channels', "
                                 f"and the second dim must be equal to 'lora_rank'. ")

    def _check_type_of_data(self, param_init_type, compute_dtype):
        if param_init_type not in VALID_TENSOR_DATATYPE:
            raise TypeError(f"For {self.cls_name}, the 'param_init_type' must be mindspore.dtype.float16 or "
                            f"mindspore.dtype.float32, but got {param_init_type}.")
        if compute_dtype not in VALID_TENSOR_DATATYPE:
            raise TypeError(f"For {self.cls_name}, the 'compute_dtype' must be mindspore.dtype.float16 or "
                            f"mindspore.dtype.float32, but got {compute_dtype}.")
