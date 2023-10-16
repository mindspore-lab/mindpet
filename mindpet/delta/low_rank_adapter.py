#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023, All rights reserved.
"""Low Rank Adapter Cell"""
from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer, Initializer
from mindspore.common.parameter import Parameter
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
try:
    from mindspore.nn.transformer.layers import _args_type_validator_check, _valid_value_checks
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
    from mindformers.modules.layers import _args_type_validator_check, _valid_value_checks
from mindpet.delta.delta_constants import VALID_TENSOR_DATATYPE
from mindpet.utils.version_control import get_activation, _activation

class LowRankLinear(nn.Cell):
    """
    定义微调算法LowRankLinear层, 其主要结构包含两个compactor。为LowRankAdapterLayer所调用。

    Args:
        in_channels (int): LowRankLinear输入Tensor的空间维度
        out_channels (int): LowRankLinear输出Tensor的空间维度
        rank: LowRankLinear的隐藏size
        weight_init (Union[str, Initializer]):
            权重参数的初始化方法。它的类型可以是str或Initializer。
            当使用str时，值引用自类initializer；更多细节请参考Initializer的值。
            默认值："normal"。
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): 偏置参数的初始化方法。
            它的类型可以是Tensor，str，Initializer或numbers.Number。
            当使用str时，值引用自类initializer；更多细节请参考Initializer的值。
            当使用Tensor时，数据类型与输入Tensor相同。
            默认值："zeros"
        has_bias (int): LowRankLinear的权重矩阵是否有偏置
        param_init_type (dtype.Number): 表示dense中模块的参数初始化类型
        compute_dtype (dtype.Number): 表示dense中矩阵乘法的计算类型
    """

    @_args_type_validator_check(in_channels=Validator.check_positive_int,
                                out_channels=Validator.check_positive_int,
                                rank=Validator.check_positive_int,
                                has_bias=Validator.check_bool,
                                param_init_type=_valid_value_checks(VALID_TENSOR_DATATYPE,
                                                                    "LowRankLinear"),
                                compute_dtype=_valid_value_checks(VALID_TENSOR_DATATYPE,
                                                                  "LowRankLinear"))
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 rank: int = 1,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias: bool = True,
                 param_init_type: mstype = mstype.float32,
                 compute_dtype: mstype = mstype.float16):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.weight_init = weight_init
        self.mindpet_delta_low_rank_adapter_weight_left = \
            Parameter(initializer(self.weight_init, [in_channels, rank], param_init_type),
                      name="mindpet_delta_low_rank_adapter_weight_left")
        self.mindpet_delta_low_rank_adapter_weight_right = \
            Parameter(initializer(self.weight_init, [rank, out_channels], param_init_type),
                      name="mindpet_delta_low_rank_adapter_weight_right")
        self.has_bias = has_bias

        self.bias = None
        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError(f"For '{self.cls_name}', bias init shape error. The ndim of 'bias_init' must "
                                     f"be equal to 1, and the first dim must be equal to 'out_channels'. But got "
                                     f"'bias_init': {bias_init}, 'out_channels': {out_channels}.")
            self.bias = Parameter(initializer(
                bias_init, [out_channels], param_init_type), name="bias")
            self.bias_add = P.BiasAdd()

        self.compt_dtype = compute_dtype
        self.matmul_weight = P.MatMul(transpose_b=False)
        self.matmul_input = P.MatMul(transpose_b=False)
        self.reshape = P.Reshape()
        self.shape_op = P.Shape()

    def construct(self, input_tensor):
        """Forward"""
        # get input_x info
        x_shape = self.shape_op(input_tensor)
        ori_dtype = F.dtype(input_tensor)

        # reshape input_tensor to compute
        input_tensor = self.reshape(input_tensor, (-1, x_shape[-1]))

        # compute weight
        weight = self.matmul_weight(self.cast(self.mindpet_delta_low_rank_adapter_weight_left, self.compt_dtype),
                                    self.cast(self.mindpet_delta_low_rank_adapter_weight_right, self.compt_dtype))

        input_tensor = self.cast(input_tensor, self.compt_dtype)
        input_tensor = self.matmul_input(input_tensor, weight)

        if self.has_bias:
            input_tensor = self.bias_add(
                input_tensor, self.cast(self.bias, self.compt_dtype))

        out_shape = x_shape[:-1] + (-1,)
        input_tensor = self.reshape(input_tensor, out_shape)

        output = self.cast(input_tensor, ori_dtype)
        return output

    def shard(self, strategy_matmul_weight, strategy_matmul_input, strategy_bias=None):
        self.matmul_weight.shard(strategy_matmul_weight)
        self.matmul_input.shard(strategy_matmul_input)
        if self.has_bias:
            self.bias_add.shard(strategy_bias)
        return self


class LowRankAdapterLayer(nn.Cell):
    """
    定义微调算法LowRankAdapter模块结构, 初始化LowRankAdapter层的参数, 包括矩阵参数、激活层等。

    Args:
        hidden_size (int): 隐藏层输出的特征向量维度，亦是LowRankAdapter模块输入的特征向量维度。
        reduction_factor (int): Low-Rank Adapter结构向下投影的维度值缩减倍数。
            如计算得bottleneck_dim = hidden_size//reduction_factor，
            bottleneck_dim即为向下投影的维度值。
            默认值为1。
        low_rank_size (int): LowRankAdapter模块中'内层'bottleneck的隐藏大小。
        low_rank_w_init (Union[str, Initializer, None]):
            Low-Rank Adapter结构向下或向上投影的权重参数的初始化方法。它可以是str，Initializer。
            当使用str时，值引用自类initializer；更多细节请参考Initializer的值。
            默认值："xavier_uniform"。
        non_linearity (str): Low-Rank Adapter结构中投影后所使用的激活函数。
            str的值引用自mindspore中 `get_activation` 方法所支持的激活函数类型。
            默认为 'gelu'
        param_init_type (dtype.Number): 表示Low-Rank Adapter结构的参数初始化类型。
            其值应为dtype.float32或dtype.float16。
            默认值：dtype.float32。
        compute_dtype (dtype.Number): 表示Low-Rank Adapter结构的矩阵乘法的计算类型。
            其值应为dtype.float32或dtype.float16。
            默认值：dtype.float16。

    Inputs:
        input_tensor (Tensor): 网络的所有输入组成的元组，shape为(*, in_channels)的Tensor，in_channels与入参中的hidden_size类型一致。
    Outputs:
        output (Tensor): LowRankAdapter dense的计算结果，shape为 (*, out_channels)，out_channels与入参中的hidden_size类型一致。
    """

    @_args_type_validator_check(hidden_size=Validator.check_positive_int,
                                reduction_factor=Validator.check_positive_int,
                                low_rank_size=Validator.check_positive_int,
                                non_linearity=_valid_value_checks(list(_activation.keys()),
                                                                  "LowRankAdapterLayer"),
                                param_init_type=_valid_value_checks(VALID_TENSOR_DATATYPE,
                                                                    "LowRankAdapterLayer"),
                                compute_dtype=_valid_value_checks(VALID_TENSOR_DATATYPE,
                                                                  "LowRankAdapterLayer"))
    def __init__(
            self,
            hidden_size: int,
            reduction_factor: int,
            low_rank_size: int = 1,
            low_rank_w_init="xavier_uniform",
            non_linearity: str = "gelu",
            param_init_type: mstype = mstype.float32,
            compute_dtype: mstype = mstype.float16):
        self._check_low_rank_w_init(low_rank_w_init)
        self._check_reduction_factor(hidden_size, reduction_factor)
        super().__init__()

        self.bottleneck_size = hidden_size // reduction_factor
        self.non_linearity = non_linearity

        self.mindpet_delta_low_rank_adapter_down_sampler = LowRankLinear(in_channels=hidden_size,
                                                                    out_channels=self.bottleneck_size,
                                                                    rank=low_rank_size,
                                                                    weight_init=low_rank_w_init,
                                                                    param_init_type=param_init_type,
                                                                    compute_dtype=compute_dtype)
        self.mindpet_delta_low_rank_adapter_non_linear = get_activation(
            non_linearity)
        self.mindpet_delta_low_rank_adapter_up_sampler = LowRankLinear(in_channels=self.bottleneck_size,
                                                                  out_channels=hidden_size,
                                                                  rank=low_rank_size,
                                                                  weight_init=low_rank_w_init,
                                                                  param_init_type=param_init_type,
                                                                  compute_dtype=compute_dtype)
        self.residual_add = P.Add()

    def construct(self, input_tensor):
        """Forward"""
        # get input_x info
        x_shape = P.Shape()(input_tensor)
        ori_dtype = F.dtype(input_tensor)

        # reshape input_tensor to compute
        input_tensor = P.Reshape()(input_tensor, (-1, x_shape[-1]))

        # calculate adapter_out
        adapter_down_sampler_output = self.mindpet_delta_low_rank_adapter_down_sampler(
            input_tensor)
        adapter_non_linear_output = self.mindpet_delta_low_rank_adapter_non_linear(
            adapter_down_sampler_output)
        adapter_output = self.mindpet_delta_low_rank_adapter_up_sampler(
            adapter_non_linear_output)

        # residual connection, add input and adapter_output
        input_tensor = self.residual_add(input_tensor, adapter_output)

        # recover the previous outshape and dtype
        out_shape = x_shape[:-1] + (-1,)
        input_tensor = P.Reshape()(input_tensor, out_shape)
        output = P.Cast()(input_tensor, ori_dtype)
        return output

    def shard(self, strategy_matmul_down_sampler_weight=None,
              strategy_matmul_down_sampler_input=None,
              strategy_bias_down_sampler=None,
              strategy_non_linearity=None,
              strategy_matmul_up_sampler_weight=None,
              strategy_matmul_up_sampler_input=None,
              strategy_bias_up_sampler=None,
              strategy_residual_add=None):
        """
        Set the shard for the LowRankAdapterLayer. All the strategy size should be equal to the inputs.

        Note:
            It is valid only in semi auto parallel or auto parallel mode.
            In other parallel modes, strategies set here will be ignored.

        Args:
            strategy_matmul_down_sampler_weight (tuple): The strategy for the
                LowRankAdapter-down_sampler-matmul_weight matmul.
            strategy_matmul_down_sampler_input (tuple): The strategy for the
                LowRankAdapter-down_sampler-matmul_input matmul.
            strategy_bias_down_sampler (tuple): The strategy for the
                LowRankAdapter-down_sampler-bias_add.
            strategy_non_linearity (tuple): The strategy for the LowRankAdapter non_linearity.
            strategy_matmul_up_sampler_weight (tuple): The strategy for the
                LowRankAdapter-up_sampler-matmul_weight matmul.
            strategy_matmul_up_sampler_input (tuple): The strategy for the
                LowRankAdapter-up_sampler-matmul_input matmul.
            strategy_bias_up_sampler (tuple): The strategy for the
                LowRankAdapter-up_sampler-bias_add.
            strategy_residual_add (tuple): The strategy for the residual_add.
        """
        try:
            self.mindpet_delta_low_rank_adapter_down_sampler.shard(
                strategy_matmul_down_sampler_weight, strategy_matmul_down_sampler_input, strategy_bias_down_sampler)
            self.mindpet_delta_low_rank_adapter_up_sampler.shard(
                strategy_matmul_up_sampler_weight, strategy_matmul_up_sampler_input, strategy_bias_up_sampler)
            # some operations has many primitives, need to manually set the shard
            if self.non_linearity.lower() == "leakyrelu":
                self.mindpet_delta_low_rank_adapter_non_linear.select_op.shard(
                    (strategy_non_linearity[0], strategy_non_linearity[0]))
            elif self.non_linearity.lower() == "logsigmoid":
                self.mindpet_delta_low_rank_adapter_non_linear.mul.shard(
                    (strategy_non_linearity[0], ()))
                self.mindpet_delta_low_rank_adapter_non_linear.exp.shard(
                    strategy_non_linearity)
                self.mindpet_delta_low_rank_adapter_non_linear.add.shard(
                    (strategy_non_linearity[0], ()))
                self.mindpet_delta_low_rank_adapter_non_linear.rec.shard(
                    strategy_non_linearity)
                self.mindpet_delta_low_rank_adapter_non_linear.log.shard(
                    strategy_non_linearity)
            elif self.non_linearity.lower() == "logsoftmax":
                raise ValueError("The 'LogSoftmax' function is not supported in semi auto parallel "
                                 "or auto parallel mode.")
            else:
                getattr(self.mindpet_delta_low_rank_adapter_non_linear,
                        self.non_linearity).shard(strategy_non_linearity)
            self.residual_add.shard(strategy_residual_add)

        except Exception as ex:
            # pylint: disable=W0719
            raise Exception(
                f"Exception occurred when set the shard for LowRankAdapterLayer, error message: {str(ex)}") from ex

        return self

    def _check_low_rank_w_init(self, low_rank_w_init):
        if not isinstance(low_rank_w_init, (str, Initializer)):
            raise TypeError(f"For '{self.cls_name}', the type of the 'low_rank_w_init' argument should be 'string' "
                            f"or 'initializer', but got {type(low_rank_w_init)}.")

    def _check_reduction_factor(self, hidden_size, reduction_factor):
        if reduction_factor > hidden_size:
            raise ValueError(f"For '{self.cls_name}', the value of the 'reduction_factor' argument should be smaller "
                             f"than 'hidden_size', but got reduction_factor: {reduction_factor}, "
                             f"hidden_size: {hidden_size}.")


class LowRankAdapterDense(nn.Dense):
    """
    定义微调算法LowRankAdapter dense层，继承nn.Dense。

    Args:
        in_channels (int): LowRankAdapterDense层输入Tensor的空间维度。
        out_channels (int): LowRankAdapterDense层输出Tensor的空间维度。
        weight_init (Union[Tensor, str, Initializer, numbers.Number]):
            LowRankAdapterDense中全连接层权重参数的初始化方法。
            它的类型可以是Tensor，str，Initializer或numbers.Number。
            当使用str时，值引用自类initializer；更多细节请参考Initializer的值。
            当使用Tensor时，数据类型与输入Tensor相同。
            默认值："normal"。
        bias_init (Union[Tensor, str, Initializer, numbers.Number]):
            LowRankAdapterDense中全连接层偏置参数的初始化方法。
            它的类型可以是Tensor，str，Initializer或numbers.Number。
            当使用str时，值引用自类initializer；更多细节请参考Initializer的值。
            当使用Tensor时，数据类型与输入Tensor相同。
            默认值："zeros"。
        has_bias (bool): 是否有偏置
        activation (Union[str, Cell, Primitive, None]): 激活函数，是与创建层的输入具有相同数据类型的权重矩阵
        reduction_factor (int): bottleneck的瓶颈倍数，bottleneck_dim = hidden_size//reduction_factor
        low_rank_size (int): LowRankAdapter模块中内层bottleneck的隐藏大小
        low_rank_w_init (Union[str, Initializer, None]):
            Low-Rank Adapter结构向下或向上投影的权重参数的初始化方法。它可以是str或Initializer。
            当使用str时，值引用自类initializer；更多细节请参考Initializer的值。
            默认值："xavier_uniform"。
        non_linearity (str): Low-Rank Adapter结构中投影后所使用的激活函数。
            str的值引用自mindspore中 `get_activation` 方法所支持的激活函数类型。
            默认为 'gelu'。
        param_init_type (dtype.Number): 表示dense中Low-Rank Adapter结构的参数初始化类型。
            其值应为dtype.float32或dtype.float16。
            默认值：dtype.float32。
        compute_dtype (dtype.Number): 表示dense中Low-Rank Adapter结构的矩阵乘法的计算类型。
            其值应为dtype.float32或dtype.float16。
            默认值：dtype.float16。

    Inputs:
        input_tensor (Tensor): 网络的所有输入组成的元组，shape为(*, in_channels)的Tensor，in_channels与入参中的in_channels类型一致。
    Outputs:
        output (Tensor): LowRankAdapter dense的计算结果，shape为 (*, out_channels)，out_channels与入参中的out_channels类型一致。
    """

    @_args_type_validator_check(in_channels=Validator.check_positive_int,
                                out_channels=Validator.check_positive_int,
                                has_bias=Validator.check_bool,
                                reduction_factor=Validator.check_positive_int,
                                low_rank_size=Validator.check_positive_int,
                                non_linearity=_valid_value_checks(list(_activation.keys()),
                                                                  "LowRankAdapterDense"),
                                param_init_type=_valid_value_checks(VALID_TENSOR_DATATYPE,
                                                                    "LowRankAdapterDense"),
                                compute_dtype=_valid_value_checks(VALID_TENSOR_DATATYPE,
                                                                  "LowRankAdapterDense"))
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None,
                 reduction_factor: int = 1,
                 low_rank_size: int = 1,
                 low_rank_w_init="xavier_uniform",
                 non_linearity: str = "gelu",
                 param_init_type: mstype = mstype.float32,
                 compute_dtype: mstype = mstype.float16):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         weight_init=weight_init,
                         bias_init=bias_init,
                         has_bias=has_bias,
                         activation=activation)
        self.mindpet_delta_low_rank_adapter = LowRankAdapterLayer(hidden_size=out_channels,
                                                             reduction_factor=reduction_factor,
                                                             low_rank_size=low_rank_size,
                                                             low_rank_w_init=low_rank_w_init,
                                                             non_linearity=non_linearity,
                                                             param_init_type=param_init_type,
                                                             compute_dtype=compute_dtype, )
        self.compt_dtype = compute_dtype
        self.cast = P.Cast()
        self.act_name = activation

    def construct(self, input_tensor):
        """Forward"""
        # get input_x info
        x_shape = self.shape_op(input_tensor)
        ori_dtype = F.dtype(input_tensor)

        # reshape input_tensor to compute
        input_tensor = self.reshape(input_tensor, (-1, x_shape[-1]))

        # start to linear compute
        weight = self.cast(self.weight, self.compt_dtype)
        input_tensor = self.cast(input_tensor, self.compt_dtype)

        input_tensor = self.matmul(input_tensor, weight)
        if self.has_bias:
            input_tensor = self.bias_add(
                input_tensor, self.cast(self.bias, self.compt_dtype))
        if self.activation_flag:
            input_tensor = self.activation(input_tensor)

        # calculate low_rank_adapter_out
        input_tensor = self.mindpet_delta_low_rank_adapter(input_tensor)

        # recover the previous outshape and dtype
        out_shape = x_shape[:-1] + (-1,)
        input_tensor = self.reshape(input_tensor, out_shape)
        output = self.cast(input_tensor, ori_dtype)
        return output

    def shard(self, strategy_matmul_org=None,
              strategy_bias_org=None,
              strategy_activation_org=None,
              strategy_matmul_down_sampler_weight=None,
              strategy_matmul_down_sampler_input=None,
              strategy_bias_down_sampler=None,
              strategy_non_linearity=None,
              strategy_matmul_up_sampler_weight=None,
              strategy_matmul_up_sampler_input=None,
              strategy_bias_up_sampler=None,
              strategy_residual_add=None):
        """
        Set the shard for the LowRankAdapterDense. All the strategy size should be equal to the inputs.

        Note:
            It is valid only in semi auto parallel or auto parallel mode.
            In other parallel modes, strategies set here will be ignored.

        Args:
            strategy_matmul_org (tuple): The strategy for the origin dense matmul.
            strategy_bias_org (tuple): The strategy for the origin dense bias_add.
            strategy_activation_org (tuple): The strategy for the LowRankAdapterDense dropout.
            strategy_matmul_down_sampler_weight (tuple): The strategy for the
                LowRankAdapter-down_sampler-matmul_weight matmul.
            strategy_matmul_down_sampler_input (tuple): The strategy for the
                LowRankAdapter-down_sampler-matmul_input matmul.
            strategy_bias_down_sampler (tuple): The strategy for the
                LowRankAdapter-down_sampler-bias_add.
            strategy_non_linearity (tuple): The strategy for the LowRankAdapter non_linearity.
            strategy_matmul_up_sampler_weight (tuple): The strategy for the
                LowRankAdapter-up_sampler-matmul_weight matmul.
            strategy_matmul_up_sampler_input (tuple): The strategy for the
                LowRankAdapter-up_sampler-matmul_input matmul.
            strategy_bias_up_sampler (tuple): The strategy for the
                LowRankAdapter-up_sampler-bias_add.
            strategy_residual_add (tuple): The strategy for the residual_add.
        """
        try:
            # set origin dense strategy
            self.matmul.shard(strategy_matmul_org)
            if self.has_bias:
                self.bias_add.shard(strategy_bias_org)
            if self.activation_flag and isinstance(self.act_name, str):
                if self.act_name.lower() == "leakyrelu":
                    self.activation.select_op.shard(
                        (strategy_activation_org[0], strategy_activation_org[0]))
                elif self.act_name.lower() == "logsigmoid":
                    self.activation.mul.shard((strategy_activation_org[0], ()))
                    self.activation.exp.shard(strategy_activation_org)
                    self.activation.add.shard((strategy_activation_org[0], ()))
                    self.activation.rec.shard(strategy_activation_org)
                    self.activation.log.shard(strategy_activation_org)
                elif self.act_name.lower() == "logsoftmax":
                    raise ValueError("The 'LogSoftmax' function is not supported in semi auto parallel "
                                     "or auto parallel mode.")
                else:
                    getattr(self.activation, self.act_name).shard(
                        strategy_activation_org)

            # set low_rank_adapter strategy
            self.mindpet_delta_low_rank_adapter.shard(strategy_matmul_down_sampler_weight,
                                                 strategy_matmul_down_sampler_input,
                                                 strategy_bias_down_sampler,
                                                 strategy_non_linearity,
                                                 strategy_matmul_up_sampler_weight,
                                                 strategy_matmul_up_sampler_input,
                                                 strategy_bias_up_sampler,
                                                 strategy_residual_add)
        except Exception as ex:
            # pylint: disable=W0719
            raise Exception(
                f"Exception occurred when set the shard for LowRankAdapterDense, error message: {str(ex)}") from ex

        return self
