#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2010-2022. All rights reserved.

import sys
sys.path.append('.')

import os
import logging
import unittest
import argparse

import mindspore
import numpy as np
import pytest
from mindspore import dtype as mstype
from mindspore.common.tensor import Tensor

from mindpet.delta.adapter import AdapterDense
from mindpet.delta.adapter import AdapterLayer

logging.getLogger().setLevel(logging.INFO)

class TestAdapter(unittest.TestCase):
    # _check_in_channels
    def test_check_type_with_float_in_channels(self):
        logging.info('Start test_check_type_with_float_in_channels.')
        with self.assertRaises(TypeError) as ex:
            AdapterDense(in_channels=32.5, out_channels=32, bottleneck_size=8)
        logging.error(ex.exception)
        logging.info('Finish test_check_type_with_float_in_channels.')

    def test_check_value_with_zero_in_channels(self):
        logging.info('Start test_check_value_with_zero_in_channels.')
        with self.assertRaises(ValueError) as ex:
            AdapterDense(in_channels=0, out_channels=32, bottleneck_size=8)
        logging.error(ex.exception)
        logging.info('Finish test_check_value_with_zero_in_channels.')

    def test_check_value_with_negative_in_channels(self):
        logging.info('Start test_check_value_with_negative_in_channels.')
        with self.assertRaises(ValueError) as ex:
            AdapterDense(in_channels=-32, out_channels=32, bottleneck_size=8)
        logging.error(ex.exception)
        logging.info('Finish test_check_value_with_negative_in_channels.')

    # _check_out_channels
    def test_check_type_with_float_out_channels(self):
        logging.info('Start test_check_type_with_float_out_channels.')
        with self.assertRaises(TypeError) as ex:
            AdapterDense(in_channels=32, out_channels=32.5, bottleneck_size=8)
        logging.error(ex.exception)
        logging.info('Finish test_check_type_with_float_out_channels.')

    def test_check_value_with_zero_out_channels(self):
        logging.info('Start test_check_value_with_zero_out_channels.')
        with self.assertRaises(ValueError) as ex:
            AdapterDense(in_channels=32, out_channels=0, bottleneck_size=8)
        logging.error(ex.exception)
        logging.info('Finish test_check_value_with_zero_out_channels.')

    def test_check_value_with_negative_out_channels(self):
        logging.info('Start test_check_value_with_negative_out_channels.')
        with self.assertRaises(ValueError) as ex:
            AdapterDense(in_channels=32, out_channels=-32, bottleneck_size=8)
        logging.error(ex.exception)
        logging.info(
            'Finish test_check_value_with_negative_out_channels.')

    # _check_bottleneck_size
    def test_check_num_with_zero_bottleneck_size(self):
        logging.info('Start test_check_num_with_zero_bottleneck_size.')
        with self.assertRaises(ValueError) as ex:
            AdapterDense(in_channels=1, out_channels=1, bottleneck_size=0)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_zero_bottleneck_size.')
    
    def test_check_num_with_negative_bottleneck_size(self):
        logging.info('Start test_check_num_with_negative_bottleneck_size.')
        with self.assertRaises(ValueError) as ex:
            AdapterDense(in_channels=1, out_channels=1, bottleneck_size=-1)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_negative_bottleneck_size.')
    
    def test_check_type_of_data_with_float_bottleneck_size(self):
        logging.info('Start test_check_type_of_data_with_float_bottleneck_size.')
        with self.assertRaises(TypeError) as ex:
            AdapterDense(in_channels=1, out_channels=1, bottleneck_size=1.5)
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_float_bottleneck_size.')

    # check param_init_type
    def test_check_type_of_data_with_int_param_init_type(self):
        logging.info('Start test_check_type_of_data_with_int_param_init_type.')
        with self.assertRaises(TypeError) as ex:
            AdapterDense(in_channels=1, out_channels=1, bottleneck_size=8, param_init_type=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_int_param_init_type.')

    def test_check_type_of_data_with_str_param_init_type(self):
        logging.info('Start test_check_type_of_data_with_str_param_init_type.')
        with self.assertRaises(TypeError) as ex:
            AdapterDense(in_channels=1, out_channels=1, bottleneck_size=8, param_init_type='a')
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_str_param_init_type.')
    
    # check compute_dtype
    def test_check_type_of_data_with_int_compute_dtype(self):
        logging.info('Start test_check_type_of_data_with_int_compute_dtype.')
        with self.assertRaises(TypeError) as ex:
            AdapterDense(in_channels=1, out_channels=1, bottleneck_size=8, compute_dtype=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_int_compute_dtype.')

    def test_check_type_of_data_with_str_compute_dtype(self):
        logging.info('Start test_check_type_of_data_with_str_compute_dtype.')
        with self.assertRaises(TypeError) as ex:
            AdapterDense(in_channels=1, out_channels=1, bottleneck_size=8, compute_dtype='a')
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_str_compute_dtype.')
    
    # check activation
    def test_check_type_of_data_with_no_legal_activation(self):
        logging.info('Start test_check_type_of_data_with_no_legal_activation.')
        with self.assertRaises(KeyError) as ex:
            AdapterDense(in_channels=1, out_channels=1, bottleneck_size=8, activation='a')
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_no_legal_activation.')

    def test_check_type_of_data_with_int_activation(self):
        logging.info('Start test_check_type_of_data_with_int_activation.')
        with self.assertRaises(TypeError) as ex:
            AdapterDense(in_channels=1, out_channels=1, bottleneck_size=8, activation=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_int_activation.')

    # check weight_init
    def test_check_type_of_data_with_no_legal_weight_init(self):
        logging.info('Start test_check_type_of_data_with_no_legal_weight_init.')
        with self.assertRaises(ValueError) as ex:
            AdapterDense(in_channels=1, out_channels=1, bottleneck_size=8, weight_init='a')
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_no_legal_weight_init.')

    # check bias_init
    def test_check_type_of_data_with_no_legal_bias_init(self):
        logging.info('Start test_check_type_of_data_with_no_legal_bias_init.')
        with self.assertRaises(ValueError) as ex:
            AdapterDense(in_channels=1, out_channels=1, bottleneck_size=8, bias_init='a')
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_no_legal_bias_init.')

    # check non_linearity
    def test_check_type_of_data_with_no_legal_non_linearity(self):
        logging.info('Start test_check_type_of_data_with_no_legal_non_linearity.')
        with self.assertRaises(TypeError) as ex:
            AdapterDense(in_channels=1, out_channels=1, bottleneck_size=8, non_linearity='a')
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_no_legal_non_linearity.')

    def test_check_type_of_data_with_int_non_linearity(self):
        logging.info('Start test_check_type_of_data_with_int_non_linearity.')
        with self.assertRaises(TypeError) as ex:
            AdapterDense(in_channels=1, out_channels=1, bottleneck_size=8, non_linearity=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_int_non_linearity.')

    # param initialization
    def test_params_with_legal_bottleneck_size(self):
        logging.info('Start test_params_with_legal_bottleneck_size.')
        adapter = AdapterDense(in_channels=32, out_channels=32, bottleneck_size=64)
        self.assertEqual(64, adapter.bottleneck_size)
        logging.info('Finish test_params_with_legal_bottleneck_size')

    # construct
    def test_construct_with_weight_init(self):
        logging.info('Start test_construct_with_weight_init.')
        adapter = AdapterDense(in_channels=32, out_channels=32, bottleneck_size=8, weight_init='XavierUniform')
        input_tensor = Tensor(np.ones((8, 16, 32)), mstype.float32)
        adapter.construct(input_tensor)
        logging.info('Finish test_construct_with_weight_init.')

    def test_construct_with_bias_init(self):
        logging.info('Start test_construct_with_bias_init.')
        adapter = AdapterDense(in_channels=32, out_channels=32, bottleneck_size=8, bias_init='TruncatedNormal')
        input_tensor = Tensor(np.ones((8, 16, 32)), mstype.float32)
        adapter.construct(input_tensor)
        logging.info('Finish test_construct_with_bias_init.')

    def test_construct_with_false_has_bias(self):
        logging.info('Start test_construct_with_false_has_bias.')
        adapter = AdapterDense(in_channels=32, out_channels=32, bottleneck_size=8, has_bias=False)
        input_tensor = Tensor(np.ones((8, 16, 32)), mstype.float32)
        adapter.construct(input_tensor)
        logging.info('Finish test_construct_with_false_has_bias.')

    def test_construct_with_activation_flag(self):
        logging.info('Start test_construct_with_activation_flag.')
        adapter = AdapterDense(in_channels=32, out_channels=32, bottleneck_size=8, activation='sigmoid')
        input_tensor = Tensor(np.ones((8, 16, 32)), mstype.float32)
        adapter.construct(input_tensor)
        logging.info('Finish test_construct_with_activation_flag.')

    def test_construct_with_non_linearity(self):
        logging.info('Start test_construct_with_non_linearity.')
        adapter = AdapterDense(in_channels=32, out_channels=32, bottleneck_size=8, activation='sigmoid')
        input_tensor = Tensor(np.ones((8, 16, 32)), mstype.float32)
        adapter.construct(input_tensor)
        logging.info('Finish test_construct_with_non_linearity.')

    def test_shard_with_leakyrelu_activation(self):
        logging.info('Start test_shard_with_LeakyReLU_activation')
        adapter_dense = AdapterDense(in_channels=32, out_channels=32, bottleneck_size=8, activation='leakyrelu')
        leakyrelu_strategy_activation = ((1, 1), (1, 1))
        adapter_dense.shard(strategy_activation_org=leakyrelu_strategy_activation)
        self.assertEqual(adapter_dense.activation.select_op.in_strategy, leakyrelu_strategy_activation)
        logging.info('Finish test_shard_with_LeakyReLU_activation')

    def test_shard_with_logsigmoid_activation(self):
        logging.info('Start test_shard_with_logsigmoid_activation')
        adapter_dense = AdapterDense(in_channels=32, out_channels=32, bottleneck_size=8, activation='logsigmoid')
        logsigmoid_strategy_activation = ((1, 1), (1, 1))
        adapter_dense.shard(strategy_activation_org=logsigmoid_strategy_activation)
        self.assertEqual(adapter_dense.activation.log.in_strategy, logsigmoid_strategy_activation)
        logging.info('Finish test_shard_with_logsigmoid_activation')

    def test_shard_with_invalid_activation(self):
        logging.info('Start test_shard_with_invalid_activation')
        with self.assertRaises(Exception) as ex:
            adapter_dense = AdapterDense(in_channels=32, out_channels=32, activation="logsoftmax")
            adapter_dense.shard(strategy_activation_org=((1, 1), (1, 1)))
        logging.error(ex.exception)
        logging.info('Finish test_shard_with_invalid_activation')

    def test_shard_with_invalid_type_strategy(self):
        logging.info('Start test_shard_with_invalid_type_strategy')
        with self.assertRaises(Exception) as ex:
            adapter_dense = AdapterDense(in_channels=32, out_channels=32)
            adapter_dense.shard(strategy_matmul_org=1)
        logging.error(ex.exception)
        logging.info('Finish test_shard_with_invalid_type_strategy')


class TestAdapterLayer(unittest.TestCase):
    # check hidden_size
    def test_check_type_with_float_hidden_size(self):
        logging.info('Start test_check_type_with_float_hidden_size.')
        with self.assertRaises(TypeError) as ex:
            AdapterLayer(hidden_size=16.5, bottleneck_size=8)
        logging.error(ex.exception)
        logging.info('Finish test_check_type_with_float_hidden_size.')

    def test_check_value_with_zero_hidden_size(self):
        logging.info('Start test_check_value_with_zero_hidden_size.')
        with self.assertRaises(ValueError) as ex:
            AdapterLayer(hidden_size=0, bottleneck_size=8)
        logging.error(ex.exception)
        logging.info('Finish test_check_value_with_zero_hidden_size.')

    def test_check_value_with_negative_hidden_size(self):
        logging.info('Start test_check_value_with_negative_hidden_size.')
        with self.assertRaises(ValueError) as ex:
            AdapterLayer(hidden_size=-16, bottleneck_size=8)
        logging.error(ex.exception)
        logging.info(
            'Finish test_check_value_with_negative_hidden_size.')

    # check bottleneck_size
    def test_check_num_with_zero_bottleneck_size(self):
        logging.info('Start test_check_num_with_zero_bottleneck_size.')
        with self.assertRaises(ValueError) as ex:
            AdapterLayer(hidden_size=1, bottleneck_size=0)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_zero_bottleneck_size.')
    
    def test_check_num_with_negative_bottleneck_size(self):
        logging.info('Start test_check_num_with_negative_bottleneck_size.')
        with self.assertRaises(ValueError) as ex:
            AdapterLayer(hidden_size=1, bottleneck_size=-1)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_negative_bottleneck_size.')
    
    def test_check_type_of_data_with_float_bottleneck_size(self):
        logging.info('Start test_check_type_of_data_with_float_bottleneck_size.')
        with self.assertRaises(TypeError) as ex:
            AdapterLayer(hidden_size=1, bottleneck_size=1.5)
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_float_bottleneck_size.')

    # check param_init_type
    def test_check_type_of_data_with_int_param_init_type(self):
        logging.info('Start test_check_type_of_data_with_int_param_init_type.')
        with self.assertRaises(TypeError) as ex:
            AdapterLayer(hidden_size=1, bottleneck_size=8, param_init_type=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_int_param_init_type.')

    def test_check_type_of_data_with_str_param_init_type(self):
        logging.info('Start test_check_type_of_data_with_str_param_init_type.')
        with self.assertRaises(TypeError) as ex:
            AdapterLayer(hidden_size=1, bottleneck_size=8, param_init_type='a')
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_str_param_init_type.')

    # check compute_dtype
    def test_check_type_of_data_with_int_compute_dtype(self):
        logging.info('Start test_check_type_of_data_with_int_compute_dtype.')
        with self.assertRaises(TypeError) as ex:
            AdapterLayer(hidden_size=1, bottleneck_size=8, compute_dtype=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_int_compute_dtype.')

    def test_check_type_of_data_with_str_compute_dtype(self):
        logging.info('Start test_check_type_of_data_with_str_compute_dtype.')
        with self.assertRaises(TypeError) as ex:
            AdapterLayer(hidden_size=1, bottleneck_size=8, compute_dtype='a')
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_str_compute_dtype.')

    # check non_linearity
    def test_check_type_of_data_with_no_legal_non_linearity(self):
        logging.info('Start test_check_type_of_data_with_no_legal_non_linearity.')
        with self.assertRaises(TypeError) as ex:
            AdapterLayer(hidden_size=1, bottleneck_size=8, non_linearity='a')
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_no_legal_non_linearity.')

    def test_check_type_of_data_with_int_non_linearity(self):
        logging.info('Start test_check_type_of_data_with_int_non_linearity.')
        with self.assertRaises(TypeError) as ex:
            AdapterLayer(hidden_size=1, bottleneck_size=8, non_linearity=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_int_non_linearity.')

    # param initialization
    def test_params_with_legal_bottleneck_size(self):
        logging.info('Start test_params_with_legal_bottleneck_size.')
        adapter = AdapterLayer(hidden_size=32, bottleneck_size=64)
        self.assertEqual(64, adapter.bottleneck_size)
        logging.info('Finish test_params_with_legal_bottleneck_size')

    # construct
    def test_construct_with_non_linearity(self):
        logging.info('Start test_construct_with_non_linearity.')
        adapter = AdapterLayer(hidden_size=32, bottleneck_size=8, non_linearity='sigmoid')
        input_tensor = Tensor(np.ones((8, 16, 32)), mstype.float32)
        adapter.construct(input_tensor)
        logging.info('Finish test_construct_with_non_linearity.')

    def test_shard_with_leakyrelu_activation(self):
        logging.info('Start test_shard_with_LeakyReLU_activation')
        adapter_layer = AdapterLayer(hidden_size=32, bottleneck_size=8, non_linearity='leakyrelu')
        leakyrelu_strategy_non_linearity = ((1, 1), (1, 1))
        adapter_layer.shard(strategy_non_linearity=leakyrelu_strategy_non_linearity)
        self.assertEqual(adapter_layer.mindpet_delta_adapter_block.mindpet_delta_adapter_non_linear.select_op.in_strategy, 
            leakyrelu_strategy_non_linearity)
        logging.info('Finish test_shard_with_LeakyReLU_activation')

    def test_shard_with_logsigmoid_activation(self):
        logging.info('Start test_shard_with_logsigmoid_activation')
        adapter_layer = AdapterLayer(hidden_size=32, bottleneck_size=8, non_linearity='logsigmoid')
        logsigmoid_strategy_non_linearity = ((1, 1), (1, 1))
        adapter_layer.shard(strategy_non_linearity=logsigmoid_strategy_non_linearity)
        self.assertEqual(adapter_layer.mindpet_delta_adapter_block.mindpet_delta_adapter_non_linear.log.in_strategy, 
            logsigmoid_strategy_non_linearity)
        logging.info('Finish test_shard_with_logsigmoid_activation')

    def test_shard_with_invalid_activation(self):
        logging.info('Start test_shard_with_invalid_activation')
        with self.assertRaises(Exception) as ex:
            adapter_layer = AdapterLayer(hidden_size=32, bottleneck_size=8, non_linearity='logsoftmax')
            adapter_layer.shard(strategy_non_linearity=((1, 1), (1, 1)))
        logging.error(ex.exception)
        logging.info('Finish test_shard_with_invalid_activation')

    def test_shard_with_invalid_type_strategy(self):
        logging.info('Start test_shard_with_invalid_type_strategy')
        with self.assertRaises(Exception) as ex:
            adapter_layer = AdapterLayer(hidden_size=32, bottleneck_size=8)
            adapter_layer.shard(strategy_matmul_down_sampler=1)
        logging.error(ex.exception)
        logging.info('Finish test_shard_with_invalid_type_strategy')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()
    mindspore.set_context(device_id=args.device_id)
    pytest.main(["-s", os.path.abspath(__file__)])
 