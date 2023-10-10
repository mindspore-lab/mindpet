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
import numpy
import pytest
from mindspore import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import One

from mindpet.delta.lora import LoRADense

logging.getLogger().setLevel(logging.INFO)

class TestLoRADense(unittest.TestCase):
    # _check_num
    def test_check_num_with_zero_lora_rank(self):
        logging.info('Start test_check_num_with_zero_lora_rank.')
        with self.assertRaises(ValueError) as ex:
            LoRADense(in_channels=1, out_channels=1, lora_rank=0, lora_alpha=2, lora_dropout=0.,
                      param_init_type=mstype.float32)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_zero_lora_rank.')

    def test_check_num_with_float_lora_rank(self):
        logging.info('Start test_check_num_with_float_lora_rank.')
        with self.assertRaises(TypeError) as ex:
            LoRADense(in_channels=1, out_channels=1, lora_rank=1.2, lora_alpha=2, lora_dropout=0.,
                      param_init_type=mstype.float32)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_float_lora_rank.')

    def test_check_num_with_negative_lora_rank(self):
        logging.info('Start test_check_num_with_negative_lora_rank.')
        with self.assertRaises(ValueError) as ex:
            LoRADense(in_channels=1, out_channels=1, lora_rank=-5, lora_alpha=2, lora_dropout=0.,
                      param_init_type=mstype.float32)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_negative_lora_rank.')

    def test_check_num_with_str_lora_alpha(self):
        logging.info('Start test_check_num_with_str_lora_alpha.')
        with self.assertRaises(TypeError) as ex:
            LoRADense(in_channels=1, out_channels=1, lora_rank=2, lora_alpha='a', lora_dropout=0.,
                      param_init_type=mstype.float32)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_str_lora_alpha.')

    def test_check_num_with_bool_lora_alpha(self):
        logging.info('Start test_check_num_with_bool_lora_alpha.')
        with self.assertRaises(TypeError) as ex:
            LoRADense(in_channels=1, out_channels=1, lora_rank=2, lora_alpha=True, lora_dropout=0.,
                      param_init_type=mstype.float32)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_bool_lora_alpha.')

    def test_check_num_with_zero_lora_alpha(self):
        logging.info('Start test_check_num_with_zero_lora_alpha.')
        with self.assertRaises(ValueError) as ex:
            LoRADense(in_channels=1, out_channels=1, lora_rank=2, lora_alpha=0.0, lora_dropout=0.,
                      param_init_type=mstype.float32)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_zero_lora_alpha.')

    def test_check_num_with_int_lora_dropout(self):
        logging.info('Start test_check_num_with_str_lora_dropout.')
        with self.assertRaises(TypeError) as ex:
            LoRADense(in_channels=1, out_channels=1, lora_rank=2, lora_alpha=2, lora_dropout=0,
                      param_init_type=mstype.float32)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_str_lora_dropout.')

    def test_check_num_with_negative_lora_dropout(self):
        logging.info('Start test_check_num_with_negative_lora_dropout.')
        with self.assertRaises(ValueError) as ex:
            LoRADense(in_channels=1, out_channels=1, lora_rank=2, lora_alpha=2, lora_dropout=-0.5,
                      param_init_type=mstype.float32)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_negative_lora_dropout.')

    def test_check_num_with_lora_dropout_equal_to_one(self):
        logging.info('Start test_check_num_with_lora_dropout_equal_to_one.')
        with self.assertRaises(ValueError) as ex:
            LoRADense(in_channels=1, out_channels=1, lora_rank=2, lora_alpha=2, lora_dropout=1.0,
                      param_init_type=mstype.float32)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_lora_dropout_equal_to_one.')

    # _check_init
    def test_check_init_with_lora_a_init_shape_not_equal_to_2(self):
        logging.info('Start test_check_init_with_lora_a_init_shape_not_equal_to_2.')
        with self.assertRaises(ValueError) as ex:
            LoRADense(in_channels=2, out_channels=2, lora_rank=2, lora_alpha=2, lora_dropout=0.,
                      lora_a_init=Tensor(shape=(2, 2, 2), dtype=mstype.float32, init='normal'))
        logging.error(ex.exception)
        logging.info('Finish test_check_init_with_lora_a_init_shape_not_equal_to_2.')

    def test_check_init_with_lora_a_init_shape0_not_equal_to_lora_rank(self):
        logging.info('Start test_check_init_with_lora_a_init_shape0_not_equal_to_lora_rank.')
        with self.assertRaises(ValueError) as ex:
            LoRADense(in_channels=2, out_channels=2, lora_rank=5, lora_alpha=2, lora_dropout=0.,
                      lora_a_init=Tensor(shape=(2, 2), dtype=mstype.float32, init='normal'))
        logging.error(ex.exception)
        logging.info('Finish test_check_init_with_lora_a_init_shape0_not_equal_to_lora_rank.')

    def test_check_init_with_lora_a_init_shape1_not_equal_to_in_channels(self):
        logging.info('Start test_check_init_with_lora_a_init_shape1_not_equal_to_in_channels.')
        with self.assertRaises(ValueError) as ex:
            LoRADense(in_channels=5, out_channels=2, lora_rank=2, lora_alpha=2, lora_dropout=0.,
                      lora_a_init=Tensor(shape=(2, 2), dtype=mstype.float32, init='normal'))
        logging.error(ex.exception)
        logging.info('Finish test_check_init_with_lora_a_init_shape1_not_equal_to_in_channels.')

    def test_check_init_with_lora_b_init_shape_not_equal_to_two(self):
        logging.info('Start test_check_init_with_lora_b_init_shape_not_equal_to_two.')
        with self.assertRaises(ValueError) as ex:
            LoRADense(in_channels=2, out_channels=2, lora_rank=2, lora_alpha=2, lora_dropout=0.,
                      lora_b_init=Tensor(shape=(2, 2, 2), dtype=mstype.float32, init='normal'))
        logging.error(ex.exception)
        logging.info('Finish test_check_init_with_lora_b_init_shape_not_equal_to_two.')

    def test_check_init_with_lora_b_init_shape0_not_equal_to_out_channels(self):
        logging.info('Start test_check_init_with_lora_b_init_shape0_not_equal_to_out_channels.')
        with self.assertRaises(ValueError) as ex:
            LoRADense(in_channels=2, out_channels=5, lora_rank=5, lora_alpha=2, lora_dropout=0.,
                      lora_b_init=Tensor(shape=(2, 2), dtype=mstype.float32, init='normal'))
        logging.error(ex.exception)
        logging.info('Finish test_check_init_with_lora_b_init_shape0_not_equal_to_out_channels.')

    def test_check_init_with_lora_b_init_shape1_not_equal_to_lora_rank(self):
        logging.info('Start test_check_init_with_lora_b_init_shape1_not_equal_to_lora_rank.')
        with self.assertRaises(ValueError) as ex:
            LoRADense(in_channels=2, out_channels=2, lora_rank=5, lora_alpha=2, lora_dropout=0.,
                      lora_b_init=Tensor(shape=(2, 2), dtype=mstype.float32, init='normal'))
        logging.error(ex.exception)
        logging.info('Finish test_check_init_with_lora_b_init_shape1_not_equal_to_lora_rank.')

    # _check_type_of_data
    def test_check_type_of_data_with_int_param_init_type(self):
        logging.info('Start test_check_type_of_data_with_int_param_init_type.')
        with self.assertRaises(TypeError) as ex:
            LoRADense(in_channels=1, out_channels=1, lora_rank=2, lora_alpha=2, lora_dropout=0.,
                      param_init_type=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_int_param_init_type.')

    def test_check_type_of_data_with_str_param_init_type(self):
        logging.info('Start test_check_type_of_data_with_str_param_init_type.')
        with self.assertRaises(TypeError) as ex:
            LoRADense(in_channels=1, out_channels=1, lora_rank=2, lora_alpha=2, lora_dropout=0.,
                      param_init_type='a')
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_str_param_init_type.')

    def test_check_type_of_data_with_int_compute_dtype(self):
        logging.info('Start test_check_type_of_data_with_int_compute_dtype.')
        with self.assertRaises(TypeError) as ex:
            LoRADense(in_channels=1, out_channels=1, lora_rank=2, lora_alpha=2, lora_dropout=0.,
                      compute_dtype=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_int_compute_dtype.')

    def test_check_type_of_data_with_str_compute_dtype(self):
        logging.info('Start test_check_type_of_data_with_str_compute_dtype.')
        with self.assertRaises(TypeError) as ex:
            LoRADense(in_channels=1, out_channels=1, lora_rank=2, lora_alpha=2, lora_dropout=0.,
                      compute_dtype='a')
        logging.error(ex.exception)
        logging.info('Finish test_check_type_of_data_with_str_compute_dtype.')

    # param initialization
    def test_params_with_legal_lora_rank(self):
        logging.info('Start test_params_with_legal_lora_rank')
        lora = LoRADense(in_channels=1, out_channels=1, lora_rank=2, lora_alpha=2, lora_dropout=0.)
        self.assertEqual(2, lora.lora_rank)
        logging.info("Finish test_params_with_legal_lora_rank")

    def test_params_with_legal_mindpet_delta_lora_a(self):
        logging.info('Start test_params_with_legal_mindpet_delta_lora_a')
        lora = LoRADense(in_channels=1, out_channels=1, lora_rank=2, lora_alpha=4, lora_dropout=0.9,
                         lora_a_init=Tensor(shape=(2, 1), dtype=mstype.int8, init=One()))
        target = Tensor([[1], [1]]).asnumpy() == lora.mindpet_delta_lora_a.asnumpy()
        for result in target:
            self.assertTrue(result)
        logging.info("Finish test_params_with_legal_mindpet_delta_lora_a")

    def test_params_with_legal_mindpet_delta_lora_b(self):
        logging.info('Start test_params_with_legal_mindpet_delta_lora_b')
        lora = LoRADense(in_channels=3, out_channels=2, lora_rank=3, lora_alpha=4, lora_dropout=0.9,
                         lora_b_init=Tensor(shape=(2, 3), dtype=mstype.int8, init=One()))
        target = Tensor([[1, 1, 1], [1, 1, 1]]).asnumpy() == lora.mindpet_delta_lora_b.asnumpy()
        for _ in target:
            for result in _:
                self.assertTrue(result)
        logging.info("Finish test_params_with_legal_mindpet_delta_lora_b")

    # construct
    def test_construct_with_has_bias(self):
        logging.info('Start test_construct_with_false_has_bias')
        lora = LoRADense(in_channels=2, out_channels=2, lora_rank=3, lora_alpha=4, lora_dropout=0.0, has_bias=False)
        input_tensor = Tensor([[1, 2], [3, 4]], dtype=mstype.float32)
        lora.construct(input_tensor)
        logging.info('Finish test_construct_with_has_bias')

    def test_construct_with_activation_flag(self):
        logging.info('Start test_construct_with_activation_flag')
        lora = LoRADense(in_channels=3, out_channels=2, lora_rank=3, lora_alpha=4, lora_dropout=0.9,
                         activation='relu')
        input_tensor = Tensor([[1, 2, 3], [1, 2, 3]], dtype=mstype.float32)
        lora.construct(input_tensor)
        logging.info('Finish test_construct_with_activation_flag')

    def test_construct_with_x_shape_not_equal_to_2(self):
        logging.info('Start test_construct_with_x_shape_not_equal_to_2')
        lora = LoRADense(in_channels=3, out_channels=2, lora_rank=3, lora_alpha=4, lora_dropout=0.0)
        input_tensor = Tensor(numpy.ones(shape=[3, 2, 3]), mstype.float32)
        lora.construct(input_tensor)
        logging.info('Finish test_construct_with_x_shape_not_equal_to_2')

    def test_lora_shard_with_leakyrelu_activation(self):
        logging.info('Start test_lora_shard_with_leakyrelu_activation')
        lora = LoRADense(in_channels=1, out_channels=1, lora_rank=2, lora_alpha=2, lora_dropout=0.,
                         activation="leakyrelu")
        leakyrelu_strategy_activation = ((1, 1), (1, 1))
        lora.shard(strategy_activation=leakyrelu_strategy_activation)
        self.assertEqual(lora.activation.select_op.in_strategy, leakyrelu_strategy_activation)
        logging.info('Finish test_lora_shard_with_leakyrelu_activation')

    def test_lora_shard_with_logsigmoid_activation(self):
        logging.info('Start test_lora_shard_with_logsigmoid_activation')
        lora = LoRADense(in_channels=1, out_channels=1, lora_rank=2, lora_alpha=2, lora_dropout=0.,
                         activation="logsigmoid")
        logsigmoid_strategy_activation = ((1, 1), (1, 1))
        lora.shard(strategy_activation=logsigmoid_strategy_activation)
        self.assertEqual(lora.activation.log.in_strategy, logsigmoid_strategy_activation)
        logging.info('Finish test_lora_shard_with_logsigmoid_activation')

    def test_lora_shard_with_invalid_activation(self):
        logging.info('Start test_lora_shard_with_invalid_activation')
        with self.assertRaises(Exception) as ex:
            lora = LoRADense(in_channels=1, out_channels=1, lora_rank=2, lora_alpha=2,
                             lora_dropout=0., activation="logsoftmax")
            lora.shard(strategy_activation=((1, 1), (1, 1)))
        logging.error(ex.exception)
        logging.info('Finish test_lora_shard_with_invalid_activation')

    def test_lora_shard_with_invalid_type_strategy(self):
        logging.info('Start test_lora_shard_with_invalid_type_strategy')
        with self.assertRaises(Exception) as ex:
            lora = LoRADense(in_channels=1, out_channels=1, lora_rank=2, lora_alpha=2, lora_dropout=0.)
            lora.shard(strategy_org_dense_matmul=1)
        logging.error(ex.exception)
        logging.info('Finish test_lora_shard_with_invalid_type_strategy')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()
    mindspore.set_context(device_id=args.device_id)
    pytest.main(["-s", os.path.abspath(__file__)])
