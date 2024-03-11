#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2010-2022. All rights reserved.
import sys
sys.path.append('.')
import os
import logging
import unittest
import operator
import argparse
import mindspore

import numpy
import pytest
from mindspore import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import One

from mindpet.delta.low_rank_adapter import LowRankAdapterDense, LowRankAdapterLayer

logging.getLogger().setLevel(logging.INFO)

class TestLowRankAdapterDense(unittest.TestCase):
    # _check_origin_params
    # _check_in_channels
    def test_dense_check_type_with_float_in_channels(self):
        logging.info('Start test_dense_check_type_with_float_in_channels.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterDense(in_channels=5.1,
                                      out_channels=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info('Finish test_dense_check_type_with_float_in_channels.')

    def test_dense_check_value_with_zero_in_channels(self):
        logging.info('Start test_dense_check_value_with_zero_in_channels.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterDense(in_channels=0,
                                      out_channels=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info('Finish test_dense_check_value_with_zero_in_channels.')

    def test_dense_check_value_with_negative_in_channels(self):
        logging.info('Start test_dense_check_value_with_negative_in_channels.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterDense(in_channels=-5,
                                      out_channels=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_value_with_negative_in_channels.')

    # _check_out_channels
    def test_dense_check_type_with_float_out_channels(self):
        logging.info('Start test_dense_check_type_with_float_out_channels.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4.1,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info('Finish test_dense_check_type_with_float_out_channels.')

    def test_dense_check_value_with_zero_out_channels(self):
        logging.info('Start test_dense_check_value_with_zero_out_channels.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=0,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info('Finish test_dense_check_value_with_zero_out_channels.')

    def test_dense_check_value_with_negative_out_channels(self):
        logging.info(
            'Start test_dense_check_value_with_negative_out_channels.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=-4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_value_with_negative_out_channels.')

    # _check activation

    def test_dense_check_type_of_data_with_illegal_str_activation(self):
        logging.info(
            'Start test_dense_check_type_of_data_with_illegal_str_activation.')
        with self.assertRaises(KeyError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      activation="1",
                                      reduction_factor=2)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_type_of_data_with_illegal_str_activation.')

    def test_dense_check_type_of_data_with_int_activation(self):
        logging.info(
            'Start test_dense_check_type_of_data_with_int_activation.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      activation=1,
                                      reduction_factor=2)

        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_type_of_data_with_int_activation.')

    def test_dense_check_type_of_data_with_none_activation(self):
        logging.info(
            'Start test_dense_check_type_of_data_with_none_activation.')
        net = LowRankAdapterDense(in_channels=5,
                                  out_channels=4,
                                  activation=None,
                                  reduction_factor=2)
        logging.info(
            'Finish test_dense_check_type_of_data_with_none_activation.')

    # _check weight_init
    def test_dense_check_type_of_data_with_illegal_str_weight_init(self):
        logging.info(
            'Start test_dense_check_type_of_data_with_illegal_str_weight_init.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      weight_init="1",
                                      reduction_factor=2)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_type_of_data_with_illegal_str_weight_init.')

    def test_dense_check_init_with_weight_init_shape_length_not_equal_to_two(self):
        logging.info(
            'Start test_dense_check_init_with_weight_init_shape_length_not_equal_to_two.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      weight_init=Tensor(numpy.ones(
                                          (2, 4, 5)), mstype.float32),
                                      reduction_factor=3)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_init_with_weight_init_shape_length_not_equal_to_two.')

    def test_dense_check_init_with_weight_init_shape0_not_equal_to_out_channels(self):
        logging.info(
            'Start test_dense_check_init_with_weight_init_shape0_not_equal_to_out_channels.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      weight_init=Tensor(
                                          numpy.ones((5, 5)), mstype.float32),
                                      reduction_factor=3)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_init_with_weight_init_shape0_not_equal_to_out_channels.')

    def test_dense_check_init_with_weight_init_shape1_not_equal_to_in_channels(self):
        logging.info(
            'Start test_dense_check_init_with_weight_init_shape1_not_equal_to_in_channels.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      weight_init=Tensor(
                                          numpy.ones((4, 3)), mstype.float32),
                                      reduction_factor=3)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_init_with_weight_init_shape1_not_equal_to_in_channels.')

    # _check bias_init
    def test_dense_check_type_of_data_with_illegal_str_bias_init(self):
        logging.info(
            'Start test_dense_check_type_of_data_with_illegal_str_bias_init.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      bias_init="1",
                                      reduction_factor=2)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_type_of_data_with_illegal_str_bias_init.')

    def test_dense_check_init_with_bias_init_shape_length_not_equal_to_one(self):
        logging.info(
            'Start test_dense_check_init_with_bias_init_shape_length_not_equal_to_one.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      bias_init=Tensor(numpy.ones(
                                          (2, 4)), mstype.float32),
                                      reduction_factor=3)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_init_with_bias_init_shape_length_not_equal_to_one.')

    def test_dense_check_init_with_bias_init_shape1_not_equa0_to_out_channels(self):
        logging.info(
            'Start test_dense_check_init_with_bias_init_shape1_not_equa0_to_out_channels.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      bias_init=Tensor(
                                          numpy.ones((5)), mstype.float32),
                                      reduction_factor=3)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_init_with_bias_init_shape1_not_equa0_to_out_channels.')

    # _check has_bias
    def test_dense_check_type_of_data_with_str_has_bias(self):
        logging.info('Start test_dense_check_type_of_data_with_str_has_bias.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      has_bias="1",
                                      reduction_factor=2)
        logging.error(ex.exception)
        logging.info('Finish test_dense_check_type_of_data_with_str_has_bias.')

    def test_dense_check_type_of_data_with_int_has_bias(self):
        logging.info('Start test_dense_check_type_of_data_with_int_has_bias.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      has_bias=1,
                                      reduction_factor=2)
        logging.error(ex.exception)
        logging.info('Finish test_dense_check_type_of_data_with_int_has_bias.')

    # _check_reduction_factor
    def test_dense_check_type_with_float_reduction_factor(self):
        logging.info(
            'Start test_dense_check_type_with_float_reduction_factor.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      reduction_factor=3.2,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_type_with_float_reduction_factor.')

    def test_dense_check_value_with_zero_reduction_factor(self):
        logging.info(
            'Start test_dense_check_value_with_zero_reduction_factor.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      reduction_factor=0,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_value_with_zero_reduction_factor.')

    def test_dense_check_value_with_negative_reduction_factor(self):
        logging.info(
            'Start test_dense_check_value_with_negative_reduction_factor.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      reduction_factor=-3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_value_with_negative_reduction_factor.')

    # _check_low_rank_size
    def test_dense_check_type_with_float_low_rank_size(self):
        logging.info('Start test_dense_check_type_with_float_low_rank_size.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      reduction_factor=3,
                                      low_rank_size=1.2,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info('Finish test_dense_check_type_with_float_low_rank_size.')

    def test_dense_check_value_with_zero_low_rank_size(self):
        logging.info('Start test_dense_check_value_with_zero_low_rank_size.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      reduction_factor=3,
                                      low_rank_size=0,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info('Finish test_dense_check_value_with_zero_low_rank_size.')

    def test_dense_check_value_with_negative_low_rank_size(self):
        logging.info(
            'Start test_dense_check_value_with_negative_low_rank_size.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      reduction_factor=3,
                                      low_rank_size=-1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_value_with_negative_low_rank_size.')

    # _check_param_init_type
    def test_dense_check_type_of_data_with_int_param_init_type(self):
        logging.info(
            'Start test_dense_check_type_of_data_with_int_param_init_type.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=1,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_type_of_data_with_int_param_init_type.')

    def test_dense_check_type_of_data_with_str_param_init_type(self):
        logging.info(
            'Start test_dense_check_type_of_data_with_str_param_init_type.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type="a",
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_type_of_data_with_str_param_init_type.')

    def test_dense_check_type_of_data_with_illegal_dtype_param_init_type(self):
        logging.info(
            'Start test_dense_check_type_of_data_with_illegal_dtype_param_init_type.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float64,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_type_of_data_with_illegal_dtype_param_init_type.')

    # _check_compute_dtype
    def test_dense_check_type_of_data_with_int_compute_dtype(self):
        logging.info(
            'Start test_dense_check_type_of_data_with_int_compute_dtype.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=1)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_type_of_data_with_int_compute_dtype.')

    def test_dense_check_type_of_data_with_str_compute_dtype(self):
        logging.info(
            'Start test_dense_check_type_of_data_with_str_compute_dtype.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype='a')
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_type_of_data_with_str_compute_dtype.')

    def test_dense_check_type_of_data_with_illegal_dtype_compute_dtype(self):
        logging.info(
            'Start test_dense_check_type_of_data_with_illegal_dtype_compute_dtype.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float64)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_type_of_data_with_illegal_dtype_compute_dtype.')

    # _check_non_linearity

    def test_dense_check_type_of_data_with_int_non_linearity(self):
        logging.info(
            'Start test_dense_check_type_of_data_with_int_non_linearity.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity=1,
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_type_of_data_with_int_non_linearity.')

    def test_dense_check_value_of_data_with_illegal_str_non_linearity(self):
        logging.info(
            'Start test_dense_check_value_of_data_with_illegal_str_non_linearity.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="1",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_value_of_data_with_illegal_str_non_linearity.')

        # _check_low_rank_w_init

    def test_dense_check_type_of_data_with_int_low_rank_w_init(self):
        logging.info(
            'Start test_dense_check_type_of_data_with_int_low_rank_w_init.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      low_rank_w_init=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_type_of_data_with_int_low_rank_w_init.')

    def test_dense_check_type_of_data_with_illegal_str_low_rank_w_init(self):
        logging.info(
            'Start test_dense_check_type_of_data_with_illegal_str_low_rank_w_init.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      low_rank_w_init="a",
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_dense_check_type_of_data_with_illegal_str_low_rank_w_init.')

        # construct

    def test_dense_construct_with_param_init_type(self):
        logging.info('Start test_dense_construct_with_param_init_type.')
        input_tensor = Tensor(numpy.ones((2, 5)), mstype.float32)
        net = LowRankAdapterDense(in_channels=5,
                                  out_channels=4,
                                  reduction_factor=2,
                                  param_init_type=mstype.float32)
        output = net(input_tensor)
        check_shape_value = (2, 4)
        self.assertEqual(True, operator.eq(output.shape, check_shape_value))
        logging.info('Finish test_dense_construct_with_param_init_type.')

    def test_dense_construct_with_compute_dtype(self):
        logging.info('Start test_dense_construct_with_compute_dtype.')
        input_tensor = Tensor(numpy.ones((2, 5)), mstype.float32)
        net = LowRankAdapterDense(in_channels=5,
                                  out_channels=4,
                                  reduction_factor=2,
                                  compute_dtype=mstype.float16)
        output = net(input_tensor)
        check_shape_value = (2, 4)
        self.assertEqual(True, operator.eq(output.shape, check_shape_value))
        logging.info('Finish test_dense_construct_with_compute_dtype.')

    def test_dense_construct_with_low_rank_size(self):
        logging.info('Start test_dense_construct_with_low_rank_size.')
        input_tensor = Tensor(numpy.ones((2, 5)), mstype.float32)
        net = LowRankAdapterDense(in_channels=5,
                                  out_channels=4,
                                  reduction_factor=2,
                                  low_rank_size=1)
        output = net(input_tensor)
        check_shape_value = (2, 4)
        self.assertEqual(True, operator.eq(output.shape, check_shape_value))
        logging.info('Finish test_dense_construct_with_low_rank_size.')

    def test_dense_construct_with_low_rank_w_init(self):
        logging.info('Start test_dense_construct_with_low_rank_size.')
        input_tensor = Tensor(numpy.ones((2, 5)), mstype.float32)
        net = LowRankAdapterDense(in_channels=5,
                                  out_channels=4,
                                  reduction_factor=2,
                                  low_rank_w_init="xavier_uniform")
        output = net(input_tensor)
        check_shape_value = (2, 4)
        self.assertEqual(True, operator.eq(output.shape, check_shape_value))
        logging.info('Finish test_dense_construct_with_low_rank_w_init.')

    def test_dense_construct_with_non_linearity(self):
        logging.info('Start test_dense_construct_with_non_linearity.')
        input_tensor = Tensor(numpy.ones((2, 5)), mstype.float32)
        net = LowRankAdapterDense(in_channels=5,
                                  out_channels=4,
                                  reduction_factor=2,
                                  non_linearity="gelu")
        output = net(input_tensor)
        check_shape_value = (2, 4)
        self.assertEqual(True, operator.eq(output.shape, check_shape_value))
        logging.info('Finish test_dense_construct_with_non_linearity.')

    def test_dense_construct_with_weight_init(self):
        logging.info('Start test_dense_construct_with_weight_init.')
        input_tensor = Tensor(numpy.ones((2, 5)), mstype.float32)
        net = LowRankAdapterDense(in_channels=5,
                                  out_channels=4,
                                  weight_init='normal',
                                  reduction_factor=2)
        output = net(input_tensor)
        check_shape_value = (2, 4)
        self.assertEqual(True, operator.eq(output.shape, check_shape_value))
        logging.info('Finish test_dense_construct_with_weight_init.')

    def test_dense_construct_with_bias_init(self):
        logging.info('Start test_dense_construct_with_bias_init.')
        input_tensor = Tensor(numpy.ones((2, 5)), mstype.float32)
        net = LowRankAdapterDense(in_channels=5,
                                  out_channels=4,
                                  bias_init='zeros',
                                  has_bias=True,
                                  reduction_factor=2)
        output = net(input_tensor)
        check_shape_value = (2, 4)
        self.assertEqual(True, operator.eq(output.shape, check_shape_value))
        logging.info('Finish test_dense_construct_with_bias_init.')

    def test_dense_construct_with_false_has_bias(self):
        logging.info('Start test_dense_construct_with_false_has_bias.')
        input_tensor = Tensor(numpy.ones((2, 5)), mstype.float32)
        net = LowRankAdapterDense(in_channels=5,
                                  out_channels=4,
                                  has_bias=False,
                                  reduction_factor=2)
        output = net(input_tensor)
        check_shape_value = (2, 4)
        self.assertEqual(True, operator.eq(output.shape, check_shape_value))
        logging.info('Finish test_dense_construct_with_false_has_bias.')

    def test_dense_construct_with_activation(self):
        logging.info('Start test_dense_construct_with_activation.')
        input_tensor = Tensor(numpy.ones((2, 5)), mstype.float32)
        net = LowRankAdapterDense(in_channels=5,
                                  out_channels=4,
                                  activation='relu',
                                  reduction_factor=2)
        output = net(input_tensor)
        check_shape_value = (2, 4)
        self.assertEqual(True, operator.eq(output.shape, check_shape_value))
        logging.info('Finish test_dense_construct_with_activation.')

    def test_dense_construct_with_shape_bigger_than_2_input(self):
        logging.info(
            'Start test_dense_construct_with_shape_bigger_than_2_input')
        input_tensor = Tensor(numpy.ones((3, 2, 5)), mstype.float32)
        net = LowRankAdapterDense(in_channels=5,
                                  out_channels=4,
                                  reduction_factor=2)
        output = net(input_tensor)
        check_shape_value = (3, 2, 4)
        self.assertEqual(True, operator.eq(output.shape, check_shape_value))
        logging.info(
            'Finish test_dense_construct_with_shape_bigger_than_2_input')

    def test_dense_shard_with_invalid_activation(self):
        logging.info('Start test_dense_shard_with_invalid_activation')
        with self.assertRaises(Exception) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      activation="logsoftmax",
                                      reduction_factor=2)
            net.shard(strategy_activation_org=((1, 1), (1, 1)))
        logging.error(ex.exception)
        logging.info('Finish test_dense_shard_with_invalid_activation')

    def test_dense_shard_with_invalid_type_strategy(self):
        logging.info('Start test_dense_shard_with_invalid_type_strategy')
        with self.assertRaises(Exception) as ex:
            net = LowRankAdapterDense(in_channels=5,
                                      out_channels=4,
                                      reduction_factor=2)
            net.shard(strategy_matmul_org=1)
        logging.error(ex.exception)
        logging.info('Finish test_dense_shard_with_invalid_type_strategy')


class TestLowRankAdapterLayer(unittest.TestCase):
    # _check_hidden_size
    def test_layer_check_type_with_float_out_channels(self):
        logging.info('Start test_layer_check_type_with_float_out_channels.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterLayer(hidden_size=4.1,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info('Finish test_layer_check_type_with_float_out_channels.')

    def test_layer_check_value_with_zero_out_channels(self):
        logging.info('Start test_layer_check_value_with_zero_out_channels.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterLayer(hidden_size=0,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info('Finish test_layer_check_value_with_zero_out_channels.')

    def test_layer_check_value_with_negative_out_channels(self):
        logging.info(
            'Start test_layer_check_value_with_negative_out_channels.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterLayer(hidden_size=-4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_layer_check_value_with_negative_out_channels.')

    # _check_reduction_factor

    def test_layer_check_type_with_float_reduction_factor(self):
        logging.info(
            'Start test_layer_check_type_with_float_reduction_factor.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterLayer(hidden_size=4,
                                      reduction_factor=3.2,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_layer_check_type_with_float_reduction_factor.')

    def test_layer_check_value_with_zero_reduction_factor(self):
        logging.info(
            'Start test_layer_check_value_with_zero_reduction_factor.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterLayer(hidden_size=4,
                                      reduction_factor=0,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_layer_check_value_with_zero_reduction_factor.')

    def test_layer_check_value_with_negative_reduction_factor(self):
        logging.info(
            'Start test_layer_check_value_with_negative_reduction_factor.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterLayer(hidden_size=4,
                                      reduction_factor=-3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_layer_check_value_with_negative_reduction_factor.')

    # _check_low_rank_size
    def test_layer_check_type_with_float_low_rank_size(self):
        logging.info('Start test_layer_check_type_with_float_low_rank_size.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterLayer(hidden_size=4,
                                      reduction_factor=3,
                                      low_rank_size=1.2,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info('Finish test_layer_check_type_with_float_low_rank_size.')

    def test_layer_check_value_with_zero_low_rank_size(self):
        logging.info('Start test_layer_check_value_with_zero_low_rank_size.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterLayer(hidden_size=4,
                                      reduction_factor=3,
                                      low_rank_size=0,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info('Finish test_layer_check_value_with_zero_low_rank_size.')

    def test_layer_check_value_with_negative_low_rank_size(self):
        logging.info(
            'Start test_layer_check_value_with_negative_low_rank_size.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterLayer(hidden_size=4,
                                      reduction_factor=3,
                                      low_rank_size=-1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_layer_check_value_with_negative_low_rank_size.')

    # _check_param_init_type
    def test_layer_check_type_of_data_with_int_param_init_type(self):
        logging.info(
            'Start test_layer_check_type_of_data_with_int_param_init_type.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterLayer(hidden_size=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=1,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_layer_check_type_of_data_with_int_param_init_type.')

    def test_layer_check_type_of_data_with_str_param_init_type(self):
        logging.info(
            'Start test_layer_check_type_of_data_with_str_param_init_type.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterLayer(hidden_size=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type="a",
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_layer_check_type_of_data_with_str_param_init_type.')

    def test_layer_check_type_of_data_with_illegal_dtype_param_init_type(self):
        logging.info(
            'Start test_layer_check_type_of_data_with_illegal_dtype_param_init_type.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterLayer(hidden_sizes=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float64,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_layer_check_type_of_data_with_illegal_dtype_param_init_type.')

    # _check_compute_dtype
    def test_layer_check_type_of_data_with_int_compute_dtype(self):
        logging.info(
            'Start test_layer_check_type_of_data_with_int_compute_dtype.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterLayer(hidden_size=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=1)
        logging.error(ex.exception)
        logging.info(
            'Finish test_layer_check_type_of_data_with_int_compute_dtype.')

    def test_layer_check_type_of_data_with_str_compute_dtype(self):
        logging.info(
            'Start test_layer_check_type_of_data_with_str_compute_dtype.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterLayer(hidden_size=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype='a')
        logging.error(ex.exception)
        logging.info(
            'Finish test_layer_check_type_of_data_with_str_compute_dtype.')

    def test_layer_check_type_of_data_with_illegal_dtype_compute_dtype(self):
        logging.info(
            'Start test_layer_check_type_of_data_with_illegal_dtype_compute_dtype.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterLayer(hidden_size=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float64)
        logging.error(ex.exception)
        logging.info(
            'Finish test_layer_check_type_of_data_with_illegal_dtype_compute_dtype.')

    # _check_non_linearity

    def test_layer_check_type_of_data_with_int_non_linearity(self):
        logging.info(
            'Start test_layer_check_type_of_data_with_int_non_linearity.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterLayer(hidden_size=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity=1,
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_layer_check_type_of_data_with_int_non_linearity.')

    def test_layer_check_value_of_data_with_illegal_str_non_linearity(self):
        logging.info(
            'Start test_layer_check_value_of_data_with_illegal_str_non_linearity.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterLayer(hidden_size=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      non_linearity="1",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_layer_check_value_of_data_with_illegal_str_non_linearity.')

        # _check_low_rank_w_init

    def test_layer_check_type_of_data_with_int_low_rank_w_init(self):
        logging.info(
            'Start test_layer_check_type_of_data_with_int_low_rank_w_init.')
        with self.assertRaises(TypeError) as ex:
            net = LowRankAdapterLayer(hidden_size=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      low_rank_w_init=1,
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_layer_check_type_of_data_with_int_low_rank_w_init.')

    def test_layer_check_type_of_data_with_illegal_str_low_rank_w_init(self):
        logging.info(
            'Start test_layer_check_type_of_data_with_illegal_str_low_rank_w_init.')
        with self.assertRaises(ValueError) as ex:
            net = LowRankAdapterLayer(hidden_size=4,
                                      reduction_factor=3,
                                      low_rank_size=1,
                                      low_rank_w_init="a",
                                      non_linearity="relu",
                                      param_init_type=mstype.float32,
                                      compute_dtype=mstype.float16)
        logging.error(ex.exception)
        logging.info(
            'Finish test_layer_check_type_of_data_with_illegal_str_low_rank_w_init.')

        # construct

    def test_layer_construct_with_param_init_type(self):
        logging.info('Start test_layer_construct_with_param_init_type.')
        input_tensor = Tensor(numpy.ones((2, 4)), mstype.float32)
        net = LowRankAdapterLayer(hidden_size=4,
                                  reduction_factor=2,
                                  param_init_type=mstype.float32)
        output = net(input_tensor)
        check_shape_value = (2, 4)
        self.assertEqual(True, operator.eq(output.shape, check_shape_value))
        logging.info('Finish test_layer_construct_with_param_init_type.')

    def test_layer_construct_with_compute_dtype(self):
        logging.info('Start test_layer_construct_with_compute_dtype.')
        input_tensor = Tensor(numpy.ones((2, 4)), mstype.float32)
        net = LowRankAdapterLayer(hidden_size=4,
                                  reduction_factor=2,
                                  compute_dtype=mstype.float16)
        output = net(input_tensor)
        check_shape_value = (2, 4)
        self.assertEqual(True, operator.eq(output.shape, check_shape_value))
        logging.info('Finish test_layer_construct_with_compute_dtype.')

    def test_layer_construct_with_low_rank_size(self):
        logging.info('Start test_layer_construct_with_low_rank_size.')
        input_tensor = Tensor(numpy.ones((2, 4)), mstype.float32)
        net = LowRankAdapterLayer(hidden_size=4,
                                  reduction_factor=2,
                                  low_rank_size=1)
        output = net(input_tensor)
        check_shape_value = (2, 4)
        self.assertEqual(True, operator.eq(output.shape, check_shape_value))
        logging.info('Finish test_layer_construct_with_low_rank_size.')

    def test_layer_construct_with_low_rank_w_init(self):
        logging.info('Start test_layer_construct_with_low_rank_size.')
        input_tensor = Tensor(numpy.ones((2, 4)), mstype.float32)
        net = LowRankAdapterLayer(hidden_size=4,
                                  reduction_factor=2,
                                  low_rank_w_init="xavier_uniform")
        output = net(input_tensor)
        check_shape_value = (2, 4)
        self.assertEqual(True, operator.eq(output.shape, check_shape_value))
        logging.info('Finish test_layer_construct_with_low_rank_w_init.')

    def test_layer_construct_with_non_linearity(self):
        logging.info('Start test_layer_construct_with_non_linearity.')
        input_tensor = Tensor(numpy.ones((2, 4)), mstype.float32)
        net = LowRankAdapterLayer(hidden_size=4,
                                  reduction_factor=2,
                                  non_linearity="gelu")
        output = net(input_tensor)
        check_shape_value = (2, 4)
        self.assertEqual(True, operator.eq(output.shape, check_shape_value))
        logging.info('Finish test_layer_construct_with_non_linearity.')

    def test_layer_construct_with_shape_bigger_than_2_input(self):
        logging.info(
            'Start test_layer_construct_with_shape_bigger_than_2_input')
        input_tensor = Tensor(numpy.ones((3, 2, 4)), mstype.float32)
        net = LowRankAdapterLayer(hidden_size=4,
                                  reduction_factor=2)
        output = net(input_tensor)
        check_shape_value = (3, 2, 4)
        self.assertEqual(True, operator.eq(output.shape, check_shape_value))
        logging.info(
            'Finish test_layer_construct_with_shape_bigger_than_2_input')

    def test_layer_shard_with_invalid_activation(self):
        logging.info('Start test_layer_shard_with_invalid_activation')
        with self.assertRaises(Exception) as ex:
            net = LowRankAdapterLayer(hidden_size=4,
                                      activation="logsoftmax",
                                      reduction_factor=2)
            net.shard(strategy_activation_org=((1, 1), (1, 1)))
        logging.error(ex.exception)
        logging.info('Finish test_layer_shard_with_invalid_activation')

    def test_layer_shard_with_invalid_type_strategy(self):
        logging.info('Start test_layer_shard_with_invalid_type_strategy')
        with self.assertRaises(Exception) as ex:
            net = LowRankAdapterLayer(hidden_size=4,
                                      reduction_factor=2)
            net.shard(strategy_matmul_org=1)
        logging.error(ex.exception)
        logging.info('Finish test_layer_shard_with_invalid_type_strategy')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()
    mindspore.set_context(device_id=args.device_id)
    pytest.main(["-s", os.path.abspath(__file__)])
