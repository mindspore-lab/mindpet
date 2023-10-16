#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
test p-tuning v2
"""

import argparse
import logging
import os
import shutil
import unittest

import mindspore

import sys
sys.path.append(".")

import pytest

from mindpet.delta.ptuning2 import PrefixEncoder
from mindpet.utils.constants import DEFAULT_FLAGS, DEFAULT_MODES

logging.getLogger().setLevel(logging.INFO)
LOCAL_PATH = os.path.join("/", "tmp", "ut").replace("\\", "/")
LOCAL_FILE = os.path.join(LOCAL_PATH, "ut_sample.txt")


class TestPrefixEncoder(unittest.TestCase):
    """
    Test PrefixEncoder ut
    """

    @classmethod
    def setUpClass(cls):
        """
        setup
        """
        logging.info("-----准备执行单元测试前置动作, 创建本地临时文件-----")
        if not os.path.exists(LOCAL_PATH):
            os.makedirs(LOCAL_PATH, exist_ok=True)

        if not os.path.exists(LOCAL_FILE):
            with os.fdopen(
                os.open(LOCAL_FILE, DEFAULT_FLAGS, DEFAULT_MODES), "w+"
            ) as file:
                file.write("ut test sample")

    @classmethod
    def tearDownClass(cls):
        """
        tear down
        """
        logging.info("-----单元测试执行完毕, 清除本地临时文件-----")
        if os.path.exists(LOCAL_PATH):
            shutil.rmtree(LOCAL_PATH)

        if os.path.exists(LOCAL_FILE):
            shutil.rmtree(LOCAL_FILE)

    def test_pre_seq_len_correct(self):
        """
        test pre_seq_len correct
        """
        logging.info("Start test_pre_seq_len")
        prefix = PrefixEncoder(
            pre_seq_len=10,
            num_layers=2,
            num_heads=8,
            kv_channels=32,
            prefix_projection=False,
            projection_dim=32,
            dropout_prob=0.1,
        )
        self.assertEqual(10, prefix.pre_seq_len)
        logging.info("Finish test_pre_seq_len_correct")

    def test_num_layers_correct(self):
        """
        test num_layers correct
        """
        logging.info("Start test_num_layers_correct")
        prefix = PrefixEncoder(
            pre_seq_len=10,
            num_layers=2,
            num_heads=8,
            kv_channels=32,
            prefix_projection=False,
            projection_dim=32,
            dropout_prob=0.1,
        )
        self.assertEqual(2, prefix.num_layers)
        logging.info("Finish test_num_layers_correct")

    def test_num_heads_correct(self):
        """
        test num_heads correct
        """
        logging.info("Start test_num_heads_correct")
        prefix = PrefixEncoder(
            pre_seq_len=10,
            num_layers=2,
            num_heads=8,
            kv_channels=32,
            prefix_projection=False,
            projection_dim=32,
            dropout_prob=0.1,
        )
        self.assertEqual(8, prefix.num_heads)
        logging.info("Finish test_num_heads_correct")

    def test_kv_channels_correct(self):
        """
        test kv_channels correct
        """
        logging.info("Start test_kv_channels_correct")
        prefix = PrefixEncoder(
            pre_seq_len=10,
            num_layers=2,
            num_heads=8,
            kv_channels=32,
            prefix_projection=False,
            projection_dim=32,
            dropout_prob=0.1,
        )
        self.assertEqual(32, prefix.kv_channels)
        logging.info("Finish test_kv_channels_correct")

    def test_projection_dim_correct(self):
        """
        test projection_dim correct
        """
        logging.info("Start test_projection_dim_correct")
        prefix = PrefixEncoder(
            pre_seq_len=10,
            num_layers=2,
            num_heads=8,
            kv_channels=32,
            prefix_projection=True,
            projection_dim=32,
            dropout_prob=0.1,
        )
        self.assertEqual(32, prefix.projection_dim)
        logging.info("Finish test_projection_dim_correct")

    def test_pre_seq_len_is_not_positive_integer(self):
        """
        test pre_seq_len is not positive integer
        """
        logging.info("Start test_pre_seq_len_is_not_positive_integer")
        self.assertRaises(
            ValueError,
            PrefixEncoder,
            pre_seq_len=-1,
            num_layers=2,
            num_heads=8,
            kv_channels=32,
            prefix_projection=False,
            projection_dim=32,
            dropout_prob=0.1,
        )
        logging.info("Finish test_pre_seq_len_is_not_positive_integer")

    def test_num_layers_is_not_positive_integer(self):
        """
        test num_layers is not positive integer
        """
        logging.info("Start test_num_layers_is_not_positive_integer")
        self.assertRaises(
            ValueError,
            PrefixEncoder,
            pre_seq_len=10,
            num_layers=-1,
            num_heads=8,
            kv_channels=32,
            prefix_projection=False,
            projection_dim=32,
            dropout_prob=0.1,
        )
        logging.info("Finish test_num_layers_is_not_positive_integer")

    def test_num_heads_is_not_positive_integer(self):
        """
        test num_heads is not positive integer
        """
        logging.info("Start test_num_heads_is_not_positive_integer")
        self.assertRaises(
            ValueError,
            PrefixEncoder,
            pre_seq_len=10,
            num_layers=2,
            num_heads=-1,
            kv_channels=32,
            prefix_projection=False,
            projection_dim=32,
            dropout_prob=0.1,
        )
        logging.info("Finish test_num_heads_is_not_positive_integer")

    def test_kv_channels_is_not_positive_integer(self):
        """
        test kv_channels is not positive integer
        """
        logging.info("Start test_kv_channels_is_not_positive_integer")
        self.assertRaises(
            ValueError,
            PrefixEncoder,
            pre_seq_len=10,
            num_layers=2,
            num_heads=8,
            kv_channels=-1,
            prefix_projection=False,
            projection_dim=32,
            dropout_prob=0.1,
        )
        logging.info("Finish test_kv_channels_is_not_positive_integer")

    def test_projection_dim_is_not_positive_integer(self):
        """
        test projection_dim is not positive integer
        """
        logging.info("Start test_projection_dim_is_not_positive_integer")
        self.assertRaises(
            ValueError,
            PrefixEncoder,
            pre_seq_len=10,
            num_layers=2,
            num_heads=8,
            kv_channels=32,
            prefix_projection=True,
            projection_dim=-1,
            dropout_prob=0.1,
        )
        logging.info("Finish test_projection_dim_is_not_positive_integer")

    def test_dropout_prob_negative(self):
        """
        test dropout_prob is negative
        """
        logging.info("Start test_dropout_prob_is_negative")
        self.assertRaises(
            ValueError,
            PrefixEncoder,
            pre_seq_len=10,
            num_layers=2,
            num_heads=8,
            kv_channels=32,
            prefix_projection=False,
            projection_dim=32,
            dropout_prob=-0.1,
        )
        logging.info("Finish test_dropout_prob_is_negative")

    def test_dropout_prob_is_one(self):
        """
        test dropout_prob is one
        """
        logging.info("Start test_dropout_prob_scope")
        self.assertRaises(
            ValueError,
            PrefixEncoder,
            pre_seq_len=10,
            num_layers=2,
            num_heads=8,
            kv_channels=32,
            prefix_projection=False,
            projection_dim=32,
            dropout_prob=1.0,
        )
        logging.info("Finish test_dropout_prob_scope")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()
    mindspore.set_context(device_id=args.device_id)
    pytest.main(["-s", os.path.abspath(__file__)])
