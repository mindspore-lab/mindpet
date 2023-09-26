#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2010-2022. All rights reserved.
import sys
sys.path.append('.')
import logging
import os
import shutil
import unittest
import pytest
import argparse
import mindspore

from mindpet.delta.prefix_layer import PrefixLayer
from mindpet.utils.constants import DEFAULT_MODES, DEFAULT_FLAGS

logging.getLogger().setLevel(logging.INFO)
LOCAL_PATH = os.path.join('/', 'tmp', 'ut').replace('\\', '/')
LOCAL_FILE = os.path.join(LOCAL_PATH, 'ut_sample.txt')


class TestPrefixLayer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.info('-----准备执行单元测试前置动作, 创建本地临时文件-----')
        if not os.path.exists(LOCAL_PATH):
            os.makedirs(LOCAL_PATH, exist_ok=True)

        if not os.path.exists(LOCAL_FILE):
            with os.fdopen(os.open(LOCAL_FILE, DEFAULT_FLAGS, DEFAULT_MODES), 'w+') as file:
                file.write('ut test sample')

    @classmethod
    def tearDownClass(cls):
        logging.info('-----单元测试执行完毕, 清除本地临时文件-----')
        if os.path.exists(LOCAL_PATH):
            shutil.rmtree(LOCAL_PATH)

        if os.path.exists(LOCAL_FILE):
            shutil.rmtree(LOCAL_FILE)

    def test_prefix_num_correct(self):
        logging.info('Start test_prefix_num_correct')
        prefix = PrefixLayer(prefix_token_num=10, batch_size=2, num_heads=512,
                             hidden_dim=512, embed_dim=512, mid_dim=512, dropout_rate=0.9)
        self.assertEqual(10, prefix.prefix_token_num)
        logging.info("Finish test_prefix_num_correct")

    def test_batch_size_correct(self):
        logging.info('Start test_batch_size_correct')
        prefix = PrefixLayer(prefix_token_num=10, batch_size=2, num_heads=512,
                             hidden_dim=512, embed_dim=512, mid_dim=512, dropout_rate=0.9)
        self.assertEqual(2, prefix.batch_size)
        logging.info("Finish test_batch_size_correct")

    def test_num_heads_correct(self):
        logging.info('Start test_num_heads_correct')
        prefix = PrefixLayer(prefix_token_num=10, batch_size=2, num_heads=512,
                             hidden_dim=512, embed_dim=512, mid_dim=512, dropout_rate=0.9)
        self.assertEqual(512, prefix.num_heads)
        logging.info("Finish test_num_heads_correct")

    def test_hidden_dim_correct(self):
        logging.info('Start test_hidden_dim_correct')
        prefix = PrefixLayer(prefix_token_num=10, batch_size=2, num_heads=512,
                             hidden_dim=512, embed_dim=512, mid_dim=512, dropout_rate=0.9)
        self.assertEqual(512, prefix.hidden_dim)
        logging.info("Finish test_hidden_dim_correct")

    def test_embed_dim_correct(self):
        logging.info('Start test_embed_dim_correct')
        prefix = PrefixLayer(prefix_token_num=10, batch_size=2, num_heads=512,
                             hidden_dim=512, embed_dim=512, mid_dim=512, dropout_rate=0.9)
        self.assertEqual(512, prefix.embed_dim)
        logging.info("Finish test_embed_dim_correct")

    def test_mid_dim_correct(self):
        logging.info('Start test_mid_dim_correct')
        prefix = PrefixLayer(prefix_token_num=10, batch_size=2, num_heads=512,
                             hidden_dim=512, embed_dim=512, mid_dim=512, dropout_rate=0.9)
        self.assertEqual(512, prefix.mid_dim)
        logging.info("Finish test_mid_dim_correct")

    def test_past_key_reparam_is_not_none(self):
        logging.info('Start test_past_key_reparam_is_not_none')
        prefix = PrefixLayer(prefix_token_num=10, batch_size=2, num_heads=512,
                             hidden_dim=512, embed_dim=512, mid_dim=512, dropout_rate=0.9)
        self.assertIsNotNone(prefix.past_key_reparam)
        logging.info("Finish test_past_key_reparam_is_not_none")

    def test_past_value_reparam_is_not_none(self):
        logging.info('Start test_past_value_reparam_is_not_none')
        prefix = PrefixLayer(prefix_token_num=10, batch_size=2, num_heads=512,
                             hidden_dim=512, embed_dim=512, mid_dim=512, dropout_rate=0.9)
        self.assertIsNotNone(prefix.past_value_reparam)
        logging.info("Finish test_past_value_reparam_is_not_none")

    def test_prefix_num_is_not_integer(self):
        logging.info("Start test_prefix_num_is_not_integer")
        self.assertRaises(TypeError, PrefixLayer, prefix_token_num=10.0, batch_size=2, num_heads=1024,
                          hidden_dim=1024, embed_dim=1024, mid_dim=1024, dropout_rate=0.1)
        logging.info("Finish test_prefix_num_is_not_integer")

    def test_prefix_num_is_not_positive_integer(self):
        logging.info("Start test_prefix_num_is_not_positive_integer")
        self.assertRaises(ValueError, PrefixLayer, prefix_token_num=-1, batch_size=2, num_heads=1024,
                          hidden_dim=1024, embed_dim=1024, mid_dim=1024, dropout_rate=0.1)
        logging.info("Finish test_prefix_num_is_not_positive_integer")

    def test_batch_size_is_not_integer(self):
        logging.info("Start test_batch_size_is_not_integer")
        self.assertRaises(ValueError, PrefixLayer, prefix_token_num=10, batch_size=-2, num_heads=1024,
                          hidden_dim=1024, embed_dim=1024, mid_dim=1024, dropout_rate=0.1)
        logging.info("Finish test_batch_size_is_not_integer")

    def test_batch_size_is_not_positive_integer(self):
        logging.info("Start test_batch_size_is_not_positive_integer")
        self.assertRaises(ValueError, PrefixLayer, prefix_token_num=10, batch_size=-2, num_heads=1024,
                          hidden_dim=1024, embed_dim=1024, mid_dim=1024, dropout_rate=0.1)
        logging.info("Finish test_batch_size_is_not_positive_integer")

    def test_num_heads_is_not_integer(self):
        logging.info("Start test_num_heads_is_not_integer")
        self.assertRaises(TypeError, PrefixLayer, prefix_token_num=10, batch_size=2, num_heads=1024.0,
                          hidden_dim=1024, embed_dim=1024, mid_dim=1024, dropout_rate=0.1)
        logging.info("Finish test_num_heads_is_not_integer")

    def test_num_heads_is_not_positive_integer(self):
        logging.info("Start test_num_heads_is_not_positive_integer")
        self.assertRaises(ValueError, PrefixLayer, prefix_token_num=10, batch_size=2, num_heads=-1024,
                          hidden_dim=1024, embed_dim=1024, mid_dim=1024, dropout_rate=0.1)
        logging.info("Finish test_num_heads_is_not_positive_integer")

    def test_hidden_dim_is_not_integer(self):
        logging.info("Start test_hidden_dim_is_not_integer")
        self.assertRaises(TypeError, PrefixLayer, prefix_token_num=10, batch_size=2, num_heads=1024,
                          hidden_dim=1024.0, embed_dim=1024, mid_dim=1024, dropout_rate=0.1)
        logging.info("Finish test_hidden_dim_is_not_integer")

    def test_hidden_dim_is_not_positive_integer(self):
        logging.info("Start test_hidden_dim_is_not_positive_integer")
        self.assertRaises(ValueError, PrefixLayer, prefix_token_num=10, batch_size=2, num_heads=1024,
                          hidden_dim=-1024, embed_dim=1024, mid_dim=1024, dropout_rate=0.1)
        logging.info("Finish test_hidden_dim_is_not_positive_integer")

    def test_embed_dim_is_not_integer(self):
        logging.info("Start test_embed_dim_is_not_integer")
        self.assertRaises(TypeError, PrefixLayer, prefix_token_num=10, batch_size=2, num_heads=1024,
                          hidden_dim=1024, embed_dim=1024.0, mid_dim=1024, dropout_rate=0.1)
        logging.info("Finish test_embed_dim_is_not_integer")

    def test_embed_dim_is_not_positive_integer(self):
        logging.info("Start test_embed_dim_is_not_positive_integer")
        self.assertRaises(ValueError, PrefixLayer, prefix_token_num=10, batch_size=2, num_heads=1024,
                          hidden_dim=1024, embed_dim=-1024, mid_dim=1024, dropout_rate=0.1)
        logging.info("Finish test_embed_dim_is_not_positive_integer")

    def test_mid_dim_is_not_integer(self):
        logging.info("Start test_mid_dim_is_not_integer")
        self.assertRaises(TypeError, PrefixLayer, prefix_token_num=10, batch_size=2, num_heads=1024,
                          hidden_dim=1024, embed_dim=1024, mid_dim=1024.0, dropout_rate=0.1)
        logging.info("Finish test_mid_dim_is_not_integer")

    def test_mid_dim_is_not_positive_integer(self):
        logging.info("Start test_mid_dim_is_not_positive_integer")
        self.assertRaises(ValueError, PrefixLayer, prefix_token_num=10, batch_size=2, num_heads=1024,
                          hidden_dim=1024, embed_dim=1024, mid_dim=-1024, dropout_rate=0.1)
        logging.info("Finish test_mid_dim_is_not_positive_integer")

    def test_dropout_rate_negative(self):
        logging.info("Start test_dropout_rate_is_zero")
        self.assertRaises(ValueError, PrefixLayer, prefix_token_num=10, batch_size=2, num_heads=1024,
                          hidden_dim=1024, embed_dim=1024, mid_dim=1024, dropout_rate=-0.1)
        logging.info("Finish test_dropout_rate_is_zero")

    def test_dropout_rate_is_one(self):
        logging.info("Start test_dropout_rate_scope")
        self.assertRaises(ValueError, PrefixLayer, prefix_token_num=10, batch_size=2, num_heads=1024,
                          hidden_dim=1024, embed_dim=1024, mid_dim=1024, dropout_rate=1.0)
        logging.info("Finish test_dropout_rate_scope")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()
    mindspore.set_context(device_id=args.device_id)
    pytest.main(["-s", os.path.abspath(__file__)])
