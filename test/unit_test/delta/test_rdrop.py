#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
import sys
sys.path.append('.')
import logging
import os
import unittest
import argparse
import mindspore
import numpy
import pytest
from mindspore import Tensor

from mindpet.delta.r_drop import rdrop_repeat, RDropLoss

logging.getLogger().setLevel(logging.INFO)

class TestRDrop(unittest.TestCase):
    def test_repeat_success(self):
        logging.info('Start test_repeat_success.')

        label_ids = Tensor(numpy.array([1, 2, 3, 4]))
        logits = Tensor(numpy.array([1, 2, 3, 4]))
        self.assertEqual(len(label_ids), 4)
        self.assertEqual(len(logits), 4)
        repeat_items = rdrop_repeat(label_ids, logits)
        for item in repeat_items:
            self.assertEqual(len(item), 8)
            self.assertTrue((item == Tensor(numpy.array([1, 1, 2, 2, 3, 3, 4, 4]))).asnumpy().all())
        logging.info('Finish test_repeat_success.')

    def test_init_rdrop(self):
        logging.info('Start test_init_rdrop.')

        label_ids = Tensor(numpy.array([1, 0, 4, 1, 0, 4]), mindspore.float32)
        logits = Tensor(numpy.random.randn(6), mindspore.float32)
        r_drop_loss = RDropLoss()
        return_loss = r_drop_loss.construct(logits, label_ids)
        self.assertIsInstance(return_loss, mindspore.Tensor)
        logging.info('Finish test_init_rdrop.')

    # 有bug
    # def test_alpha_valid(self):
    #     logging.info('Start test_alpha_valid.')

    #     label_ids = Tensor(numpy.array([1, 0, 4, 1, 0, 4]), mindspore.float32)
    #     logits = Tensor(numpy.random.randn(6), mindspore.float32)
    #     r_drop_loss = RDropLoss()
    #     with self.assertRaises(AssertionError) as ex:
    #         return_loss = r_drop_loss.construct(logits, label_ids, alpha='test')
    #     logging.info('Finish test_alpha_valid.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()
    mindspore.set_context(device_id=args.device_id)
    pytest.main(["-s", os.path.abspath(__file__)])
