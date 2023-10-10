#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
import sys
sys.path.append('.')
import os
import unittest
import logging
import pytest
from mindpet.utils.entrance_monitor import EntranceMonitor
from mindpet.utils.constants import ENTRANCE_TYPE

logging.getLogger().setLevel(logging.INFO)


class TestEntranceMonitor(unittest.TestCase):
    def test_create_entrance_monitor_twice(self):
        """
        测试EntranceMonitor在连续构造两次实例对象的情况
        """
        logging.info('Start test_create_entrance_monitor_twice,')

        monitor1 = EntranceMonitor()
        monitor1.init()
        monitor1.set_value(ENTRANCE_TYPE, 'CLI')

        monitor2 = EntranceMonitor()
        result = monitor2.get_value(ENTRANCE_TYPE)

        self.assertEqual(result, 'CLI')

        logging.info('Finish test_create_entrance_monitor_twice.')


if __name__ == '__main__':
    pytest.main(['-s', os.path.abspath(__file__)])
