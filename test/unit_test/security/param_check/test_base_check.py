#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
import sys
sys.path.append('.')
import os
import logging
import unittest
import pytest
from mindpet.security.param_check.base_check import BaseCheckParam, BaseCheck
from mindpet.utils.constants import DEFAULT_FLAGS, DEFAULT_MODES

logging.getLogger().setLevel(logging.INFO)


class TestBaseCheck(unittest.TestCase):
    path = None
    file_path = None

    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join('/', 'tmp', 'test_base_check')
        if not os.path.exists(cls.path):
            os.makedirs(cls.path)

        cls.file_path = os.path.join(cls.path, 'base_check_file.txt')
        with os.fdopen(os.open(cls.file_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('base_check_file_content.')

        os.chmod(cls.file_path, 0o750)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.file_path):
            os.remove(cls.file_path)

        if os.path.exists(cls.path):
            os.rmdir(cls.path)

    def test_base_check(self):
        """
        测试BaseCheck类在正常校验流程下的处理
        """
        base_check_param = BaseCheckParam(mode='750', path_including_file=True, force_quit=True, quiet=True)
        checker = BaseCheck(option_name='base_check', option_value=self.file_path)
        checker.check(base_check_param)
        self.assertEqual(checker.option_value, self.file_path)


if __name__ == '__main__':
    pytest.main(['-s', os.path.abspath(__file__)])
