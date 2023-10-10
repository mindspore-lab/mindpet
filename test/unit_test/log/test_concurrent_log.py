#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import sys
sys.path.append('.')
import os
import shutil
import time
import logging
from unittest import TestCase
import multiprocessing
import pytest

from mindpet.log.log import get_logger
from mindpet.log.log_utils import get_env_info

CONFIG_TEST_BASE_DIR = os.path.expanduser('~/.cache/Huawei/mxTuningKit/log')

logging.getLogger().setLevel(logging.INFO)


def single(device_id):
    os.environ['RANK_ID'] = str(device_id)
    os.environ['RANK_SIZE'] = '8'
    os.environ['DEVICE_ID'] = str(device_id)
    os.environ['SERVER_ID'] = '0'

    test_logger = get_logger(logger_name="test_logger", to_std=True, file_name=['test_logger.log'], file_level=['INFO'],
                             append_rank_dir=True)
    test_logger.info(f'device id is {device_id}')


class TestConcurrentLog(TestCase):
    @classmethod
    def clear_folder(cls):
        if os.path.exists(CONFIG_TEST_BASE_DIR):
            shutil.rmtree(CONFIG_TEST_BASE_DIR, ignore_errors=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.clear_folder()

    def test_create_file(self):
        logging.info('Start test_create_file.')
        for device_id in range(8):
            cur_process = multiprocessing.Process(target=single, args=(device_id,))
            cur_process.start()

        time.sleep(0.5)
        log_dir_file_list = os.listdir(CONFIG_TEST_BASE_DIR)
        self.assertIn('node_0', log_dir_file_list)
        device_dirs = os.path.join(CONFIG_TEST_BASE_DIR, 'node_0')
        device_file_list = os.listdir(device_dirs)
        for device_id in range(8):
            self.assertIn(f'device_{device_id}', device_file_list)
        logging.info('Finish test_create_file.')

    def test_log_record(self):
        logging.info('Start test_log_record.')
        for device_id in range(8):
            cur_process = multiprocessing.Process(target=single, args=(device_id,))
            cur_process.start()
        time.sleep(0.5)

        device_dirs = os.path.join(CONFIG_TEST_BASE_DIR, 'node_0')
        for device_id in range(8):
            log_dir = os.path.join(device_dirs, f'device_{device_id}')
            with open(os.path.join(log_dir, 'test_logger.log'), 'r') as log_file:
                content = log_file.read()
                self.assertIn(f'device id is {device_id}', content)
        logging.info('Finish test_log_record.')

    def test_logger_should_to_std(self):
        logging.info('Start test_logger_should_to_std.')
        os.environ['RANK_ID'] = '0'
        os.environ['RANK_SIZE'] = '8'
        os.environ['DEVICE_ID'] = '0'
        os.environ['SERVER_ID'] = '0'

        logger_should_to_std = get_logger(logger_name="logger_should_to_std", to_std=True,
                                          file_name=['logger_should_to_std.log'],
                                          file_level=['INFO'],
                                          append_rank_dir=True)
        logger_should_to_std.set_logger()
        self.assertEqual(len(logger_should_to_std.handlers), 2)
        logging.info('Finish test_logger_should_to_std.')

    def test_logger_should_not_to_std(self):
        logging.info('Start test_logger_should_not_to_std.')
        os.environ['RANK_ID'] = '1'
        os.environ['RANK_SIZE'] = '8'
        os.environ['DEVICE_ID'] = '1'
        os.environ['SERVER_ID'] = '0'

        logger_should_not_to_std = get_logger(logger_name="logger_should_not_to_std", to_std=True,
                                              file_name=['logger_should_not_to_std.log'],
                                              file_level=['INFO'],
                                              append_rank_dir=True)
        logger_should_not_to_std.set_logger()
        self.assertEqual(len(logger_should_not_to_std.handlers), 1)
        logging.info('Finish test_logger_should_not_to_std.')

    def test_file_handler_not_same(self):
        logging.info('Start test_file_handler_not_same.')

        handler_list = list()
        for device_id in range(8):
            os.environ['RANK_ID'] = str(device_id)
            os.environ['RANK_SIZE'] = '8'
            os.environ['DEVICE_ID'] = str(device_id)
            os.environ['SERVER_ID'] = '0'

            test_logger = get_logger(logger_name=f"test_logger_{device_id}", to_std=True, file_name=['test_logger.log'],
                                     file_level=['INFO'],
                                     append_rank_dir=True)
            test_logger.set_logger()
            handler_list.extend(test_logger.handlers)
        time.sleep(0.5)
        self.assertEqual(len(handler_list), 9)
        logging.info('Finish test_file_handler_not_same.')

    def test_get_env_info_less_than_zero(self):
        logging.info('Start test_get_env_info_less_than_zero.')
        os.environ['RANK_ID'] = '-1'
        self.assertEqual(get_env_info('RANK_ID', 0), 0)
        logging.info('Finish test_get_env_info_less_than_zero.')

    def test_get_env_info_illegal(self):
        logging.info('Start test_get_env_info_illegal.')
        os.environ['RANK_ID'] = 'abc'
        self.assertEqual(get_env_info('RANK_ID', 0), 0)
        logging.info('Finish test_get_env_info_illegal.')


if __name__ == '__main__':
    pytest.main(['-s', os.path.abspath(__file__)])
