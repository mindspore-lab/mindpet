#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import sys
sys.path.append('.')
import os
import copy
import logging
from unittest import TestCase
import pytest
import mindpet.log.log as ac
from mindpet.log.log import validate_file_input_format

LOG_STD_INPUT_PARAM = {
    'to_std': True,
    'stdout_nodes': None,
    'stdout_devices': None,
    'stdout_level': 'WARNING'
}
LOG_FILE_INPUT_PARAM = {
    'file_level': ['INFO'],
    'file_save_dir': "~/.cache/Huawei/mxFoundationModel/log/",
    'append_rank_dir': False,
    'file_name': ['logger.INFO.log']
}
logging.getLogger().setLevel(logging.INFO)


class TestValidateInputFormat(TestCase):
    LOG_STD_INPUT_PARAM = None
    LOG_FILE_INPUT_PARAM = None

    def setUp(self) -> None:
        logging.info('Start test_model_config_check_init_with_none_task_object.')
        self.get_logger_std_input = copy.deepcopy(LOG_STD_INPUT_PARAM)
        self.get_logger_file_input = copy.deepcopy(LOG_FILE_INPUT_PARAM)

    def test_to_std(self):
        logging.info('Start test_to_std.')
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input['to_std'] = 0
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_std_input)
        logging.info('Finish test_to_std.')

    def stdout_nodes_or_device_test(self, key):
        logging.info('Start stdout_nodes_or_device_test.')
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = [0, 1]
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = (0, 1)
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = {'start': 0, 'end': 1}
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = '0'
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = ('0', '1')
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = {'start': 0}
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = {'start': '0', 'end': '1'}
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = {'start': 1, 'end': 0}
        ac.validate_std_input_format(**self.get_logger_std_input)
        logging.info('Finish stdout_nodes_or_device_test.')

    def test_stdout_nodes(self):
        logging.info('Start test_stdout_nodes.')
        self.stdout_nodes_or_device_test('stdout_nodes')
        logging.info('Finish test_stdout_nodes.')

    def test_stdout_devices(self):
        logging.info('Start test_stdout_devices.')
        self.stdout_nodes_or_device_test('stdout_devices')
        logging.info('Finish test_stdout_devices.')

    def level_test(self, key):
        logging.info('Start level_test.')
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = 'INFO'
        ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = 4
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = '0'
        with self.assertRaises(ValueError):
            ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = 5
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = -1
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_std_input)

        self.get_logger_std_input[key] = 3.14
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_std_input)
        logging.info('Finish level_test.')

    def file_level_test(self, key):
        logging.info('Start file_level_test.')
        ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input[key] = 'INFO'
        with self.assertRaises(TypeError):
            ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input[key] = 4
        with self.assertRaises(TypeError):
            ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input[key] = '0'
        with self.assertRaises(TypeError):
            ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input[key] = 5
        with self.assertRaises(TypeError):
            ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input[key] = -1
        with self.assertRaises(TypeError):
            ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input[key] = 3.14
        with self.assertRaises(TypeError):
            ac.validate_file_input_format(**self.get_logger_file_input)
        logging.info('Finish file_level_test.')

    def test_stdout_level(self):
        logging.info('Start test_stdout_level.')
        self.level_test('stdout_level')
        logging.info('Finish test_stdout_level.')

    def test_file_level(self):
        logging.info('Start test_file_level.')
        self.file_level_test('file_level')
        logging.info('Finish test_file_level.')

    def test_file_save_dir(self):
        logging.info('Start test_file_save_dir.')
        ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input['file_save_dir'] = ''
        ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input['file_save_dir'] = './'
        ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input['file_save_dir'] = 1
        with self.assertRaises(TypeError):
            ac.validate_file_input_format(**self.get_logger_file_input)
        logging.info('Finish test_file_save_dir.')

    def test_max_file_size(self):
        logging.info('Start test_max_file_size.')
        ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input['max_file_size'] = 3.14
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_file_input)
        logging.info('Finish test_max_file_size.')

    def test_max_num_of_files(self):
        logging.info('Start test_max_num_of_files.')
        ac.validate_file_input_format(**self.get_logger_file_input)

        self.get_logger_file_input['max_num_of_files'] = 3.14
        with self.assertRaises(TypeError):
            ac.validate_std_input_format(**self.get_logger_file_input)
        logging.info('Finish test_max_num_of_files.')

    def test_get_rank_info(self):
        logging.info('Start test_get_rank_info.')
        os.environ['RANK_ID'] = '2'
        os.environ['RANK_SIZE'] = '8'
        os.environ['DEVICE_ID'] = '1'
        os.environ['SERVER_ID'] = '0'

        rank_id, rank_size, server_id, device_id = ac.get_rank_info()
        self.assertEqual(rank_id, 2)
        self.assertEqual(rank_size, 8)
        self.assertEqual(server_id, 0)
        self.assertEqual(device_id, 1)

        os.environ.pop('RANK_ID')
        os.environ.pop('RANK_SIZE')
        os.environ.pop('SERVER_ID')
        os.environ.pop('DEVICE_ID')
        logging.info('Finish test_get_rank_info.')

    def test_convert_nodes_devices_input(self):
        logging.info('Start test_convert_nodes_devices_input.')
        var = None
        num = 4
        self.assertEqual(ac.convert_nodes_devices_input(var, num), (0, 1, 2, 3))

        var = {'start': 0, 'end': 4}
        self.assertEqual(ac.convert_nodes_devices_input(var, num), (0, 1, 2, 3))

        var = (0, 1)
        self.assertEqual(ac.convert_nodes_devices_input(var, num), (0, 1))

        var = [0, 1]
        self.assertEqual(ac.convert_nodes_devices_input(var, num), [0, 1])
        logging.info('Finish test_convert_nodes_devices_input.')

    def test_get_num_nodes_devices(self):
        logging.info('Start test_get_num_nodes_devices.')
        rank_size = 4
        num_nodes, num_devices = ac.get_num_nodes_devices(rank_size)
        self.assertEqual(num_nodes, 1)
        self.assertEqual(num_devices, 4)

        rank_size = 16
        num_nodes, num_devices = ac.get_num_nodes_devices(rank_size)
        self.assertEqual(num_nodes, 2)
        self.assertEqual(num_devices, 8)
        logging.info('Finish test_get_num_nodes_devices.')

    def test_check_list(self):
        logging.info('Start test_check_list.')
        var_name = 'stdout_nodes'
        list_var = [0, 1]
        num = 4
        ac.check_list(var_name, list_var, num)

        var_name = 'stdout_nodes'
        list_var = [0, 1, 2, 3]
        num = 2
        with self.assertRaises(ValueError):
            ac.check_list(var_name, list_var, num)
        logging.info('Finish test_check_list.')

    def test_generate_rank_list(self):
        logging.info('Start test_generate_rank_list.')
        stdout_nodes = [0, 1]
        stdout_devices = [0, 1]
        self.assertEqual(ac.generate_rank_list(stdout_nodes, stdout_devices), [0, 1, 8, 9])
        logging.info('Finish test_generate_rank_list.')


class TestValidate(TestCase):
    def test_input_validate_level_type(self):
        logging.info('Start test_input_validate_level_type.')
        with self.assertRaises(TypeError):
            ac.validate_level('std_out_level', 1)
        logging.info('Finish test_input_validate_level_type.')

    def test_input_validate_level_value(self):
        logging.info('Start test_input_validate_level_value.')
        with self.assertRaises(ValueError):
            ac.validate_level('std_out_level', "AA")
        logging.info('Finish test_input_validate_level_value.')

    def test_validate_file_input_len(self):
        logging.info('Start test_validate_file_input_len.')
        with self.assertRaises(ValueError):
            ac.validate_file_input_format(file_level=['INFO', 'ERROR'], file_name=['logger'],
                                          file_save_dir='/', append_rank_dir=False)
        logging.info('Finish test_validate_file_input_len.')

    def test_validate_file_input_type(self):
        logging.info('Start test_validate_file_input_type.')
        with self.assertRaises(TypeError):
            ac.validate_file_input_format(file_level='INFO', file_name=['logger'],
                                          file_save_dir='/', append_rank_dir=False)
            ac.validate_file_input_format(file_level='INFO', file_name='logger',
                                          file_save_dir='/', append_rank_dir=False)
        logging.info('Finish test_validate_file_input_type.')

    def test_judge_stdout_should_false(self):
        logging.info('Start test_judge_stdout_should_false.')
        self.assertFalse(ac.judge_stdout(0, 1, False))
        logging.info('Finish test_judge_stdout_should_false.')

    def test_judge_stdout_should_true(self):
        logging.info('Start test_judge_stdout_should_true.')
        self.assertTrue(ac.judge_stdout(0, 1, True, None, (0,)))
        logging.info('Finish test_judge_stdout_should_true.')

    def test_get_stream_handler(self):
        logging.info('Start test_get_stream_handler.')
        handler = ac.get_stream_handler(None, "INFO")
        self.assertIsInstance(handler, logging.Handler)
        logging.info('Finish test_get_stream_handler.')

    def test_validate_file_input_format_append_rank_type(self):
        logging.info('Start test_validate_file_input_format_append_rank_type.')
        with self.assertRaises(TypeError):
            validate_file_input_format(["INFO"], "./log", 0, ["test.log"])
        logging.info('Finish test_validate_file_input_format_append_rank_type.')

    def test_validate_file_input_format_file_name_type(self):
        logging.info('Start test_validate_file_input_format_file_name_type.')
        with self.assertRaises(TypeError):
            validate_file_input_format(["INFO"], "./log", False, "test.log")
        logging.info('Finish test_validate_file_input_format_file_name_type.')

    def test_validate_file_input_format_name_type_str(self):
        logging.info('Start test_validate_file_input_format_name_type_str.')
        with self.assertRaises(TypeError):
            validate_file_input_format(["INFO"], "./log", False, [0])
        logging.info('Finish test_validate_file_input_format_name_type_str.')

    def test_validate_file_input_format_file_name_prefix(self):
        logging.info('Start test_validate_file_input_format_file_name_prefix.')
        with self.assertRaises(ValueError):
            validate_file_input_format(["INFO"], "./log", False, ["/test.log"])
        logging.info('Finish test_validate_file_input_format_file_name_prefix.')

    def test_validate_file_input_format_filename_abs(self):
        logging.info('Start test_validate_file_input_format_filename_abs.')
        with self.assertRaises(ValueError):
            validate_file_input_format(["INFO"], "./log", False, ["../test.log"])
        logging.info('Finish test_validate_file_input_format_filename_abs.')


if __name__ == '__main__':
    pytest.main(['-s', os.path.abspath(__file__)])
