#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import sys
sys.path.append('.')
import os
import shutil
import logging
from unittest import TestCase, mock

import pytest
from mindpet.log.log import get_logger, get_file_path_list, set_logger_property, CustomizedRotatingFileHandler, \
    LOG_RECORD_MAX_LEN
from mindpet.log.log_utils import log_args_black_list_characters_replace, wrap_local_working_directory, const
from mindpet.utils.exceptions import MakeDirError, UnsupportedPlatformError, PathOwnerError, PathModeError
from mindpet.utils.constants import DEFAULT_MAX_LOG_FILE_NUM, EMPTY_STRING

MODE740 = 0o740

CONFIG_TEST_BASE_DIR = os.path.expanduser('~/.cache/Huawei/mxTuningKit/log')
CONFIG_TEST_NODE_DIR = os.path.realpath(os.path.join(CONFIG_TEST_BASE_DIR, 'node_0'))
CONFIG_TEST_LOG_DIR = os.path.realpath(os.path.join(CONFIG_TEST_NODE_DIR, 'device_0'))

LOG_MESSAGE = {
    'debug': 'debug message',
    'info': 'info message',
    'warning': 'warning message',
    'error': 'error message',
    'critical': 'critical message'
}
logging.getLogger().setLevel(logging.INFO)


class TestLogger(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_path = './temp_path'
        cls.full_path = "test"

    def test_logger_level(self):
        logging.info('Start test_logger_level.')
        logger = get_logger("logger", file_level=['INFO', 'ERROR'])
        with self.assertLogs(logger) as console:
            logger.info("A test info message")
            logger.error("A test error message")
        self.assertEqual(console.output, ['INFO:logger:A test info message', 'ERROR:logger:A test error message'])
        logging.info('Finish test_logger_level.')

    def test_logger_to_std(self):
        logging.info('Start test_logger_to_std.')
        logger = get_logger("logger", file_level=['INFO', 'ERROR'])
        with self.assertLogs(logger) as console:
            try:
                raise ValueError("value error")
            except Exception as exception:
                logger.error(exception)
        self.assertEqual(console.output, ['ERROR:logger:value error'])
        logging.info('Finish test_logger_to_std.')

    def test_get_logger_twice_should_same(self):
        logging.info('Start test_get_logger_twice_should_same.')
        logger = get_logger("logger", file_level=['INFO', 'ERROR'])
        twice = get_logger("logger")
        self.assertEqual(twice, logger)
        logging.info('Finish test_get_logger_twice_should_same.')

    def test_get_logger_diff(self):
        logging.info('Start test_get_logger_diff.')
        logger = get_logger("logger_diff", file_level=['INFO', 'ERROR'])
        other = get_logger()
        self.assertNotEqual(other, logger)
        logging.info('Finish test_get_logger_diff.')

    def test_same_file_hanlder(self):
        logging.info('Start test_same_file_hanlder.')
        service_logger = get_logger('test_same_file_hanlder', to_std=True, file_name=['test_same_file_hanlder.log'],
                                    file_level=['INFO'],
                                    append_rank_dir=False)
        service_logger.set_logger()
        service_logger_without_std = get_logger('test_same_file_hanlder_without_std', to_std=False,
                                                file_name=['test_same_file_hanlder.log'],
                                                file_level=['INFO'], append_rank_dir=False)
        service_logger_without_std.set_logger()
        self.assertEqual(len(service_logger.handlers), 2)
        self.assertEqual(len(service_logger_without_std.handlers), 1)
        self.assertIn(service_logger_without_std.handlers[0], service_logger.handlers)
        logging.info('Finish test_same_file_hanlder.')

    def test_log_args_black_list_characters_replace(self):
        logging.info('Start test_log_args_black_list_characters_replace.')
        args = list()
        args.append("\r")
        args.append("\n")
        args.append("\t")
        args.append("\f")
        args.append("\v")
        args.append("\b")
        args.append("\u000A")
        args.append("\u000C")
        args.append("\u000B")
        args.append("\u0008")
        args.append("\u0009")
        args.append("\u007F")
        replace_args = log_args_black_list_characters_replace(args)
        for arg in replace_args:
            self.assertEqual(arg, "")
        logging.info('Finish test_log_args_black_list_characters_replace.')

    def test_log_content_sapce_check(self):
        logging.info('Start test_log_content_sapce_check.')
        args = " " * 4
        replace_args = log_args_black_list_characters_replace(args)
        self.assertEqual(replace_args, " ")
        logging.info('Finish test_log_content_sapce_check.')

    def test_wrap_local_working_directory_none(self):
        logging.info('Start test_wrap_local_working_directory_none.')
        with self.assertRaises(ValueError):
            wrap_local_working_directory(None)
        logging.info('Finish test_wrap_local_working_directory_none.')

    def test_wrap_local_working_directory_empty(self):
        logging.info('Start test_wrap_local_working_directory_empty.')
        with self.assertRaises(ValueError):
            wrap_local_working_directory("")
        logging.info('Finish test_wrap_local_working_directory_empty.')

    def test_wrap_local_working_directory_home_path_empty(self):
        logging.info('Start test_wrap_local_working_directory_home_path_empty.')
        os.environ['HOME'] = ""
        temp_path = self.temp_path

        with self.assertRaises(ValueError):
            wrap_local_working_directory(temp_path)
        logging.info('Finish test_wrap_local_working_directory_home_path_empty.')

    def test_const_setter(self):
        logging.info('Start test_const_setter.')
        with self.assertRaises(ValueError):
            const.LEV = "INFO"
        logging.info('Finish test_const_setter.')

    def test_wrap_local_working_directory_success(self):
        logging.info('Start test_wrap_local_working_directory_success.')
        temp_path = self.temp_path
        os.environ['HOME'] = temp_path
        os.makedirs(temp_path, exist_ok=True, mode=MODE740)
        full_path = self.full_path
        working_directory = wrap_local_working_directory(full_path)
        self.assertTrue(os.path.exists(working_directory))
        shutil.rmtree(temp_path)
        logging.info('Finish test_wrap_local_working_directory_success.')

    def test_wrap_local_working_directory_with_config(self):
        logging.info('Start test_wrap_local_working_directory_with_config.')
        temp_path = self.temp_path
        os.environ['HOME'] = temp_path
        os.makedirs(temp_path, exist_ok=True, mode=MODE740)
        full_path = self.full_path
        path_config = {"path": "test/temp", "rule": MODE740}
        working_directory = wrap_local_working_directory(full_path, specific_path_config=path_config)
        self.assertTrue(os.path.exists(working_directory))
        shutil.rmtree(temp_path)
        logging.info('Finish test_wrap_local_working_directory_with_config.')

    def test_wrap_local_working_directory_with_config_file_exist(self):
        logging.info('Start test_wrap_local_working_directory_with_config_file_exist.')
        temp_path = self.temp_path
        os.environ['HOME'] = temp_path
        os.makedirs(temp_path, exist_ok=True, mode=MODE740)
        full_path = self.full_path
        path_config = {"path": "test/temp", "rule": MODE740}
        os.makedirs("./temp_path/.cache/Huawei/mxTuningKit/test/temp/test", exist_ok=True)
        working_directory = wrap_local_working_directory(full_path, specific_path_config=path_config)
        self.assertTrue(os.path.exists(working_directory))
        shutil.rmtree(temp_path)
        logging.info('Finish test_wrap_local_working_directory_with_config_file_exist.')

    def test_wrap_local_working_directory_with_config_illegal(self):
        logging.info('Start test_wrap_local_working_directory_with_config_illegal.')
        temp_path = self.temp_path
        os.environ['HOME'] = temp_path
        os.makedirs(temp_path, exist_ok=True, mode=MODE740)
        full_path = self.full_path
        path_config = {"path": "test"}
        with self.assertRaises(ValueError):
            working_directory = wrap_local_working_directory(full_path, specific_path_config=path_config)

        path_config = {"rule": MODE740}
        with self.assertRaises(ValueError):
            working_directory = wrap_local_working_directory(full_path, specific_path_config=path_config)

        shutil.rmtree(temp_path)
        logging.info('Finish test_wrap_local_working_directory_with_config_illegal.')

    @mock.patch("platform.system")
    def test_wrap_local_working_directory_platfrom_not_linux(self, mock_func):
        logging.info('Start test_wrap_local_working_directory_platfrom_not_linux.')
        temp_path = self.temp_path
        os.environ['HOME'] = temp_path
        os.makedirs(temp_path, exist_ok=True, mode=MODE740)
        full_path = self.full_path

        mock_func.return_value = "windows"
        with self.assertRaises(UnsupportedPlatformError):
            working_directory = wrap_local_working_directory(full_path)
        shutil.rmtree(temp_path)
        logging.info('Finish test_wrap_local_working_directory_platfrom_not_linux.')

    @mock.patch("os.makedirs")
    def test_wrap_local_working_directory_full_path_fail(self, mock_func):
        logging.info('Start test_wrap_local_working_directory_full_path_fail.')
        temp_path = self.temp_path
        os.environ['HOME'] = temp_path
        os.mkdir(temp_path, mode=MODE740)
        full_path = self.full_path
        mock_func.side_effect = RuntimeError
        with self.assertRaises(MakeDirError):
            working_directory = wrap_local_working_directory(full_path)
        shutil.rmtree(temp_path)
        logging.info('Finish test_wrap_local_working_directory_full_path_fail.')

    @mock.patch("os.makedirs")
    def test_wrap_local_working_directory_full_path_with_config_fail(self, mock_func):
        logging.info('Start test_wrap_local_working_directory_full_path_with_config_fail.')
        temp_path = self.temp_path
        full_path = self.full_path
        os.environ['HOME'] = temp_path
        os.mkdir(temp_path, mode=MODE740)
        os.mkdir("./temp_path/.cache", mode=MODE740)
        os.mkdir("./temp_path/.cache/Huawei", mode=MODE740)
        os.mkdir("./temp_path/.cache/Huawei/mxTuningKit", mode=MODE740)
        os.makedirs("./temp_path/.cache/Huawei/mxTuningKit", exist_ok=True, mode=MODE740)
        path_config = {"path": "test/temp", "rule": MODE740}
        mock_func.side_effect = RuntimeError
        with self.assertRaises(MakeDirError):
            working_directory = wrap_local_working_directory(full_path, specific_path_config=path_config)
        shutil.rmtree(temp_path)
        logging.info('Finish test_wrap_local_working_directory_full_path_with_config_fail.')

    @mock.patch("os.makedirs")
    def test_wrap_local_working_directory_full_path_with_wrap_fail(self, mock_func):
        logging.info('Start test_wrap_local_working_directory_full_path_with_wrap_fail.')
        temp_path = self.temp_path
        full_path = self.full_path
        os.environ['HOME'] = temp_path
        os.mkdir(temp_path, mode=MODE740)
        os.mkdir("./temp_path/.cache", mode=MODE740)
        os.mkdir("./temp_path/.cache/Huawei", mode=MODE740)
        os.mkdir("./temp_path/.cache/Huawei/mxTuningKit", mode=MODE740)
        os.mkdir("./temp_path/.cache/Huawei/mxTuningKit/test", mode=MODE740)
        os.mkdir("./temp_path/.cache/Huawei/mxTuningKit/test/temp", mode=MODE740)
        path_config = {"path": "test/temp", "rule": MODE740}
        mock_func.side_effect = RuntimeError
        with self.assertRaises(MakeDirError):
            working_directory = wrap_local_working_directory(full_path, specific_path_config=path_config)
        shutil.rmtree(temp_path)
        logging.info('Finish test_wrap_local_working_directory_full_path_with_wrap_fail.')

    def test_get_file_path_list_base_dir_none(self):
        logging.info('Start test_get_file_path_list_base_dir_none.')
        const.get_local_default_log_file_dir()
        base_save_dir = os.path.expanduser(const.local_default_log_file_dir)
        file_path = os.path.join(base_save_dir, "test.log")
        path_list = get_file_path_list(base_save_dir=None, append_rank_dir=False, server_id=0, rank_id=0,
                                       file_name=["test.log"])
        self.assertIn(file_path, path_list)
        logging.info('Finish test_get_file_path_list_base_dir_none.')

    @mock.patch("mindpet.log.log.check_link_path")
    def test_get_file_path_list_base_dir_link_path(self, mock_func):
        logging.info('Start test_get_file_path_list_base_dir_link_path.')
        mock_func.return_value = True
        const.get_local_default_log_file_dir()
        base_save_dir = os.path.expanduser(const.local_default_log_file_dir)
        file_path = os.path.join(base_save_dir, "node_0/device_0/test.log")
        path_list = get_file_path_list(base_save_dir=None, append_rank_dir=True, server_id=0, rank_id=0,
                                       file_name=["test.log"])
        self.assertIn(file_path, path_list)
        logging.info('Finish test_get_file_path_list_base_dir_link_path.')

    def test_logger_source(self):
        logging.info('Start test_logger_source.')
        service_logger = get_logger('service', to_std=True, file_name=['service.log'], file_level=['INFO'],
                                    append_rank_dir=False)
        set_logger_property("TEST")
        service_logger.info(LOG_MESSAGE.get('info'))

        log_dir_file_list = os.listdir(CONFIG_TEST_BASE_DIR)
        self.assertIn('service.log', log_dir_file_list)

        with open(os.path.join(CONFIG_TEST_BASE_DIR, 'service.log'), 'r') as log_file:
            content = log_file.read()
            res_str_prefix = "[unknown] [None] : "
            self.assertIn(res_str_prefix + LOG_MESSAGE.get('info'), content)
        logging.info('Finish test_logger_source.')

    def test_logger_content_length(self):
        logging.info('Start test_logger_content_length.')
        service_logger = get_logger('service', to_std=False, file_name=['service.log'], file_level=['INFO'],
                                    append_rank_dir=False)
        info_mes = "*" * LOG_RECORD_MAX_LEN
        imfo_double = info_mes * 2
        service_logger.info(imfo_double)

        log_dir_file_list = os.listdir(CONFIG_TEST_BASE_DIR)
        self.assertIn('service.log', log_dir_file_list)

        with open(os.path.join(CONFIG_TEST_BASE_DIR, 'service.log'), 'r') as log_file:
            content = log_file.read()
            real_content = content.split(" ")[-1].replace("\n", EMPTY_STRING)
            self.assertTrue(len(real_content) <= LOG_RECORD_MAX_LEN)
        logging.info('Finish test_logger_content_length.')

    def test_do_chmod_lock_file(self):
        logging.info('Start test_do_chmod_lock_file.')

        class TestHandler(CustomizedRotatingFileHandler):

            def test_method(self):
                self._do_chmod(None)

            def _do_chmod(self, mode):
                super(TestHandler, self)._do_chmod(mode)

        handler = TestHandler(filename="test.log")
        handler.test_method()

        logging.info('Finish test_do_chmod_lock_file.')

    def test_logger_do_rollover(self):
        logging.info('Start test_logger_do_rollover.')
        service_logger = get_logger('test', to_std=False, file_name=['service.log'], file_level=['INFO'],
                                    append_rank_dir=True, max_file_size=0.0001)
        for _ in range(100):
            service_logger.info("test")
        log_dir_file_list = os.listdir(CONFIG_TEST_LOG_DIR)
        for num in range(DEFAULT_MAX_LOG_FILE_NUM):
            if num == 0:
                self.assertIn('service.log', log_dir_file_list)
            else:
                self.assertIn(f'service.log.{num}', log_dir_file_list)
        shutil.rmtree(CONFIG_TEST_LOG_DIR)
        logging.info('Finish test_logger_do_rollover.')

    def test_logger_do_rollover_file_num(self):
        logging.info('Start test_logger_do_rollover_file_num.')
        temp_log_file = "test.log"
        handler = CustomizedRotatingFileHandler(filename=temp_log_file, backupCount=-1)
        handler.doRollover()
        self.assertTrue(os.path.exists(temp_log_file))

        os.remove(temp_log_file)
        logging.info('Finish test_logger_do_rollover_file_num.')


class TestGetLogger(TestCase):

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(CONFIG_TEST_BASE_DIR):
            shutil.rmtree(CONFIG_TEST_BASE_DIR, ignore_errors=True)

    def test_get_logger(self):
        logging.info('Start test_get_logger.')
        logger = get_logger('test_logger', file_name=['test_logger.INFO.log', 'test_logger.ERROR.log'],
                            file_level=['INFO', 'ERROR'])
        logger.debug(LOG_MESSAGE.get('debug'))
        logger.info(LOG_MESSAGE.get('info'))
        logger.warning(LOG_MESSAGE.get('warning'))
        logger.error(LOG_MESSAGE.get('error'))
        logger.critical(LOG_MESSAGE.get('critical'))

        log_dir_file_list = os.listdir(CONFIG_TEST_LOG_DIR)
        self.assertIn('test_logger.ERROR.log', log_dir_file_list)
        self.assertIn('test_logger.INFO.log', log_dir_file_list)

        with open(os.path.join(CONFIG_TEST_LOG_DIR, 'test_logger.INFO.log'), 'r') as log_file:
            content = log_file.read()
            self.assertNotIn(LOG_MESSAGE.get('debug'), content)
            self.assertIn(LOG_MESSAGE.get('info'), content)
            self.assertIn(LOG_MESSAGE.get('warning'), content)
            self.assertIn(LOG_MESSAGE.get('error'), content)
            self.assertIn(LOG_MESSAGE.get('critical'), content)

        with open(os.path.join(CONFIG_TEST_LOG_DIR, 'test_logger.ERROR.log'), 'r') as log_file:
            content = log_file.read()
            self.assertNotIn(LOG_MESSAGE.get('debug'), content)
            self.assertNotIn(LOG_MESSAGE.get('info'), content)
            self.assertNotIn(LOG_MESSAGE.get('warning'), content)
            self.assertIn(LOG_MESSAGE.get('error'), content)
            self.assertIn(LOG_MESSAGE.get('critical'), content)
        logging.info('Finish test_get_logger.')

    def test_get_logger_without_std(self):
        logging.info('Start test_get_logger_without_std.')
        service_logger_without_std = get_logger('service_logger_without_std', to_std=False, file_name=['service.log'],
                                                file_level=['INFO'], append_rank_dir=False)
        service_logger_without_std.debug(LOG_MESSAGE.get('debug'))
        service_logger_without_std.info(LOG_MESSAGE.get('info'))
        service_logger_without_std.warning(LOG_MESSAGE.get('warning'))
        service_logger_without_std.error(LOG_MESSAGE.get('error'))
        service_logger_without_std.critical(LOG_MESSAGE.get('critical'))

        log_dir_file_list = os.listdir(CONFIG_TEST_BASE_DIR)
        self.assertIn('service.log', log_dir_file_list)

        with open(os.path.join(CONFIG_TEST_BASE_DIR, 'service.log'), 'r') as log_file:
            content = log_file.read()
            self.assertNotIn(LOG_MESSAGE.get('debug'), content)
            self.assertIn(LOG_MESSAGE.get('info'), content)
            self.assertIn(LOG_MESSAGE.get('warning'), content)
            self.assertIn(LOG_MESSAGE.get('error'), content)
            self.assertIn(LOG_MESSAGE.get('critical'), content)
        logging.info('Finish test_get_logger_without_std.')

    def test_get_service_logger(self):
        logging.info('Start test_get_service_logger.')
        service_logger = get_logger('service', to_std=True, file_name=['service.log'], file_level=['INFO'],
                                    append_rank_dir=False)
        service_logger.debug(LOG_MESSAGE.get('debug'))
        service_logger.info(LOG_MESSAGE.get('info'))
        service_logger.warning(LOG_MESSAGE.get('warning'))
        service_logger.error(LOG_MESSAGE.get('error'))
        service_logger.critical(LOG_MESSAGE.get('critical'))

        log_dir_file_list = os.listdir(CONFIG_TEST_BASE_DIR)
        self.assertIn('service.log', log_dir_file_list)

        with open(os.path.join(CONFIG_TEST_BASE_DIR, 'service.log'), 'r') as log_file:
            content = log_file.read()
            self.assertNotIn(LOG_MESSAGE.get('debug'), content)
            self.assertIn(LOG_MESSAGE.get('info'), content)
            self.assertIn(LOG_MESSAGE.get('warning'), content)
            self.assertIn(LOG_MESSAGE.get('error'), content)
            self.assertIn(LOG_MESSAGE.get('critical'), content)
        logging.info('Finish test_get_service_logger.')

    def test_get_stream_handler(self):
        logging.info('Start test_get_stream_handler.')
        std_format = '[prefix-- %(levelname)s] : %(message)s'
        logger = get_logger("tk_logger", file_level=['INFO', 'ERROR'], stdout_format=std_format)
        with self.assertLogs(logger) as console:
            try:
                raise ValueError("value error")
            except Exception as exception:
                logger.error(exception)
        logging.info('Finish test_get_stream_handler.')


if __name__ == '__main__':
    pytest.main(['-s', os.path.abspath(__file__)])
