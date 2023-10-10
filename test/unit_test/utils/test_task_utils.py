#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
import sys
sys.path.append('.')
import os
import logging
import shutil
import subprocess
import unittest
import unittest.mock as mock
import click
import pytest
from mindpet.utils.exceptions import MakeDirError
from mindpet.utils.task_utils import timeout_monitor, create_output_path_subdir_with_uuid, model_config_keys_check_item, \
    model_config_params_check_item, extend_model_config_freeze_command, handle_exception_log

logging.getLogger().setLevel(logging.INFO)


class TestTaskUtils(unittest.TestCase):
    @mock.patch('os.killpg')
    def test_timeout_monitor(self, mock_func):
        """
        测试超时机制是否能够有效终止子进程作业
        :param mock_func: mock方法
        """
        logging.info('Start test_timeout_monitor.')

        mock_func.side_effect = ProcessLookupError

        # task_file_name中的任务执行时间为20s
        task_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resource/timeout_monitor_task.py')

        # 组装子任务subprocess进程对象
        cmd = ['python', task_file_name]
        process = subprocess.Popen(cmd, env=os.environ, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=False)

        # 超时机制限制2s
        timeout_monitor(task_name='sub_task', task_process=process, timeout=2)

        logging.info('Finish test_timeout_monitor.')

    @mock.patch('uuid.uuid4')
    def test_create_output_path_subdir_with_duplicate_uuid(self, mock_func):
        """
        测试创建包含uuid的output_path时出现重名时的处理
        :param mock_func: mock方法
        """
        logging.info('Start test_create_output_path_subdir_with_duplicate_uuid.')

        mock_func.side_effect = ['uuid1', 'uuid2']

        output_path = os.path.join('/', 'tmp', 'task_utils', 'outputs')

        duplicate_output_full_path = os.path.join(output_path, 'MINDPET_UUID1')
        if not os.path.exists(duplicate_output_full_path):
            os.makedirs(duplicate_output_full_path)

        result_path = create_output_path_subdir_with_uuid(output_path)

        self.assertIn('UUID2', result_path)

        if os.path.exists(duplicate_output_full_path):
            shutil.rmtree(duplicate_output_full_path)

        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        logging.info('Finish test_create_output_path_subdir_with_duplicate_uuid.')

    @mock.patch('os.makedirs')
    def test_create_output_path_subdir_with_uuid_when_makedirs_throw_exception(self, mock_func):
        """
        测试生成output_path子文件夹时出现Exception类异常的情况
        :param mock_func: mock 方法
        """
        logging.info('Start test_create_output_path_subdir_with_uuid_when_makedirs_throw_exception.')

        mock_func.side_effect = RuntimeError

        output_path = os.path.join('/', 'tmp', 'task_utils', 'outputs')

        with self.assertRaises(MakeDirError):
            create_output_path_subdir_with_uuid(output_path)

        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        logging.info('Finish test_create_output_path_subdir_with_uuid_when_makedirs_throw_exception.')

    @mock.patch('mindpet.utils.task_utils.ModelConfigKeysInfoError')
    def test_model_config_keys_check_item_throw_attribute_error(self, mock_func):
        """
        测试model_config配置信息中的首层key名称合法性时， 出现content为None触发AttributeError的问题
        :param mock_func: mock方法
        """
        logging.info('Start test_model_config_keys_check_item_throw_attribute_error.')

        mock_func.side_effect = AttributeError

        with self.assertRaises(AttributeError):
            model_config_keys_check_item(content=None, config_keys='params')

        logging.info('Finish test_model_config_keys_check_item_throw_attribute_error.')

    def test_model_config_params_check_item_with_no_params_config(self):
        """
        测试model_config中params内容校验方法, 在接收不到params部分内容的情况
        """
        logging.info('Start test_model_config_params_check_item_with_no_params_config.')
        model_config_params_check_item(task_object=None, content={'freeze': ['modules']})
        logging.info('Finish test_model_config_params_check_item_with_no_params_config.')

    def test_extend_model_config_freeze_command_with_no_freeze_config(self):
        """
        测试探测model_config中包含freeze方法时扩充命令行参数时, 接收不到freeze参数的情况
        """
        logging.info('Start test_extend_model_config_freeze_command_with_no_freeze_config.')
        extend_model_config_freeze_command(task_object=None, model_config_path=None, content={'params': {'lr': '1e-4'}})
        logging.info('Finish test_extend_model_config_freeze_command_with_no_freeze_config.')

    def test_handle_exception_log_with_click_abort_exception(self):
        """
        测试handle_exception_log方法接收click.exceptions.Abort异常时的处理
        """
        logging.info('Start test_handle_exception_log_with_click_abort_exception.')
        exception_object = click.exceptions.Abort()
        handle_exception_log(exception_object)
        logging.info('Finish test_handle_exception_log_with_click_abort_exception.')

    def test_handle_exception_log_with_click_no_such_option_exception(self):
        """
        测试handle_exception_log方法接收click.exceptions.NoSuchOption异常时的处理
        """
        logging.info('Start test_handle_exception_log_with_click_no_such_option_exception.')
        exception_object = click.exceptions.NoSuchOption(option_name='option_name')
        handle_exception_log(exception_object)
        logging.info('Finish test_handle_exception_log_with_click_no_such_option_exception.')

    def test_handle_exception_log_with_exception(self):
        """
        测试handle_exception_log方法接收Exception异常(无错误信息)时的处理
        """
        logging.info('Start test_handle_exception_log_with_exception.')
        exception_object = RuntimeError()
        handle_exception_log(exception_object)
        logging.info('Finish test_handle_exception_log_with_exception.')

    def test_handle_exception_log_with_exception_containing_err_code(self):
        """
        测试handle_exception_log方法接收Exception异常(包含错误码)时的处理
        """
        logging.info('Start test_handle_exception_log_with_exception_containing_err_code.')
        exception_object = RuntimeError(1)
        handle_exception_log(exception_object)
        logging.info('Finish test_handle_exception_log_with_exception_containing_err_code.')

    def test_handle_exception_log_with_extra_exception_type(self):
        """
        测试handle_exception_log方法在接收其他类型异常的处理
        """
        logging.info('Start test_handle_exception_log_with_extra_exception_type.')
        exception_object = KeyboardInterrupt
        handle_exception_log(exception_object)
        logging.info('Finish test_handle_exception_log_with_extra_exception_type.')


if __name__ == '__main__':
    pytest.main(['-s', os.path.abspath(__file__)])
