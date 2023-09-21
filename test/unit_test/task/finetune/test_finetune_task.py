#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging
import os
import shutil
import unittest
from unittest import mock
import pytest
import yaml
from mindpet.task.finetune.finetune_task import FinetuneTask
from mindpet.utils.constants import DEFAULT_FLAGS, DEFAULT_MODES
from mindpet.utils.exceptions import PathLoopError, ReadYamlFileError, ModelConfigKeysInfoError, \
    ModelConfigParamsInfoError, TaskError, CreateProcessError, MonitorProcessRspError, UnexpectedError

logging.getLogger().setLevel(logging.INFO)


class TestFinetuneTask(unittest.TestCase):
    root_path = None
    data_path = None
    output_path = None
    pretrained_model_path = None
    model_config_path = None
    boot_file_path = None

    @classmethod
    def setUpClass(cls):
        cls.root_path = os.path.join('/', 'tmp', 'finetune_task')
        if not os.path.exists(cls.root_path):
            os.makedirs(cls.root_path, exist_ok=True)

        cls.data_path = os.path.join(cls.root_path, 'data_path')
        if not os.path.exists(cls.data_path):
            os.makedirs(cls.data_path, exist_ok=True)

        cls.output_path = os.path.join(cls.root_path, 'output_path')
        if not os.path.exists(cls.output_path):
            os.makedirs(cls.output_path, exist_ok=True)

        cls.pretrained_model_path = os.path.join(cls.root_path, 'pretrained_model_files')
        if not os.path.exists(cls.pretrained_model_path):
            os.makedirs(cls.pretrained_model_path, exist_ok=True)

        cls.model_config_path = os.path.join(cls.root_path, 'model_config.yaml')
        model_config_content = {'params': {'lr': '1e-4'}, 'freeze': {'block1': 'layer1'}}
        with os.fdopen(os.open(cls.model_config_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            yaml.dump(model_config_content, file)

        cls.boot_file_path = os.path.join(cls.root_path, 'boot.py')
        with os.fdopen(os.open(cls.boot_file_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('print("enter into boot.py process.")')

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.model_config_path):
            os.remove(cls.model_config_path)

        if os.path.exists(cls.boot_file_path):
            os.remove(cls.boot_file_path)

        if os.path.exists(cls.data_path):
            shutil.rmtree(cls.data_path)

        if os.path.exists(cls.output_path):
            shutil.rmtree(cls.output_path)

        if os.path.exists(cls.pretrained_model_path):
            os.rmdir(cls.pretrained_model_path)

        if os.path.exists(cls.root_path):
            shutil.rmtree(cls.root_path)

    def test_finetune_task_init_with_in_out_loop_path(self):
        """
        测试微调任务构造方法中, 传入了循环嵌套路径的情况
        """
        logging.info('Start test_finetune_task_init_with_in_out_loop_path.')

        output_path = os.path.join(self.data_path, 'inner_outputs')

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        with self.assertRaises(PathLoopError):
            FinetuneTask(data_path=self.data_path,
                         output_path=output_path,
                         pretrainde_model_path=self.pretrained_model_path,
                         model_config_path=self.model_config_path,
                         boot_file_path=self.boot_file_path)

        if os.path.exists(output_path):
            os.rmdir(output_path)

        logging.info('Finish test_finetune_task_init_with_in_out_loop_path.')

    def test_finetune_task_init_with_invalid_model_config_path(self):
        """
        测试微调任务构造方法中, 参数model_config_path传入不合理类型参数的情况
        """
        logging.info('Start test_finetune_task_init_with_invalid_model_config_path.')

        model_config_path = os.path.join(self.root_path, 'temp_model_config.yaml')

        with self.assertRaises(ReadYamlFileError):
            FinetuneTask(data_path=self.data_path,
                         output_path=self.output_path,
                         pretrainde_model_path=self.pretrained_model_path,
                         model_config_path=model_config_path,
                         boot_file_path=self.boot_file_path)

        if os.path.exists(model_config_path):
            os.remove(model_config_path)

        logging.info('Finish test_finetune_task_init_with_invalid_model_config_path.')

    def test_finetune_task_init_with_invalid_key_in_model_config_path(self):
        """
        测试微调任务构造方法中, 参数model_config_path传入不合理类型参数的情况
        """
        logging.info('Start test_finetune_task_init_with_invalid_key_in_model_config_path.')

        yaml_content = {'invalid_key': 'value'}

        model_config_path = os.path.join(self.root_path, 'temp_model_config.yaml')

        with os.fdopen(os.open(model_config_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            yaml.dump(yaml_content, file)

        with self.assertRaises(ModelConfigKeysInfoError):
            FinetuneTask(data_path=self.data_path,
                         output_path=self.output_path,
                         pretrained_model_path=self.pretrained_model_path,
                         model_config_path=model_config_path,
                         boot_file_path=self.boot_file_path)

        if os.path.exists(model_config_path):
            os.remove(model_config_path)

        logging.info('Finish test_finetune_task_init_with_invalid_key_in_model_config_path.')

    def test_finetune_task_init_with_none_params_in_model_config_path(self):
        """
        测试微调任务构造方法中, 参数model_config_path文件中, params部分信息为None的情况
        """
        logging.info('Start test_finetune_task_init_with_none_params_in_model_config_path.')

        yaml_content = {'params': None}

        model_config_path = os.path.join(self.root_path, 'temp_model_config.yaml')

        with os.fdopen(os.open(model_config_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            yaml.dump(yaml_content, file)

        with self.assertRaises(ModelConfigParamsInfoError):
            FinetuneTask(data_path=self.data_path,
                         output_path=self.output_path,
                         pretrained_model_path=self.pretrained_model_path,
                         model_config_path=model_config_path,
                         boot_file_path=self.boot_file_path)

        if os.path.exists(model_config_path):
            os.remove(model_config_path)

        logging.info('Finish test_finetune_task_init_with_none_params_in_model_config_path.')

    def test_finetune_task_init_with_invalid_params_config_in_model_config_path(self):
        """
        测试微调任务构造方法中, 参数model_config_path文件中, params部分包含不合理信息的情况
        """
        logging.info('Start test_finetune_task_init_with_invalid_params_config_in_model_config_path.')

        yaml_content = {'params': {'--key': 'value'}}

        model_config_path = os.path.join(self.root_path, 'temp_model_config.yaml')

        with os.fdopen(os.open(model_config_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            yaml.dump(yaml_content, file)

        with self.assertRaises(ModelConfigParamsInfoError):
            FinetuneTask(data_path=self.data_path,
                         output_path=self.output_path,
                         pretrained_model_path=self.pretrained_model_path,
                         model_config_path=model_config_path,
                         boot_file_path=self.boot_file_path)

        if os.path.exists(model_config_path):
            os.remove(model_config_path)

        logging.info('Finish test_finetune_task_init_with_invalid_params_config_in_model_config_path.')

    @mock.patch('tk.task.finetune.finetune_task.read_yaml_file')
    def test_finetune_task_init_with_read_model_config_yaml_throw_exception(self, mock_func):
        """
        测试FinetuneTask在解析model_config.yaml文件params时触发Exception类型异常的情况
        :param mock_func: mock方法
        """
        logging.info('Start test_finetune_task_init_with_read_model_config_yaml_throw_exception.')

        mock_func.side_effect = RuntimeError
        with self.assertRaises(UnexpectedError):
            self._get_default_finetune_task()

        logging.info('Finish test_finetune_task_init_with_read_model_config_yaml_throw_exception.')

    def test_finetune_task_init_without_model_config_path(self):
        """
        测试微调任务构造方法中, 不传model_config_path的情况
        """
        logging.info('Start test_finetune_task_init_without_model_config_path.')

        finetune_task = FinetuneTask(data_path=self.data_path,
                                     output_path=self.output_path,
                                     pretrained_model_path=self.pretrained_model_path,
                                     model_config_path=None,
                                     boot_file_path=self.boot_file_path)

        param_dict = dict()
        command_params = finetune_task.command_params

        for i in range(0, len(command_params), 2):
            param_dict[command_params[i]] = command_params[i + 1]

        self.assertNotIn('--lr', param_dict.keys())

        logging.info('Finish test_finetune_task_init_without_model_config_path.')

    @mock.patch('tk.task.finetune.finetune_task.FinetuneTask._process_param_and_command')
    def test_finetune_task_init_with_process_param_and_command_throw_keyboard_interrupt(self, mock_func):
        """
        测试微调任务构造方法中, 组装子进程命令时抛KeyboardInterrupt异常的情况
        :param mock_func: mock方法
        """
        logging.info('Start test_finetune_task_init_with_process_param_and_command_throw_keyboard_interrupt.')

        mock_func.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self._get_default_finetune_task()

        logging.info('Finish test_finetune_task_init_with_process_param_and_command_throw_keyboard_interrupt.')

    def test_finetune_task_init_with_normal_params_and_freeze_configs_in_model_config_path(self):
        """
        测试微调任务构造方法中, model_config_path文件中params和freeze部分内容均正常的情况
        """
        logging.info('Start test_finetune_task_init_with_normal_params_and_freeze_configs_in_model_config_path.')

        finetune_task = self._get_default_finetune_task()

        param_dict = dict()
        command_params = finetune_task.command_params

        for i in range(0, len(command_params), 2):
            param_dict[command_params[i]] = command_params[i + 1]

        self.assertIn(self.output_path, param_dict.get('--output_path'))
        self.assertEqual(param_dict.get('--data_path'), self.data_path)
        self.assertEqual(param_dict.get('--pretrained_model_path'), self.pretrained_model_path)
        self.assertEqual(param_dict.get('--lr'), '1e-4')
        self.assertEqual(param_dict.get('--advanced_config'), self.model_config_path)

        logging.info('Finish test_finetune_task_init_with_normal_params_and_freeze_configs_in_model_config_path.')

    def test_finetune_task_with_wrong_subtask_exit_code(self):
        """
        测试微调任务执行过程中, 子进程任务返回错误退出码的情况
        """
        logging.info('Start test_finetune_task_with_wrong_subtask_exit_code.')

        if os.path.exists(self.boot_file_path):
            os.remove(self.boot_file_path)

        with os.fdopen(os.open(self.boot_file_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('raise RuntimeError')

        finetune_task = self._get_default_finetune_task()

        with self.assertRaises(TaskError):
            finetune_task.start()

        logging.info('Finish test_finetune_task_with_wrong_subtask_exit_code.')

    def test_finetune_task_with_valid_subtask_exit_code(self):
        """
        测试微调任务执行过程中, 子进程任务返回正常退出码的情况
        """
        logging.info('Start test_finetune_task_with_valid_subtask_exit_code.')

        finetune_task = self._get_default_finetune_task()

        self.assertTrue(finetune_task.start())

        logging.info('Finish test_finetune_task_with_valid_subtask_exit_code.')

    @mock.patch('subprocess.Popen')
    def test_finetune_task_with_create_subprocess_command_throw_keyboard_interrupt(self, mock_func):
        """
        测试微调任务执行过程中, 创建子进程对象时抛出KeyboardInterrupt异常的情况
        :param mock_func: mock方法
        """
        logging.info('Start test_finetune_task_with_create_subprocess_command_throw_keyboard_interrupt.')

        finetune_task = self._get_default_finetune_task()

        mock_func.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            finetune_task.start()

        logging.info('Finish test_finetune_task_with_create_subprocess_command_throw_keyboard_interrupt.')

    @mock.patch('subprocess.Popen')
    def test_finetune_task_with_create_subprocess_command_throw_exception(self, mock_func):
        """
        测试微调任务执行过程中, 创建子进程对象时抛出Exception类型异常的情况
        :param mock_func: mock方法
        """
        logging.info('Start test_finetune_task_with_create_subprocess_command_throw_exception.')

        finetune_task = self._get_default_finetune_task()

        mock_func.side_effect = RuntimeError
        with self.assertRaises(CreateProcessError):
            finetune_task.start()

        logging.info('Finish test_finetune_task_with_create_subprocess_command_throw_exception.')

    @mock.patch('subprocess.Popen')
    def test_finetune_task_with_create_subprocess_command_throw_exception_containing_err_msg(self, mock_func):
        """
        测试微调任务执行过程中, 创建子进程对象时抛出Exception类型异常(包含错误信息)的情况
        :param mock_func: mock方法
        """
        logging.info('Start test_finetune_task_with_create_subprocess_command_throw_exception_containing_err_msg.')

        finetune_task = self._get_default_finetune_task()

        mock_func.side_effect = RuntimeError('runtime error occurred!!!')
        with self.assertRaises(CreateProcessError):
            finetune_task.start()

        logging.info('Finish test_finetune_task_with_create_subprocess_command_throw_exception_containing_err_msg.')

    @mock.patch('tk.task.finetune.finetune_task.monitor_process_rsp_code')
    def test_finetune_task_with_monitor_process_rsp_code_throw_keyboard_interrupt(self, mock_func):
        """
        测试FinetuneTask在监测子进程执行过程中出现KeyboardInterrupt异常的情况
        :param mock_func: mock方法
        """
        logging.info('Start test_finetune_task_with_monitor_process_rsp_code_throw_keyboard_interrupt.')

        mock_func.side_effect = KeyboardInterrupt
        task = self._get_default_finetune_task()
        with self.assertRaises(KeyboardInterrupt):
            task.start()

        logging.info('Finish test_finetune_task_with_monitor_process_rsp_code_throw_keyboard_interrupt.')

    @mock.patch('tk.task.finetune.finetune_task.monitor_process_rsp_code')
    def test_finetune_task_with_monitor_process_rsp_code_throw_exception_containing_err_msg(self, mock_func):
        """
        测试FinetuneTask在监测子进程执行过程中出现Exception类型异常(含错误信息)的情况
        :param mock_func: mock方法
        """
        logging.info('Start test_finetune_task_with_monitor_process_rsp_code_throw_exception_containing_err_msg.')

        mock_func.side_effect = RuntimeError('finetune task raise runtime error')
        task = self._get_default_finetune_task()
        with self.assertRaises(MonitorProcessRspError):
            task.start()

        logging.info('Finish test_finetune_task_with_monitor_process_rsp_code_throw_exception_containing_err_msg.')

    @mock.patch('tk.task.finetune.finetune_task.monitor_process_rsp_code')
    def test_finetune_task_with_monitor_process_rsp_code_throw_exception(self, mock_func):
        """
        测试FinetuneTask在监测子进程执行过程中出现Exception类型异常(不包含错误信息)的情况
        :param mock_func: mock方法
        """
        logging.info('Start test_finetune_task_with_monitor_process_rsp_code_throw_exception.')

        mock_func.side_effect = RuntimeError
        task = self._get_default_finetune_task()
        with self.assertRaises(MonitorProcessRspError):
            task.start()

        logging.info('Finish test_finetune_task_with_monitor_process_rsp_code_throw_exception.')

    def _get_default_finetune_task(self):
        """
        构建默认的finetune任务
        :return: finetune任务对象
        """
        return FinetuneTask(data_path=self.data_path,
                            output_path=self.output_path,
                            pretrained_model_path=self.pretrained_model_path,
                            model_config_path=self.model_config_path,
                            boot_file_path=self.boot_file_path)


if __name__ == '__main__':
    pytest.main(['-s', os.path.abspath(__file__)])
