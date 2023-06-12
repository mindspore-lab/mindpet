#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os
import json
import logging
import shutil
import unittest
import unittest.mock as mock
import pytest
import yaml
from tk.utils.constants import DEFAULT_FLAGS, DEFAULT_MODES
from tk.task.evaluate_infer.evaluate_infer_task import EvaluateInferTask
from tk.utils.exceptions import ReadYamlFileError, CreateProcessError, TaskError, MonitorProcessRspError, \
    PathRightEscalationError, UnexpectedError

logging.getLogger().setLevel(logging.INFO)


class TestEvaluateInferTask(unittest.TestCase):
    root_path = None
    data_path = None
    output_path = None
    ckpt_path = None
    model_config_path = None
    evaluate_boot_file_path = None
    infer_boot_file_path = None

    @classmethod
    def setUpClass(cls):
        cls.root_path = os.path.join('/', 'tmp', 'evaluate_infer')
        if not os.path.exists(cls.root_path):
            os.makedirs(cls.root_path, exist_ok=True)

        cls.data_path = os.path.join(cls.root_path, 'data_path')
        if not os.path.exists(cls.data_path):
            os.makedirs(cls.data_path)

        cls.output_path = os.path.join(cls.root_path, 'output_path')
        if not os.path.exists(cls.output_path):
            os.makedirs(cls.output_path)

        cls.ckpt_path = os.path.join(cls.root_path, 'ckpt_path')
        if not os.path.exists(cls.ckpt_path):
            os.makedirs(cls.ckpt_path)

        cls.model_config_path = os.path.join(cls.root_path, 'model_config.yaml')
        model_config_content = {'params': {'lr': '1e-4'}}
        with os.fdopen(os.open(cls.model_config_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            yaml.dump(model_config_content, file)

        src_evaluate_boot_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   'resource/normal_evaluate_boot_file.py')
        cls.evaluate_boot_file_path = os.path.join(cls.root_path, 'normal_evaluate_boot_file.py')

        src_infer_boot_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                'resource/normal_infer_boot_file.py')
        cls.infer_boot_file_path = os.path.join(cls.root_path, 'normal_infer_boot_file.py')

        shutil.copyfile(src_evaluate_boot_file_path, cls.evaluate_boot_file_path)
        shutil.copyfile(src_infer_boot_file_path, cls.infer_boot_file_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.model_config_path):
            os.remove(cls.model_config_path)

        if os.path.exists(cls.evaluate_boot_file_path):
            os.remove(cls.evaluate_boot_file_path)

        if os.path.exists(cls.infer_boot_file_path):
            os.remove(cls.infer_boot_file_path)

        if os.path.exists(cls.data_path):
            shutil.rmtree(cls.data_path)

        if os.path.exists(cls.output_path):
            shutil.rmtree(cls.output_path)

        if os.path.exists(cls.ckpt_path):
            shutil.rmtree(cls.ckpt_path)

        if os.path.exists(cls.root_path):
            shutil.rmtree(cls.root_path)

    def test_evaluate_infer_task_init_with_invalid_task_type(self):
        """
        测试EvaluateInferTask输入不存在task_type的情况
        """
        logging.info('Start test_evaluate_infer_task_init_with_invalid_task_type.')

        with self.assertRaises(ValueError):
            EvaluateInferTask(task_type='others',
                              data_path=self.data_path,
                              output_path=self.output_path,
                              ckpt_path=self.ckpt_path,
                              model_config_path=self.model_config_path,
                              boot_file_path=self.evaluate_boot_file_path)

        logging.info('Finish test_evaluate_infer_task_init_with_invalid_task_type.')

    @mock.patch('tk.task.evaluate_infer.evaluate_infer_task.EvaluateInferTask._process_param_and_command')
    def test_evaluate_infer_task_init_with_process_param_and_command_throw_keyboard_interrupt(self, mock_func):
        """
        测试EvaluateInferTask在组装参数时触发KeyboardInterrupt异常的情况
        :param mock_func: mock方法
        """
        logging.info('Start test_evaluate_infer_task_init_with_process_param_and_command_throw_keyboard_interrupt.')

        mock_func.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self._get_default_evaluate_task()

        logging.info('Finish test_evaluate_infer_task_init_with_process_param_and_command_throw_keyboard_interrupt.')

    def test_evaluate_task_init_with_invalid_model_config_path(self):
        """
        测试EvaluateInferTask在接收不存在model_config_path时的情况
        """
        logging.info('Start test_evaluate_task_init_without_model_config_path.')

        model_config_path = os.path.join(self.root_path, 'temp_model_config.yaml')

        with self.assertRaises(ReadYamlFileError):
            EvaluateInferTask(task_type='evaluate',
                              data_path=self.data_path,
                              output_path=self.output_path,
                              ckpt_path=self.ckpt_path,
                              model_config_path=model_config_path,
                              boot_file_path=self.evaluate_boot_file_path)

        if os.path.exists(model_config_path):
            os.remove(model_config_path)

        logging.info('Finish test_evaluate_task_init_without_model_config_path.')

    @mock.patch('subprocess.Popen')
    def test_evaluate_infer_task_with_create_process_throw_keyboard_interrupt(self, mock_func):
        """
        测试EvaluateInferTask在创建子进程时抛出KeyboardInterrupt异常时的情况
        :param mock_func: mock方法
        """
        logging.info('Start test_evaluate_infer_task_with_create_process_throw_keyboard_interrupt.')

        mock_func.side_effect = KeyboardInterrupt
        task = self._get_default_evaluate_task()
        with self.assertRaises(KeyboardInterrupt):
            task.start()

        logging.info('Finish test_evaluate_infer_task_with_create_process_throw_keyboard_interrupt.')

    @mock.patch('subprocess.Popen')
    def test_evaluate_infer_task_with_create_process_throw_exception(self, mock_func):
        """
        测试EvaluateInferTask在创建子进程时抛出Exception类型异常时的情况
        :param mock_func: mock方法
        """
        logging.info('Start test_evaluate_infer_task_with_create_process_throw_keyboard_interrupt.')

        mock_func.side_effect = RuntimeError
        task = self._get_default_evaluate_task()
        with self.assertRaises(CreateProcessError):
            task.start()

        logging.info('Finish test_evaluate_infer_task_with_create_process_throw_keyboard_interrupt.')

    @mock.patch('subprocess.Popen')
    def test_evaluate_infer_task_with_create_process_throw_exception_containing_err_msg(self, mock_func):
        """
        测试EvaluateInferTask在创建子进程时抛出Exception类型异常时的情况, 其中异常包含信息
        :param mock_func: mock方法
        """
        logging.info('Start test_evaluate_infer_task_with_create_process_throw_exception_containing_err_msg.')

        mock_func.side_effect = RuntimeError('runtime error occurred!!!')
        task = self._get_default_evaluate_task()
        with self.assertRaises(CreateProcessError):
            task.start()

        logging.info('Finish test_evaluate_infer_task_with_create_process_throw_exception_containing_err_msg.')

    @mock.patch('tk.task.evaluate_infer.evaluate_infer_task.monitor_process_rsp_code')
    def test_evaluate_infer_task_with_monitor_process_rsp_code_throw_keyboard_interrupt(self, mock_func):
        """
        测试EvaluateInferTask在监测子任务状态时出现KeyboardInterrupt异常时的处理
        :param mock_func: mock方法
        """
        logging.info('Start test_evaluate_infer_task_with_monitor_process_rsp_code_throw_keyboard_interrupt.')

        mock_func.side_effect = KeyboardInterrupt
        task = self._get_default_evaluate_task()
        with self.assertRaises(KeyboardInterrupt):
            task.start()

        logging.info('Finish test_evaluate_infer_task_with_monitor_process_rsp_code_throw_keyboard_interrupt.')

    @mock.patch('tk.task.evaluate_infer.evaluate_infer_task.monitor_process_rsp_code')
    def test_evaluate_infer_task_with_monitor_process_rsp_code_throw_exception_containing_msg(self, mock_func):
        """
        测试EvaluateInferTask在监测子任务状态时出现Exception(包含异常信息)类型异常时的处理
        :param mock_func: mock方法
        """
        logging.info('Start test_evaluate_infer_task_with_monitor_process_rsp_code_throw_exception_containing_msg.')

        mock_func.side_effect = RuntimeError('monitor process rsp code raises runtime error')
        task = self._get_default_evaluate_task()
        with self.assertRaises(MonitorProcessRspError):
            task.start()

        logging.info('Finish test_evaluate_infer_task_with_monitor_process_rsp_code_throw_exception_containing_msg.')

    @mock.patch('tk.task.evaluate_infer.evaluate_infer_task.monitor_process_rsp_code')
    def test_evaluate_infer_task_with_monitor_process_rsp_code_throw_exception(self, mock_func):
        """
        测试EvaluateInferTask在监测子任务状态时出现Exception(不包含异常信息)类型异常时的处理
        :param mock_func: mock方法
        """
        logging.info('Start test_evaluate_infer_task_with_monitor_process_rsp_code_throw_exception.')

        mock_func.side_effect = RuntimeError
        task = self._get_default_evaluate_task()
        with self.assertRaises(MonitorProcessRspError):
            task.start()

        logging.info('Finish test_evaluate_infer_task_with_monitor_process_rsp_code_throw_exception.')

    def test_evaluate_infer_task_with_abnormal_exit_code(self):
        """
        测试EvaluateInferTask在模型代码意外退出时的状态
        """
        logging.info('Start test_evaluate_infer_task_with_abnormal_exit_code.')

        boot_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'resource/boot_file_with_runtime_error.py')

        task = EvaluateInferTask(task_type='evaluate',
                                 data_path=self.data_path,
                                 output_path=self.output_path,
                                 ckpt_path=self.ckpt_path,
                                 model_config_path=self.model_config_path,
                                 boot_file_path=boot_file_path)

        with self.assertRaises(TaskError):
            task.start()

        logging.info('Finish test_evaluate_infer_task_with_abnormal_exit_code.')

    @mock.patch('tk.task.evaluate_infer.evaluate_infer_task.read_json_file')
    def test_evaluate_task_with_read_result_json_throw_json_decode_error(self, mock_func):
        """
        测试EvaluateInferTask在校验eval_result.json时读取json文件抛出JsonDecodeError异常的情况
        :param mock_func: mock方法
        """
        logging.info('Start test_evaluate_task_with_read_result_json_throw_json_decode_error.')

        mock_func.side_effect = json.JSONDecodeError
        task = self._get_default_evaluate_task()
        result = task.start()

        self.assertEqual(result.get('status'), -1)
        self.assertEqual(result.get('error_message'), 'File eval_result.json should follow JSON format.')

        logging.info('Finish test_evaluate_task_with_read_result_json_throw_json_decode_error.')

    @mock.patch('tk.task.evaluate_infer.evaluate_infer_task.read_json_file')
    def test_evaluate_task_with_read_result_json_throw_exception(self, mock_func):
        """
        测试EvaluateInferTask在校验eval_result.json时读取json文件抛出Exception类型异常的情况
        :param mock_func: mock方法
        """
        logging.info('Start test_evaluate_task_with_read_result_json_throw_exception.')

        mock_func.side_effect = RuntimeError('read json file raise runtime error')
        task = self._get_default_evaluate_task()
        result = task.start()

        self.assertEqual(result.get('status'), -1)
        self.assertEqual(result.get('error_message'),
                         'An error occurred during reading eval_result.json: read json file raise runtime error')

        logging.info('Finish test_evaluate_task_with_read_result_json_throw_exception.')

    def test_evaluate_task_json_check_without_accessing_json_file(self):
        """
        测试EvaluateInferTask在模型代码不生成result_json的情况
        """
        logging.info('Start test_evaluate_task_json_check_without_accessing_json_file.')

        boot_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'resource/boot_file_without_generating_result_json.py')

        task = EvaluateInferTask(task_type='evaluate',
                                 data_path=self.data_path,
                                 output_path=self.output_path,
                                 ckpt_path=self.ckpt_path,
                                 model_config_path=self.model_config_path,
                                 boot_file_path=boot_file_path)

        result = task.start()

        self.assertEqual(result.get('status'), -1)

        logging.info('Finish test_evaluate_task_json_check_without_accessing_json_file.')

    def test_evaluate_infer_task_json_check_with_empty_json_file(self):
        """
        测试EvaluateInferTask在模型生成result_json内容为空的情况
        """
        logging.info('Start test_evaluate_infer_task_json_check_with_empty_json_file.')

        boot_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'resource/boot_file_with_empty_json.py')

        task = EvaluateInferTask(task_type='evaluate',
                                 data_path=self.data_path,
                                 output_path=self.output_path,
                                 ckpt_path=self.ckpt_path,
                                 model_config_path=self.model_config_path,
                                 boot_file_path=boot_file_path)

        result = task.start()

        self.assertEqual(result.get('status'), -1)
        self.assertEqual(result.get('error_message'), 'File eval_result.json is empty.')
        self.assertEqual(result.get('task_result'), '')

        logging.info('Finish test_evaluate_infer_task_json_check_with_empty_json_file.')

    @mock.patch('tk.task.evaluate_infer.evaluate_infer_task.EvaluateInferTask._check_task_result')
    def test_evaluate_infer_task_with_result_json_containing_right_escalation_risk(self, mock_func):
        """
        测试EvaluateInferTask在校验eval_result.json时发现存在提权风险时的情况
        :param mock_func: mock方法
        """
        logging.info('Start test_evaluate_infer_task_with_result_json_containing_right_escalation_risk.')

        mock_func.side_effect = PathRightEscalationError
        task = self._get_default_evaluate_task()
        result = task.start()

        self.assertEqual(result.get('status'), -1)
        self.assertEqual(result.get('error_message'), 'Permission denied when reading eval_result.json.')

        logging.info('Finish test_evaluate_infer_task_with_result_json_containing_right_escalation_risk.')

    def test_evaluate_infer_task_with_oversize_result_json(self):
        """
        测试EvaluateInferTask在模型生成过大result_json文件的情况
        """
        logging.info('Start test_evaluate_infer_task_with_oversize_result_json.')

        boot_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'resource/boot_file_with_oversize_result_json.py')

        task = EvaluateInferTask(task_type='evaluate',
                                 data_path=self.data_path,
                                 output_path=self.output_path,
                                 ckpt_path=self.ckpt_path,
                                 model_config_path=self.model_config_path,
                                 boot_file_path=boot_file_path)

        result = task.start()

        self.assertEqual(result.get('status'), -1)
        self.assertEqual(result.get('error_message'), 'File eval_result.json is too large.')

        logging.info('Finish test_evaluate_infer_task_with_oversize_result_json.')

    def test_evaluate_infer_task_with_link_result_json(self):
        """
        测试EvaluateInferTask在模型生成result_json路径为软链接时的情况
        """
        logging.info('Start test_evaluate_infer_task_with_link_result_json.')

        boot_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'resource/boot_file_with_link_result_json.py')

        task = EvaluateInferTask(task_type='evaluate',
                                 data_path=self.data_path,
                                 output_path=self.output_path,
                                 ckpt_path=self.ckpt_path,
                                 model_config_path=self.model_config_path,
                                 boot_file_path=boot_file_path)

        result = task.start()

        self.assertEqual(result.get('status'), -1)
        self.assertEqual(result.get('error_message'), 'Detect link path, reject reading file: eval_result.json.')

        logging.info('Finish test_evaluate_infer_task_with_link_result_json.')

    @mock.patch('tk.task.evaluate_infer.evaluate_infer_task.EvaluateInferTask._check_task_result')
    def test_evaluate_infer_task_with_value_error_result_json(self, mock_func):
        """
        测试EvaluateInferTask在模型生成result_json时触发ValueError时的情况
        :param mock_func: mock方法
        """
        logging.info('Start test_evaluate_infer_task_with_value_error_result_json.')

        mock_func.side_effect = ValueError

        task = self._get_default_evaluate_task()
        result = task.start()

        self.assertEqual(result.get('status'), -1)
        self.assertEqual(result.get('error_message'), 'Invalid file: eval_result.json.')

        logging.info('Finish test_evaluate_infer_task_with_value_error_result_json.')

    @mock.patch('tk.task.evaluate_infer.evaluate_infer_task.EvaluateInferTask._check_task_result')
    def test_evaluate_infer_task_with_exception_result_json(self, mock_func):
        """
        测试EvaluateInferTask在模型生成result_json时触发Exception时的情况
        :param mock_func: mock方法
        """
        logging.info('Start test_evaluate_infer_task_with_exception_result_json.')

        mock_func.side_effect = RuntimeError('result.json runtime error!!!')

        task = self._get_default_evaluate_task()
        result = task.start()

        self.assertEqual(result.get('status'), -1)
        self.assertEqual(result.get('error_message'),
                         'An error occurred during reading eval_result.json: result.json runtime error!!!')

        logging.info('Finish test_evaluate_infer_task_with_exception_result_json.')

    def test_evaluate_infer_task_with_none_model_config_path(self):
        """
        测试EvaluateInferTask在不配置model_config_path的情况
        """
        logging.info('Start test_evaluate_infer_task_with_none_model_config_path.')

        task = EvaluateInferTask(task_type='evaluate',
                                 data_path=self.data_path,
                                 output_path=self.output_path,
                                 ckpt_path=self.ckpt_path,
                                 model_config_path=None,
                                 boot_file_path=self.evaluate_boot_file_path)

        result = task.start()

        self.assertEqual(result.get('status'), 0)

        logging.info('Finish test_evaluate_infer_task_with_none_model_config_path.')

    @mock.patch('tk.task.evaluate_infer.evaluate_infer_task.read_yaml_file')
    def test_evaluate_infer_task_with_read_model_config_yaml_throw_exception(self, mock_func):
        """
        测试EvaluateInferTask在从model_config.yaml中解析params参数时发生Exception类型异常的情况
        :param mock_func: mock方法
        """
        logging.info('Start test_evaluate_infer_task_with_read_model_config_yaml_throw_exception.')

        mock_func.side_effect = RuntimeError
        with self.assertRaises(UnexpectedError):
            self._get_default_evaluate_task()

        logging.info('Finish test_evaluate_infer_task_with_read_model_config_yaml_throw_exception.')

    def test_evaluate_task(self):
        """
        正常执行evaluate任务
        """
        logging.info('Start test_evaluate_task.')

        task = self._get_default_evaluate_task()
        result = task.start()

        self.assertEqual(result.get('status'), 0)
        self.assertEqual(result.get('task_result'), 'evaluate task success')

        logging.info('Finish test_evaluate_task.')

    def test_infer_task(self):
        """
        正常执行infer任务
        """
        logging.info('Start test_infer_task.')

        task = self._get_default_infer_task()
        result = task.start()

        self.assertEqual(result.get('status'), 0)
        self.assertEqual(result.get('task_result'), 'infer task success')

        logging.info('Finish test_infer_task.')

    def _get_default_evaluate_task(self):
        """
        获取默认的evaluate任务对象
        :return: 任务对象
        """
        return EvaluateInferTask(task_type='evaluate',
                                 data_path=self.data_path,
                                 output_path=self.output_path,
                                 ckpt_path=self.ckpt_path,
                                 model_config_path=self.model_config_path,
                                 boot_file_path=self.evaluate_boot_file_path)

    def _get_default_infer_task(self):
        """
        获取默认的infer任务对象
        :return: 任务对象
        """
        return EvaluateInferTask(task_type='infer',
                                 data_path=self.data_path,
                                 output_path=self.output_path,
                                 ckpt_path=self.ckpt_path,
                                 model_config_path=self.model_config_path,
                                 boot_file_path=self.infer_boot_file_path)


if __name__ == '__main__':
    pytest.main(['-s', os.path.abspath(__file__)])
