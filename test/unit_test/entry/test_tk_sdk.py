#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging
import os
import shutil
import unittest
import pytest
import tk.tk_sdk as sdk
from tk.utils.constants import DEFAULT_FLAGS, DEFAULT_MODES


class TestTkSdk(unittest.TestCase):
    root_path = None
    data_path = None
    output_path = None
    pretrained_model_path = None
    ckpt_path = None
    finetune_model_config_path = None
    evaluate_infer_model_config_path = None
    finetune_boot_path = None
    evaluate_boot_path = None
    infer_boot_path = None

    @classmethod
    def setUpClass(cls):
        cls.root_path = os.path.join('/', 'tmp', 'test_tk_main')
        if not os.path.exists(cls.root_path):
            os.makedirs(cls.root_path)

        cls.data_path = os.path.join(cls.root_path, 'data_path')
        if not os.path.exists(cls.data_path):
            os.makedirs(cls.data_path)

        cls.output_path = os.path.join(cls.root_path, 'output_path')
        if not os.path.exists(cls.output_path):
            os.makedirs(cls.output_path)

        cls.pretrained_model_path = os.path.join(cls.root_path, 'pretrained_model_path')
        if not os.path.exists(cls.pretrained_model_path):
            os.makedirs(cls.pretrained_model_path)

        cls.ckpt_path = os.path.join(cls.root_path, 'ckpt_path')
        if not os.path.exists(cls.ckpt_path):
            os.makedirs(cls.ckpt_path)

        src_finetune_model_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      'resource/model_config/model_config_finetune.yaml')

        src_evaluate_infer_model_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                            'resource/model_config/model_config_evaluate_infer.yaml')

        src_finetune_boot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              'resource/boot_file/boot_finetune.py')

        src_evaluate_boot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              'resource/boot_file/boot_evaluate.py')

        src_infer_boot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'resource/boot_file/boot_infer.py')

        cls.finetune_model_config_path = os.path.join(cls.root_path, 'model_config_finetune.yaml')
        cls.evaluate_infer_model_config_path = os.path.join(cls.root_path, 'model_config_evaluate_infer.yaml')
        cls.finetune_boot_path = os.path.join(cls.root_path, 'boot_finetune.py')
        cls.evaluate_boot_path = os.path.join(cls.root_path, 'boot_evaluate.py')
        cls.infer_boot_path = os.path.join(cls.root_path, 'boot_infer.py')

        shutil.copyfile(src_finetune_model_config_path, cls.finetune_model_config_path)
        shutil.copyfile(src_evaluate_infer_model_config_path, cls.evaluate_infer_model_config_path)
        shutil.copyfile(src_finetune_boot_path, cls.finetune_boot_path)
        shutil.copyfile(src_evaluate_boot_path, cls.evaluate_boot_path)
        shutil.copyfile(src_infer_boot_path, cls.infer_boot_path)

        cls._chmod(cls.data_path)
        cls._chmod(cls.output_path)
        cls._chmod(cls.pretrained_model_path)
        cls._chmod(cls.ckpt_path)
        cls._chmod(cls.finetune_model_config_path)
        cls._chmod(cls.evaluate_infer_model_config_path)
        cls._chmod(cls.finetune_boot_path)
        cls._chmod(cls.evaluate_boot_path)
        cls._chmod(cls.infer_boot_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.data_path):
            shutil.rmtree(cls.data_path)

        if os.path.exists(cls.output_path):
            shutil.rmtree(cls.output_path)

        if os.path.exists(cls.pretrained_model_path):
            shutil.rmtree(cls.pretrained_model_path)

        if os.path.exists(cls.ckpt_path):
            shutil.rmtree(cls.ckpt_path)

        if os.path.exists(cls.finetune_model_config_path):
            os.remove(cls.finetune_model_config_path)

        if os.path.exists(cls.evaluate_infer_model_config_path):
            os.remove(cls.evaluate_infer_model_config_path)

        if os.path.exists(cls.finetune_boot_path):
            os.remove(cls.finetune_boot_path)

        if os.path.exists(cls.evaluate_boot_path):
            os.remove(cls.evaluate_boot_path)

        if os.path.exists(cls.infer_boot_path):
            os.remove(cls.infer_boot_path)

        if os.path.exists(cls.root_path):
            shutil.rmtree(cls.root_path)

    @classmethod
    def _chmod(cls, path):
        """
        修正path的权限
        :param path:待修正路径
        """
        os.chmod(path, 0o750)

    def test_finetune_sdk(self):
        """
        测试SDK侧finetune接口
        """
        logging.info('Start test_finetune_sdk.')

        result = sdk.finetune(data_path=self.data_path,
                              output_path=self.output_path,
                              pretrained_model_path=self.pretrained_model_path,
                              model_config_path=self.finetune_model_config_path,
                              boot_file_path=self.finetune_boot_path)

        self.assertTrue(result)

        logging.info('Finish test_finetune_sdk.')

    def test_evaluate_sdk(self):
        """
        测试SDK侧evaluate接口
        """
        logging.info('Start test_evaluate_sdk.')

        result = sdk.evaluate(data_path=self.data_path,
                              output_path=self.output_path,
                              ckpt_path=self.ckpt_path,
                              model_config_path=self.evaluate_infer_model_config_path,
                              boot_file_path=self.evaluate_boot_path)

        self.assertEqual(result.get('status'), 0)
        self.assertEqual(result.get('task_result'), 'evaluate task success.')

        logging.info('Finish test_evaluate_sdk.')

    def test_infer_sdk(self):
        """
        测试SDK侧infer接口
        """
        logging.info('Start test_infer_sdk.')

        result = sdk.infer(data_path=self.data_path,
                           output_path=self.output_path,
                           ckpt_path=self.ckpt_path,
                           model_config_path=self.evaluate_infer_model_config_path,
                           boot_file_path=self.infer_boot_path)

        self.assertEqual(result.get('status'), 0)
        self.assertEqual(result.get('task_result'), 'infer task success.')

        logging.info('Finish test_infer_sdk.')

    def test_finetune_sdk_with_default_params(self):
        """
        测试SDK侧finetune接口, 在部分参数以非索引指定下的情况
        """
        logging.info('Start test_finetune_sdk_with_default_params.')

        result = sdk.finetune(self.data_path,
                              self.output_path,
                              pretrained_model_path=self.pretrained_model_path,
                              model_config_path=self.finetune_model_config_path,
                              boot_file_path=self.finetune_boot_path)

        self.assertTrue(result)

        logging.info('Finish test_finetune_sdk_with_default_params.')

    def test_finetune_sdk_with_param_quiet(self):
        """
        测试SDK侧finetune接口, 在传入参数quiet时是否正确触发异常
        """
        logging.info('Start test_finetune_sdk_with_param_quiet.')

        result = sdk.finetune(data_path=self.data_path,
                              output_path=self.output_path,
                              pretrained_model_path=self.pretrained_model_path,
                              model_config_path=self.finetune_model_config_path,
                              boot_file_path=self.finetune_boot_path,
                              quiet=True)

        self.assertFalse(result)

        logging.info('Finish test_finetune_sdk_with_param_quiet.')

    def test_start_by_task_type_with_invalid_task_type(self):
        """
        测试SDK侧start_by_task_type传入不合理task_type时是否及时触发异常
        """
        logging.info('Start test_start_by_task_type_with_invalid_task_type.')

        args = None
        kwargs = {'data_path': self.data_path}

        result = sdk.start_by_task_type(args, kwargs, task_type='others', ret_err_msg=False)

        self.assertFalse(result)

        logging.info('Finish test_start_by_task_type_with_invalid_task_type.')

    def test_finetune_with_none_boot_file_path(self):
        """
        测试SDK侧finetune功能不传boot_file_path情况下的处理
        """
        logging.info('Start test_finetune_with_none_boot_file_path.')

        result = sdk.finetune(data_path=self.data_path,
                              output_path=self.output_path,
                              pretrained_model_path=self.pretrained_model_path,
                              model_config_path=self.finetune_model_config_path,
                              boot_file_path=None)

        self.assertFalse(result)

        logging.info('Finish test_finetune_with_none_boot_file_path.')

    def test_finetune_with_boot_file_path_invalid_file_type(self):
        """
        测试SDK侧finetune功能传非.py结尾的boot_file_path情况
        """
        logging.info('Start test_finetune_with_boot_file_path_invalid_file_type.')

        temp_boot_file_path = os.path.join(self.root_path, 'temp_boot_file.ppp')
        with os.fdopen(os.open(temp_boot_file_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('temp boot file with invalid file type.')

        result = sdk.finetune(data_path=self.data_path,
                              output_path=self.output_path,
                              pretrained_model_path=self.pretrained_model_path,
                              model_config_path=self.finetune_model_config_path,
                              boot_file_path=temp_boot_file_path)

        self.assertFalse(result)

        if os.path.exists(temp_boot_file_path):
            os.remove(temp_boot_file_path)

        logging.info('Finish test_finetune_with_boot_file_path_invalid_file_type.')

    def test_evaluate_with_none_ckpt_path(self):
        """
        测试SDK侧evaluate功能不传ckpt_path的情况
        """
        logging.info('Start test_evaluate_with_none_ckpt_path.')

        result = sdk.evaluate(data_path=self.data_path,
                              output_path=self.output_path,
                              ckpt_path=None,
                              model_config_path=self.evaluate_infer_model_config_path,
                              boot_file_path=self.evaluate_boot_path)

        self.assertFalse(result)

        logging.info('Finish test_evaluate_with_none_ckpt_path.')

    def test_finetune_with_none_data_path(self):
        """
        测试SDK侧finetune功能不传data_path参数的情况
        """
        logging.info('Start test_finetune_with_none_data_path.')

        result = sdk.finetune(data_path=None,
                              output_path=self.output_path,
                              pretrained_model_path=self.pretrained_model_path,
                              model_config_path=self.finetune_model_config_path,
                              boot_file_path=self.finetune_boot_path)

        self.assertFalse(result)

        logging.info('Finish test_finetune_with_none_data_path.')

    def test_finetune_with_none_model_config_path(self):
        """
        测试SDK侧finetune功能不传model_config_path参数的情况
        """
        logging.info('Start test_finetune_with_none_model_config_path.')

        result = sdk.finetune(data_path=self.data_path,
                              output_path=self.output_path,
                              pretrained_model_path=self.pretrained_model_path,
                              model_config_path=None,
                              boot_file_path=self.finetune_boot_path)

        self.assertTrue(result)

        logging.info('Finish test_finetune_with_none_model_config_path.')

    def test_finetune_with_model_config_path_invalid_file_type(self):
        """
        测试SDK侧finetune功能传非.yaml/.yml结尾的model_config_path的情况
        """
        logging.info('Start test_finetune_with_model_config_path_invalid_file_type.')

        temp_model_config_path = os.path.join(self.root_path, 'temp_model_config.yyy')
        with os.fdopen(os.open(temp_model_config_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('temp model config file with invalid file type.')

        result = sdk.finetune(data_path=self.data_path,
                              output_path=self.output_path,
                              pretrained_model_path=self.pretrained_model_path,
                              model_config_path=temp_model_config_path,
                              boot_file_path=self.finetune_boot_path)

        self.assertFalse(result)

        logging.info('Finish test_finetune_with_model_config_path_invalid_file_type.')

    def test_finetune_with_none_output_path(self):
        """
        测试SDK侧finetune功能不传output参数的情况
        """
        logging.info('Start test_finetune_with_none_output_path.')

        result = sdk.finetune(data_path=self.data_path,
                              output_path=None,
                              pretrained_model_path=self.pretrained_model_path,
                              model_config_path=self.finetune_model_config_path,
                              boot_file_path=self.finetune_boot_path)

        self.assertFalse(result)

        logging.info('Finish test_finetune_with_none_output_path.')

    def test_finetune_with_none_pretrained_model_path(self):
        """
        测试SDK侧finetune功能不传pretrained_model_path参数的情况
        """
        logging.info('Start test_finetune_with_none_pretrained_model_path.')

        result = sdk.finetune(data_path=self.data_path,
                              output_path=self.output_path,
                              pretrained_model_path=None,
                              model_config_path=self.finetune_model_config_path,
                              boot_file_path=self.finetune_boot_path)

        self.assertTrue(result)

        logging.info('Finish test_finetune_with_none_pretrained_model_path.')


if __name__ == '__main__':
    pytest.main(['-s', os.path.abspath(__file__)])
