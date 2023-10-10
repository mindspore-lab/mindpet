#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
import sys
sys.path.append('.')
import logging
import shutil
import os.path
import unittest
import pytest
from mindpet.task.finetune.finetune_task import FinetuneTask
from mindpet.security.param_check.model_config_params_check_util import ModelConfigParamsChecker
from mindpet.utils.exceptions import ModelConfigParamsInfoError
from mindpet.utils.constants import MODEL_CONFIG_LEN_LIMIT, MODEL_CONFIG_PARAMS_CHARACTER_BLACKLIST

logging.getLogger().setLevel(logging.INFO)


class TestModelConfigParamsChecker(unittest.TestCase):
    data_path = None
    output_path = None

    @classmethod
    def setUpClass(cls):
        root_path = os.path.join('/', 'tmp', 'task')
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        cls.data_path = os.path.join(root_path, 'data_path')
        if not os.path.exists(cls.data_path):
            os.makedirs(cls.data_path)

        cls.output_path = os.path.join(root_path, 'output_path')
        if not os.path.exists(cls.output_path):
            os.makedirs(cls.output_path)

        task_params = {'data_path': cls.data_path, 'output_path': cls.output_path}

        cls.task = FinetuneTask(**task_params)

        cls.default_params = {
            'learning_rate': 1e-4,
            'batch_size': 32,
            'model_name': 'resnet-50'
        }

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.data_path):
            shutil.rmtree(cls.data_path)

        if os.path.exists(cls.output_path):
            shutil.rmtree(cls.output_path)

    def test_model_config_check_init_with_none_task_object(self):
        """
        测试ModelConfigCheck类构造方法在task_object对象传None时的处理
        """
        logging.info('Start test_model_config_check_init_with_none_task_object.')
        with self.assertRaises(ValueError):
            ModelConfigParamsChecker(task_object=None, params_config=self.default_params)
        logging.info('Finish test_model_config_check_init_with_none_task_object.')

    def test_model_config_check_init_with_none_params_config(self):
        """
        测试ModelConfigCheck类构造方法在params_config对象传None时的处理
        """
        logging.info('Start test_model_config_check_init_with_none_params_config.')
        with self.assertRaises(ValueError):
            ModelConfigParamsChecker(task_object=self.task, params_config=None)
        logging.info('Finish test_model_config_check_init_with_none_params_config.')

    def test_model_config_check_init_with_invalid_params_config_type(self):
        """
        测试ModelConfigCheck类构造方法在传入params_config对象类型不合理时的处理
        """
        logging.info('Start test_model_config_check_init_with_invalid_params_config_type.')
        params_config = ('learning_rate', 1e-4)
        with self.assertRaises(TypeError):
            self._get_model_config_checker_for_params(params_config=params_config)
        logging.info('Finish test_model_config_check_init_with_invalid_params_config_type.')

    def test_model_config_check_init_with_normal_params_config(self):
        """
        测试ModelConfigCheck类构造方法在传入正常params_config对象时的处理
        """
        logging.info('Start test_model_config_check_init_with_normal_params_config.')
        model_config_check = self._get_model_config_checker_for_params(params_config=self.default_params)
        self.assertEqual(model_config_check.task_object, self.task)
        self.assertEqual(model_config_check.params, self.default_params)
        logging.info('Finish test_model_config_check_init_with_normal_params_config.')

    def test_model_config_check_with_params_config_containing_none_key(self):
        """
        测试ModelConfigCheck类在传入params_config信息中, 包含key为None的情况
        """
        logging.info('Start test_model_config_check_with_params_config_containing_none_key.')
        params_config = {None: 1e-4}
        model_config_check = self._get_model_config_checker_for_params(params_config=params_config)
        with self.assertRaises(ModelConfigParamsInfoError):
            model_config_check.check()
        logging.info('Finish test_model_config_check_with_params_config_containing_none_key.')

    def test_model_config_check_with_params_config_containing_none_value(self):
        """
        测试ModelConfigCheck类在传入params_config信息中, 包含value为None的情况
        """
        logging.info('Start test_model_config_check_with_params_config_containing_none_value.')
        params_config = {'learning_rate': None}
        model_config_check = self._get_model_config_checker_for_params(params_config=params_config)
        with self.assertRaises(ModelConfigParamsInfoError):
            model_config_check.check()
        logging.info('Finish test_model_config_check_with_params_config_containing_none_value.')

    def test_model_config_check_with_params_config_containing_invalid_value_type(self):
        """
        测试ModelConfigCheck类在传入params_config信息中, value值包含不合理类型值的情况
        """
        logging.info('Start test_model_config_check_with_params_config_containing_invalid_value_type.')
        params_config = {'learning_rate': {'lr': 1e-4}}
        model_config_check = self._get_model_config_checker_for_params(params_config=params_config)
        with self.assertRaises(ModelConfigParamsInfoError):
            model_config_check.check()
        logging.info('Finish test_model_config_check_with_params_config_containing_invalid_value_type.')

    def test_model_config_check_with_params_config_containing_duplicate_key_and_long_key(self):
        """
        测试ModelConfigCheck类在传入params_config信息中, 存在重复key值以及超长key值的情况
        """
        logging.info('Start test_model_config_check_with_params_config_containing_duplicate_key_and_long_key.')
        tmp_path = os.path.join('/', 'tmp')
        long_key = 'a' * (MODEL_CONFIG_LEN_LIMIT + 1)
        params_config = {'data_path': tmp_path, long_key: 'normal_value'}
        model_config_check = self._get_model_config_checker_for_params(params_config=params_config)
        with self.assertRaises(ModelConfigParamsInfoError):
            model_config_check.check()
        logging.info('Finish test_model_config_check_with_params_config_containing_duplicate_key_and_long_key.')

    def test_model_config_check_with_params_config_containing_long_value(self):
        """
        测试ModelConfigCheck类在传入params_config信息中, 存在超长value值的情况
        """
        logging.info('Start test_model_config_check_with_params_config_containing_long_value.')
        long_value = 'a' * (MODEL_CONFIG_LEN_LIMIT + 1)
        params_config = {'learning_rate': long_value}
        model_config_check = self._get_model_config_checker_for_params(params_config=params_config)
        with self.assertRaises(ModelConfigParamsInfoError):
            model_config_check.check()
        logging.info('Finish test_model_config_check_with_params_config_containing_long_value.')

    def test_model_config_check_with_params_config_containing_invalid_char_in_key(self):
        """
        测试ModelConfigCheck类在传入params_config信息中, key值中存在非法字符的情况
        """
        logging.info('Start test_model_config_check_with_params_config_containing_invalid_char_in_key.')

        for char in MODEL_CONFIG_PARAMS_CHARACTER_BLACKLIST:
            invalid_key = 'learning$rate'.replace('$', char)
            logging.info(f'Invalid params key: {invalid_key}')
            params_config = {invalid_key: 1e-4}
            model_config_check = self._get_model_config_checker_for_params(params_config=params_config)
            with self.assertRaises(ModelConfigParamsInfoError):
                model_config_check.check()
        logging.info('Finish test_model_config_check_with_params_config_containing_invalid_char_in_key.')

    def test_model_config_check_with_params_config_containing_invalid_char_in_value(self):
        """
        测试ModelConfigCheck类在传入params_config信息中, value值中存在非法字符的情况
        """
        logging.info('Start test_model_config_check_with_params_config_containing_invalid_char_in_value.')

        for char in MODEL_CONFIG_PARAMS_CHARACTER_BLACKLIST:
            invalid_value = '1e$-4'.replace('$', char)
            logging.info(f'Invalid params value: {invalid_value}')
            params_config = {'learning_rate': invalid_value}
            model_config_check = self._get_model_config_checker_for_params(params_config=params_config)
            with self.assertRaises(ModelConfigParamsInfoError):
                model_config_check.check()
        logging.info('Finish test_model_config_check_with_params_config_containing_invalid_char_in_value.')

    def test_model_config_check_with_params_config_containing_invalid_prefix_in_key(self):
        """
        测试ModelConfigCheck类在传入params_config信息中, key值存在不合理前缀的情况
        """
        logging.info('Start test_model_config_check_with_params_config_containing_invalid_prefix_in_key.')
        params_config = {'-learning_rate': 1e-4}
        model_config_check = self._get_model_config_checker_for_params(params_config=params_config)
        with self.assertRaises(ModelConfigParamsInfoError):
            model_config_check.check()
        logging.info('Finish test_model_config_check_with_params_config_containing_invalid_prefix_in_key.')

    def test_model_config_check_with_params_config_containing_invalid_prefix_in_value(self):
        """
        测试ModelConfigCheck类在传入params_config信息中, value值存在不合理前缀的情况
        """
        logging.info('Start test_model_config_check_with_params_config_containing_invalid_prefix_in_value.')
        params_config = {'learning_rate': '--1e-4'}
        model_config_check = self._get_model_config_checker_for_params(params_config=params_config)
        with self.assertRaises(ModelConfigParamsInfoError):
            model_config_check.check()
        logging.info('Finish test_model_config_check_with_params_config_containing_invalid_prefix_in_value.')

    def test_model_config_check_with_normal_params_config(self):
        """
        测试ModelConfigCheck类在传入params_config信息中, 传入正常params_config内容的情况
        """
        logging.info('Start test_model_config_check_with_normal_params_config.')
        params_config = {'learning_rate': 1e-4, 'batch_size': 32}
        model_config_check = self._get_model_config_checker_for_params(params_config=params_config)
        model_config_check.check()
        self.assertIn('--learning_rate', model_config_check.task_object.command_params)
        self.assertEqual(model_config_check.task_object.command_params[
                             model_config_check.task_object.command_params.index('--learning_rate') + 1], '0.0001')
        self.assertIn('--batch_size', model_config_check.task_object.command_params)
        self.assertEqual(model_config_check.task_object.command_params[
                             model_config_check.task_object.command_params.index('--batch_size') + 1], '32')
        logging.info('Finish test_model_config_check_with_normal_params_config.')

    def _get_model_config_checker_for_params(self, params_config):
        """
        获取ModelConfigParamsChecker对象
        :param params_config: params_config配置信息
        :return: checker对象
        """
        return ModelConfigParamsChecker(task_object=self.task, params_config=params_config)


if __name__ == '__main__':
    pytest.main(['-s', os.path.abspath(__file__)])
