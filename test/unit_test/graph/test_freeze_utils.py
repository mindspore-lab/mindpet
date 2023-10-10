#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

"""
功能: 冻结功能单元测试模块
"""
import sys
sys.path.append('.')
import logging
import os
import unittest
from fnmatch import fnmatch

import pytest
from mindspore import nn
from mindpet.graph.freeze_utils import freeze_modules, freeze_delta, freeze_from_config
from mindpet.utils.exceptions import ModelConfigFreezeInfoError
from mindformers.modules import Transformer

logging.getLogger().setLevel(logging.INFO)

CURRENT_FILE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'resource')


class TestFreezeUtils(unittest.TestCase):
    def setUp(self):
        self._global_model = SimpleNetwork()
        self._global_include = ['*embedding*', 'transformer*', 'dense.weight']
        self._global_exclude = ['transformer.encoder.blocks.*.layernorm*']
        self._global_config_path = os.path.join(CURRENT_FILE_PATH, 'test_freeze_config_file.yaml')

    ####################################################################################################
    # freeze_modules接口单元测试用例
    ####################################################################################################
    def test_freeze_modules_include_exclude(self):
        """
        测试freeze_modules接口中同时传入include和exclude的场景。
        结果应该是冻结以下几个模块：
        1. embedding.embedding_table
        2. 以transformer开头，但不匹配transformer.encoder.blocks.*.layernorm*的模块
        3. dense.weight
        """
        logging.info('Start test_freeze_modules_include_exclude.')
        model = SimpleNetwork()
        freeze_modules(model, include=self._global_include, exclude=self._global_exclude)
        for name, param in model.parameters_and_names():
            if 'embedding.embedding_table' == name or 'dense.weight' == name:
                self.assertFalse(param.requires_grad)
            elif name.startswith('transformer') and not fnmatch(name, 'transformer.encoder.blocks.*.layernorm*'):
                self.assertFalse(param.requires_grad)
            else:
                self.assertTrue(param.requires_grad)
        logging.info('Finish test_freeze_modules_include_exclude.')

    def test_freeze_modules_include(self):
        """
        测试freeze_modules接口中只传入include的场景。
        结果应该是冻结以下几个模块：
        1. embedding.embedding_table
        2. 以transformer开头的模块
        3. dense.weight
        """
        logging.info('Start test_freeze_modules_include.')
        model = SimpleNetwork()
        freeze_modules(model, include=self._global_include)
        for name, param in model.parameters_and_names():
            if 'embedding.embedding_table' == name or 'dense.weight' == name or name.startswith('transformer'):
                self.assertFalse(param.requires_grad)
            else:
                self.assertTrue(param.requires_grad)
        logging.info('Finish test_freeze_modules_include.')

    def test_freeze_modules_exclude(self):
        """
        测试freeze_modules接口中只传入exclude的场景。实际上只传入exclude没有意义，因为原模型本身是没有冻结的。
        结果应该是所有模块都没有冻结。
        """
        logging.info('Start test_freeze_modules_exclude.')
        model = SimpleNetwork()
        freeze_modules(model, exclude=self._global_exclude)
        for _, param in model.parameters_and_names():
            self.assertTrue(param.requires_grad)
        logging.info('Finish test_freeze_modules_exclude.')

    def test_freeze_modules_model_not_cell(self):
        """
        测试freeze_modules接口中model参数不是cell类型的异常场景
        """
        logging.info('Start test_freeze_modules_model_not_cell.')
        with self.assertRaises(TypeError):
            freeze_modules(123)
        logging.info('Finish test_freeze_modules_model_not_cell.')

    def test_freeze_modules_include_and_exclude_all_none(self):
        """
        测试freeze_modules接口中同时不传include和exclude的异常场景。
        """
        logging.info('Start test_freeze_modules_not_include_and_exclude.')
        with self.assertRaises(ValueError):
            freeze_modules(self._global_model)
        logging.info('Finish test_freeze_modules_not_include_and_exclude.')

    def test_freeze_modules_include_not_list(self):
        """
        测试freeze_modules接口中include不是list类型的异常场景。
        """
        logging.info('Start test_freeze_modules_include_not_list.')
        with self.assertRaises(TypeError):
            freeze_modules(self._global_model, include='*')
        logging.info('Finish test_freeze_modules_include_not_list.')

    def test_freeze_modules_include_empty_list(self):
        """
        测试freeze_modules接口中include是空list的异常场景。
        """
        logging.info('Start test_freeze_modules_include_empty_list.')
        with self.assertRaises(TypeError):
            freeze_modules(self._global_model, include=[])
        logging.info('Finish test_freeze_modules_include_empty_list.')

    def test_freeze_modules_include_is_0(self):
        """
        测试freeze_modules接口中include不是list类型的异常场景。
        """
        logging.info('Start test_freeze_modules_include_is_0.')
        with self.assertRaises(TypeError):
            freeze_modules(self._global_model, include=0)
        logging.info('Finish test_freeze_modules_include_is_0.')

    def test_freeze_modules_include_list_item_not_str(self):
        """
        测试freeze_modules接口中include不是list类型的异常场景。
        """
        logging.info('Start test_freeze_modules_include_list_item_not_str.')
        with self.assertRaises(TypeError):
            freeze_modules(self._global_model, include=[123, '*'])
        logging.info('Finish test_freeze_modules_include_list_item_not_str.')

    ####################################################################################################
    # freeze_delta接口单元测试用例
    ####################################################################################################
    def test_freeze_delta_lora_exclude(self):
        """
        测试freeze_delta接口中，传入mode=lora，以及exclude的场景。
        结果应该是冻结除了以下几个模块：
        1. 名字带lora的模块
        2. 匹配transformer.encoder.blocks.*.layernorm*的模块
        """
        logging.info('Start test_freeze_delta_lora_exclude.')
        mode = 'lora'
        freeze_delta(self._global_model, mode, exclude=self._global_exclude)
        for name, param in self._global_model.parameters_and_names():
            if 'lora' in name or \
                    fnmatch(name, 'transformer.encoder.blocks.*.layernorm*'):
                self.assertTrue(param.requires_grad)
            else:
                self.assertFalse(param.requires_grad)
        logging.info('Finish test_freeze_delta_lora_exclude.')

    def test_freeze_delta_prefixtuning_exclude(self):
        """
        测试freeze_delta接口中，传入mode=prefixtuning，以及exclude的场景。
        结果应该是冻结除了以下几个模块：
        1. 名字带prefixtuning的模块
        2. 匹配transformer.encoder.blocks.*.layernorm*的模块
        """
        logging.info('Start test_freeze_delta_prefixtuning_exclude.')
        model = SimpleNetwork()
        mode = 'prefixtuning'
        freeze_delta(model, mode, exclude=self._global_exclude)
        for name, param in model.parameters_and_names():
            if 'prefixtuning' in name or \
                    fnmatch(name, 'transformer.encoder.blocks.*.layernorm*'):
                self.assertTrue(param.requires_grad)
            else:
                self.assertFalse(param.requires_grad)
        logging.info('Finish test_freeze_delta_prefixtuning_exclude.')

    def test_freeze_delta_bitfit_exclude(self):
        """
        测试freeze_delta接口中，传入mode=bitfit，以及exclude的场景。
        结果应该是除了以下几个模块，其他结构都被冻结：
        1. 名字带bias的模块
        2. 匹配transformer.encoder.blocks.*.layernorm*的模块
        """
        logging.info('Start test_freeze_delta_bitfit_exclude.')
        mode = 'bitfit'
        freeze_delta(self._global_model, mode, exclude=self._global_exclude)
        for name, param in self._global_model.parameters_and_names():
            if 'bias' in name or \
                    fnmatch(name, 'transformer.encoder.blocks.*.layernorm*'):
                self.assertTrue(param.requires_grad)
            else:
                self.assertFalse(param.requires_grad)
        logging.info('Finish test_freeze_delta_bitfit_exclude.')

    def test_freeze_delta_model_not_cell(self):
        """
        测试freeze_delta接口中model参数不是cell类型的异常场景
        """
        logging.info('Start test_freeze_delta_model_not_cell.')
        with self.assertRaises(TypeError):
            freeze_delta(123, 'lora')
        logging.info('Finish test_freeze_delta_model_not_cell.')

    def test_freeze_delta_model_mode_empty(self):
        """
        测试freeze_delta接口中mode参数是空值的异常场景
        """
        logging.info('Start test_freeze_delta_model_mode_empty.')
        model = SimpleNetwork()
        with self.assertRaises(ValueError):
            freeze_delta(model, '')
        logging.info('Finish test_freeze_delta_model_mode_empty.')

    def test_freeze_delta_model_mode_not_str(self):
        """
        测试freeze_delta接口中mode参数不是字符串的异常场景
        """
        logging.info('Start test_freeze_delta_model_mode_not_str.')
        model = SimpleNetwork()
        with self.assertRaises(ValueError):
            freeze_delta(model, 123)
        logging.info('Finish test_freeze_delta_model_mode_not_str.')

    def test_freeze_delta_model_mode_not_delta(self):
        """
        测试freeze_delta接口中mode参数是空值的异常场景
        """
        logging.info('Start test_freeze_delta_model_mode_not_delta.')
        model = SimpleNetwork()
        with self.assertRaises(ValueError):
            freeze_delta(model, 'delta')
        logging.info('Finish test_freeze_delta_model_mode_not_delta.')

    def test_freeze_delta_exp_from_freeze_modules(self):
        """
        测试freeze_modules接口中include不是list类型的异常场景。
        """
        logging.info('Start test_freeze_delta_exp_from_freeze_modules.')
        model = SimpleNetwork()
        with self.assertRaises(Exception):
            freeze_delta(model, 'lora', include='*')
        logging.info('Finish test_freeze_delta_exp_from_freeze_modules.')

    ####################################################################################################
    # freeze_from_config接口单元测试用例
    ####################################################################################################
    def test_freeze_from_config_correct_yaml(self):
        """
        测试freeze_from_config接口中，传入正确的配置文件。
        结果应该是冻结除了以下几个模块：
        1. embedding.embedding_table
        2. 以transformer开头，但不匹配transformer.encoder.blocks.*.layernorm*的模块
        3. dense.weight
        """
        logging.info('Start test_freeze_from_config_correct_yaml.')
        model = SimpleNetwork()
        freeze_from_config(model, self._global_config_path)
        for name, param in model.parameters_and_names():
            if 'embedding.embedding_table' == name or 'dense.weight' == name:
                self.assertFalse(param.requires_grad)
            elif name.startswith('transformer') and not fnmatch(name, 'transformer.encoder.blocks.*.layernorm*'):
                self.assertFalse(param.requires_grad)
            else:
                self.assertTrue(param.requires_grad)
        logging.info('Finish test_freeze_from_config_correct_yaml.')

    def test_freeze_from_config_model_not_cell(self):
        """
        测试freeze_from_config接口中model参数不是cell类型的异常场景
        """
        logging.info('Start test_freeze_from_config_model_not_cell.')
        with self.assertRaises(TypeError):
            freeze_from_config(123, self._global_config_path)
        logging.info('Finish test_freeze_from_config_model_not_cell.')

    def test_freeze_from_config_config_path_not_str(self):
        """
        测试freeze_from_config接口中config_path参数不是字符串的异常场景
        """
        logging.info('Start test_freeze_from_config_config_path_not_str.')
        with self.assertRaises(TypeError):
            freeze_from_config(self._global_model, 123)
        logging.info('Finish test_freeze_from_config_config_path_not_str.')

    def test_freeze_from_config_config_path_not_none(self):
        """
        测试freeze_from_config接口中config_path参数是None的异常场景
        """
        logging.info('Start test_freeze_from_config_config_path_not_str.')
        with self.assertRaises(TypeError):
            freeze_from_config(self._global_model, None)
        logging.info('Finish test_freeze_from_config_config_path_not_str.')

    def test_freeze_from_config_config_path_empty(self):
        """
        测试freeze_from_config接口中config_path参数为空字符串的异常场景
        """
        logging.info('Start test_freeze_from_config_config_path_empty.')
        with self.assertRaises(ValueError):
            freeze_from_config(self._global_model, '')
        logging.info('Finish test_freeze_from_config_config_path_empty.')

    def test_freeze_from_config_no_freeze_key(self):
        """
        测试freeze_from_config接口传入的yaml文件中没有freeze关键词的异常场景
        """
        logging.info('Start test_freeze_from_config_no_freeze_key.')
        config_path = os.path.join(CURRENT_FILE_PATH, 'test_freeze_config_file_no_freeze_key.yaml')
        with self.assertRaises(ModelConfigFreezeInfoError):
            freeze_from_config(self._global_model, config_path)
        logging.info('Finish test_freeze_from_config_no_freeze_key.')

    def test_freeze_from_config_no_include_and_exclude(self):
        """
        测试freeze_from_config接口传入的yaml文件中没有include和exclude关键词的异常场景
        """
        logging.info('Start test_freeze_from_config_no_include_and_exclude.')
        logging.info("CURRENT_FILE_PATH:%s", CURRENT_FILE_PATH)
        config_path = os.path.join(CURRENT_FILE_PATH, 'test_freeze_config_file_no_include_and_exclude.yaml')
        with self.assertRaises(ModelConfigFreezeInfoError):
            freeze_from_config(self._global_model, config_path)
        logging.info('Finish test_freeze_from_config_no_include_and_exclude.')

    def test_freeze_from_config_exp_from_freeze_modules(self):
        """
        测试freeze_from_config接口传入的yaml文件中include不是列表的异常场景
        """
        logging.info('Start test_freeze_from_config_exp_from_freeze_modules.')
        logging.info("CURRENT_FILE_PATH:%s", CURRENT_FILE_PATH)
        config_path = os.path.join(CURRENT_FILE_PATH, 'test_freeze_config_file_include_not_list.yaml')
        with self.assertRaises(Exception):
            freeze_from_config(self._global_model, config_path)
        logging.info('Finish test_freeze_from_config_exp_from_freeze_modules.')


class SimpleNetwork(nn.Cell):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.embedding = nn.Embedding(10, 5, True)
        self.transformer = Transformer(batch_size=2, encoder_layers=1, decoder_layers=2, hidden_size=64,
                                          ffn_hidden_size=64, src_seq_length=20, tgt_seq_length=10)
        self.mindpet_delta_lora = nn.Dense(10, 5)
        self.mindpet_delta_prefixtuning = nn.Dense(10, 5)
        self.dense = nn.Dense(10, 5)
        self.relu = nn.ReLU()

    def construct(self, simple_input):
        em_output = self.embedding(simple_input)
        trans_output = self.transformer(em_output)
        lora_output = self.lora(trans_output)
        dense_output = self.dense(lora_output)
        relu_output = self.relu(dense_output)
        return relu_output


if __name__ == '__main__':
    pytest.main(['-s', os.path.abspath(__file__)])
