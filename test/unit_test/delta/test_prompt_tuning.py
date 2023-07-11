#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2010-2022. All rights reserved.

import os
import logging
import unittest

import mindspore
import pytest
from mindspore.common.tensor import Tensor

from tk.delta.prompt_tuning import PromptTuning

logging.getLogger().setLevel(logging.INFO)
mindspore.set_context(device_id=1)

class TestPromptTuning(unittest.TestCase):
    # _check_num
    def test_check_num_with_zero_num_virtual_tokens(self):
        logging.info('Start test_check_num_with_zero_num_virtual_tokens.')
        with self.assertRaises(ValueError) as ex:
            PromptTuning(num_virtual_tokens=0, token_dim=1, num_transformer_submodules=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_zero_num_virtual_tokens.')

    def test_check_num_with_float_num_virtual_tokens(self):
        logging.info('Start test_check_num_with_float_num_virtual_tokens.')
        with self.assertRaises(TypeError) as ex:
            PromptTuning(num_virtual_tokens=1.5, token_dim=1, num_transformer_submodules=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_float_num_virtual_tokens.')

    def test_check_num_with_negative_num_virtual_tokens(self):
        logging.info('Start test_check_num_with_negative_num_virtual_tokens.')
        with self.assertRaises(ValueError) as ex:
            PromptTuning(num_virtual_tokens=-1, token_dim=1, num_transformer_submodules=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_negative_num_virtual_tokens.')

    def test_check_num_with_str_num_virtual_tokens(self):
        logging.info('Start test_check_num_with_str_num_virtual_tokens.')
        with self.assertRaises(TypeError) as ex:
            PromptTuning(num_virtual_tokens='a', token_dim=1, num_transformer_submodules=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_str_num_virtual_tokens.')

    def test_check_num_with_bool_num_virtual_tokens(self):
        logging.info('Start test_check_num_with_bool_num_virtual_tokens.')
        with self.assertRaises(TypeError) as ex:
            PromptTuning(num_virtual_tokens=True, token_dim=1, num_transformer_submodules=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_bool_num_virtual_tokens.')

    def test_check_num_with_zero_token_dim(self):
        logging.info('Start test_check_num_with_zero_token_dim.')
        with self.assertRaises(ValueError) as ex:
            PromptTuning(num_virtual_tokens=1, token_dim=0, num_transformer_submodules=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_zero_token_dim.')

    def test_check_num_with_float_token_dim(self):
        logging.info('Start test_check_num_with_float_token_dim.')
        with self.assertRaises(TypeError) as ex:
            PromptTuning(num_virtual_tokens=1, token_dim=1.5, num_transformer_submodules=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_float_token_dim.')

    def test_check_num_with_negative_token_dim(self):
        logging.info('Start test_check_num_with_negative_token_dim.')
        with self.assertRaises(ValueError) as ex:
            PromptTuning(num_virtual_tokens=1, token_dim=-1, num_transformer_submodules=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_negative_token_dim.')

    def test_check_num_with_str_token_dim(self):
        logging.info('Start test_check_num_with_str_token_dim.')
        with self.assertRaises(TypeError) as ex:
            PromptTuning(num_virtual_tokens=1, token_dim='a', num_transformer_submodules=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_str_token_dim.')

    def test_check_num_with_bool_token_dim(self):
        logging.info('Start test_check_num_with_bool_token_dim.')
        with self.assertRaises(TypeError) as ex:
            PromptTuning(num_virtual_tokens=1, token_dim=False, num_transformer_submodules=1)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_bool_token_dim.')

    def test_check_num_with_zero_num_transformer_submodules(self):
        logging.info('Start test_check_num_with_zero_num_transformer_submodules.')
        with self.assertRaises(ValueError) as ex:
            PromptTuning(num_virtual_tokens=1, token_dim=1, num_transformer_submodules=0)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_zero_num_transformer_submodules.')

    def test_check_num_with_float_num_transformer_submodules(self):
        logging.info('Start test_check_num_with_float_num_transformer_submodules.')
        with self.assertRaises(TypeError) as ex:
            PromptTuning(num_virtual_tokens=1, token_dim=1, num_transformer_submodules=1.5)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_float_num_transformer_submodules.')

    def test_check_num_with_negative_num_transformer_submodules(self):
        logging.info('Start test_check_num_with_negative_num_transformer_submodules.')
        with self.assertRaises(ValueError) as ex:
            PromptTuning(num_virtual_tokens=1, token_dim=1, num_transformer_submodules=-1)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_negative_num_transformer_submodules.')

    def test_check_num_with_str_num_transformer_submodules(self):
        logging.info('Start test_check_num_with_str_num_transformer_submodules.')
        with self.assertRaises(TypeError) as ex:
            PromptTuning(num_virtual_tokens=1, token_dim=1, num_transformer_submodules='a')
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_str_num_transformer_submodules.')

    def test_check_num_with_bool_num_transformer_submodules(self):
        logging.info('Start test_check_num_with_bool_num_transformer_submodules.')
        with self.assertRaises(TypeError) as ex:
            PromptTuning(num_virtual_tokens=1, token_dim=1, num_transformer_submodules=True)
        logging.error(ex.exception)
        logging.info('Finish test_check_num_with_bool_num_transformer_submodules.')

    def test_params_with_legal_tk_delta_prompttuning_embedding(self):
        logging.info('Start test_params_with_legal_tk_delta_prompttuning_embedding')
        prompttuning = PromptTuning(num_virtual_tokens=1, token_dim=1, num_transformer_submodules=1)
        target = Tensor([[1], [1]]).asnumpy() == prompttuning.tk_delta_prompttuning_embedding.asnumpy()
        for result in target:
            self.assertTrue(result)
        logging.info("Finish test_params_with_legal_tk_delta_prompttuning_embedding")


if __name__ == '__main__':
    pytest.main(["-s", os.path.abspath(__file__)])
