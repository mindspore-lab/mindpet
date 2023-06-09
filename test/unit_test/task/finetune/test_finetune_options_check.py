#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging
import os
import unittest
import pytest
from tk.task.finetune.finetune_options_check import FinetuneOptionsCheck, FinetuneOptionsCheckParam
from tk.security.param_check.option_check_utils import PathLengthCheckParam, PathContentCheckParam
from tk.utils.constants import DEFAULT_PATH_LEN_MIN_LIMIT, DEFAULT_PATH_LEN_MAX_LIMIT, PATH_MODE_LIMIT, \
    DEFAULT_FLAGS, DEFAULT_MODES, DEFAULT_FILE_LEN_MIN_LIMIT, DEFAULT_FILE_LEN_MAX_LIMIT


def base_check(disk_free_space_check):
    """
    finetune接口参数通用校验
    :param disk_free_space_check: 是否启动路径所在磁盘剩余空间校验
    :return: 校验结果对象
    """
    path = os.path.join('/', 'tmp', 'finetune_options_check')

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    os.chmod(path, 0o750)

    file_path = os.path.join(path, 'finetune_options_check.txt')

    with os.fdopen(os.open(file_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
        file.write('finetune options check file content.')

    option = FinetuneOptionsCheck(
        option_name='test_finetune_options_check', option_value=file_path, disk_space_check=disk_free_space_check)

    path_length_check_param = PathLengthCheckParam(path_min_limit=DEFAULT_PATH_LEN_MIN_LIMIT,
                                                   path_max_limit=DEFAULT_PATH_LEN_MAX_LIMIT,
                                                   file_min_limit=DEFAULT_FILE_LEN_MIN_LIMIT,
                                                   file_max_limit=DEFAULT_FILE_LEN_MAX_LIMIT)

    path_content_check_param = PathContentCheckParam(base_whitelist_mode='ALL',
                                                     extra_whitelist=['/', '_', '.'])

    check_param = FinetuneOptionsCheckParam(path_length_check_param=path_length_check_param,
                                            path_content_check_param=path_content_check_param,
                                            mode=PATH_MODE_LIMIT,
                                            path_including_file=True,
                                            force_quit=True,
                                            quiet=True)

    option.check(check_param)

    if os.path.exists(file_path):
        os.remove(file_path)

    if os.path.exists(path):
        os.rmdir(path)

    return option


class TestFinetuneOptionsCheck(unittest.TestCase):
    def test_finetune_options_check_with_disk_free_space_check(self):
        """
        测试finetune接口参数通用校验, 在开启路径剩余空间校验的情况
        """
        logging.info('Start test_finetune_options_check_with_disk_free_space_check.')
        base_check(disk_free_space_check=True)
        logging.info('Finish test_finetune_options_check_with_disk_free_space_check.')

    def test_finetune_options_check_without_disk_free_space_check(self):
        """
        测试finetune接口参数通用校验, 在不开启路径剩余空间校验的情况
        """
        logging.info('Start test_finetune_options_check_without_disk_free_space_check.')
        base_check(disk_free_space_check=False)
        logging.info('Finish test_finetune_options_check_without_disk_free_space_check.')


if __name__ == '__main__':
    pytest.main(['-s', os.path.abspath(__file__)])
