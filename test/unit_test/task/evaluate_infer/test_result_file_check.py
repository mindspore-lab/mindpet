#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging
import os
import shutil
import stat
import json
import unittest
import pytest
from tk.utils.constants import SPACE_CHARACTER, PATH_MODE_LIMIT
from tk.security.param_check.option_check_utils import PathContentCheckParam
from tk.task.evaluate_infer.result_file_check import ResultFileCheckParam, ResultFileCheck
from tk.utils.exceptions import LinkPathError, FileOversizeError, PathRightEscalationError


class TestEvaluateResultCheck(unittest.TestCase):
    def test_path_existence_check(self):
        """
        测试evaluate/infer结果json文件校验, 路径存在性校验
        """
        logging.info('Start test_path_existence_check.')

        test_value = os.path.join('/', 'tmp', 'evaluate_result.json')

        base_whitelist_mode = 'ALL'
        extra_whitelist = ['.', '/', '-', '_', SPACE_CHARACTER]

        eval_result_check = ResultFileCheck(option_name='eval_path_existence_check',
                                            option_value=test_value)

        eval_path_content_check_param = PathContentCheckParam(base_whitelist_mode=base_whitelist_mode,
                                                              extra_whitelist=extra_whitelist)

        eval_check_param = ResultFileCheckParam(path_content_check_param=eval_path_content_check_param,
                                                mode=PATH_MODE_LIMIT,
                                                path_including_file=True,
                                                force_quit=True,
                                                quiet=False)

        with self.assertRaises(FileNotFoundError):
            eval_result_check.check(eval_check_param)

        logging.info('Finish test_path_existence_check.')

    def test_link_path_check(self):
        """
        测试evaluate/infer结果json文件校验, 软链接路径校验
        """
        logging.info('Start test_link_path_check.')

        root_path = os.path.join('/', 'tmp', 'evaluate_link_path_check')

        if not os.path.exists(root_path):
            os.makedirs(root_path)

        source_file_path = os.path.join(root_path, 'source_evaluate_result.json')

        if os.path.exists(source_file_path):
            os.remove(source_file_path)

        flags = os.O_RDWR | os.O_CREAT  # 允许读写, 文件不存在时新建
        modes = stat.S_IWUSR | stat.S_IRUSR  # 所有者读写
        file_content = json.dumps({'test_key': 'test_value'})

        with os.fdopen(os.open(source_file_path, flags, modes), 'w') as file:
            file.write(file_content)

        target_file_path = os.path.join(root_path, 'target_evaluate_result.json')

        if os.path.exists(target_file_path):
            os.remove(target_file_path)

        os.symlink(source_file_path, target_file_path)

        base_whitelist_mode = 'ALL'
        extra_whitelist = ['.', '/', '-', '_', SPACE_CHARACTER]

        eval_result_check = ResultFileCheck(option_name='eval_path_existence_check',
                                            option_value=target_file_path)

        eval_path_content_check_param = PathContentCheckParam(base_whitelist_mode=base_whitelist_mode,
                                                              extra_whitelist=extra_whitelist)

        eval_check_param = ResultFileCheckParam(path_content_check_param=eval_path_content_check_param,
                                                mode=PATH_MODE_LIMIT,
                                                path_including_file=True,
                                                force_quit=True,
                                                quiet=False)

        with self.assertRaises(LinkPathError):
            eval_result_check.check(eval_check_param)

        if os.path.exists(root_path):
            shutil.rmtree(root_path)

        logging.info('Finish test_link_path_check.')

    def test_path_right_escalation_check(self):
        """
        测试evaluate/infer结果json文件校验, 路径权限提升校验
        """
        logging.info('Start test_path_right_escalation_check.')

        root_path = os.path.join('/', 'tmp', 'evaluate_path_right_escalation_check')
        test_value = os.path.join(root_path, 'evaluate_result.json')

        if not os.path.exists(root_path):
            os.makedirs(root_path)

        if os.path.exists(test_value):
            os.remove(test_value)

        flags = os.O_RDWR | os.O_CREAT  # 允许读写, 文件不存在时新建
        modes = stat.S_IWUSR | stat.S_IRUSR  # 所有者读写
        file_content = json.dumps({'test_key': 'test_value'})

        with os.fdopen(os.open(test_value, flags, modes), 'w') as file:
            file.write(file_content)

        mode_777 = 0o777
        os.chmod(test_value, mode_777)

        base_whitelist_mode = 'ALL'
        extra_whitelist = ['.', '/', '-', '_', SPACE_CHARACTER]

        eval_result_check = ResultFileCheck(option_name='evaluate_path_right_escalation_check',
                                            option_value=test_value)

        eval_path_content_check_param = PathContentCheckParam(base_whitelist_mode=base_whitelist_mode,
                                                              extra_whitelist=extra_whitelist)

        eval_check_param = ResultFileCheckParam(path_content_check_param=eval_path_content_check_param,
                                                mode=PATH_MODE_LIMIT,
                                                path_including_file=True,
                                                force_quit=True,
                                                quiet=False)

        with self.assertRaises(PathRightEscalationError):
            eval_result_check.check(eval_check_param)

        if os.path.exists(root_path):
            shutil.rmtree(root_path)

        logging.info('Finish test_path_right_escalation_check.')

    def test_file_size_check(self):
        """
        测试evaluate/infer结果json文件校验, 文件大小校验
        """
        logging.info('Start test_file_size_check.')

        root_path = os.path.join('/', 'tmp', 'evaluate_path_right_escalation_check')
        test_value = os.path.join(root_path, 'evaluate_result.json')

        if not os.path.exists(root_path):
            os.makedirs(root_path)

        if os.path.exists(test_value):
            os.remove(test_value)

        flags = os.O_RDWR | os.O_CREAT  # 允许读写, 文件不存在时新建
        modes = stat.S_IWUSR | stat.S_IRUSR  # 所有者读写

        content_info = '*' * 1024 * 1024 * 1024

        file_content = json.dumps({'test_key': content_info})

        with os.fdopen(os.open(test_value, flags, modes), 'w') as file:
            file.write(file_content)

        mode_750 = 0o750
        os.chmod(test_value, mode_750)

        base_whitelist_mode = 'ALL'
        extra_whitelist = ['.', '/', '-', '_', SPACE_CHARACTER]

        eval_result_check = ResultFileCheck(option_name='evaluate_file_size_check',
                                            option_value=test_value)

        eval_path_content_check_param = PathContentCheckParam(base_whitelist_mode=base_whitelist_mode,
                                                              extra_whitelist=extra_whitelist)

        eval_check_param = ResultFileCheckParam(path_content_check_param=eval_path_content_check_param,
                                                mode=PATH_MODE_LIMIT,
                                                path_including_file=True,
                                                force_quit=True,
                                                quiet=False)

        with self.assertRaises(FileOversizeError):
            eval_result_check.check(eval_check_param)

        if os.path.exists(root_path):
            shutil.rmtree(root_path)

        logging.info('Finish test_file_size_check.')


if __name__ == '__main__':
    pytest.main(['-s', os.path.abspath(__file__)])
