#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""boot file path option"""
import os
import click
from click.exceptions import MissingParameter, BadParameter
from mindpet.utils.constants import PATH_MODE_LIMIT
from mindpet.task.finetune.finetune_options_check import FinetuneOptionsCheckParam, FinetuneOptionsCheck
from mindpet.task.options.path_check_param import default_file_length_check_param, default_file_content_check_param


class BootFilePathOption(click.core.Option):
    """BootFilePathOption class"""
    def __init__(self):
        super().__init__(
            param_decls=('-bp', '--boot_file_path'),
            type=str,
            help='boot file local path',
            callback=self.boot_file_path_callback
        )

    @staticmethod
    def _is_py_file(path):
        """
        路径是否是python文件

        :param path: 指定路径
        :return: 路径是否是python文件
        """
        return os.path.isfile(path) and path.endswith('.py')

    def boot_file_path_callback(self, ctx, param, value):
        """
        boot_file_path参数click回调方法

        :param ctx: 上下文信息
        :param param: 参数属性
        :param value: 输入值
        :return: 回调处理后的参数值
        """
        if value is None:
            raise MissingParameter('Param [boot_file_path] is required.')

        check_param = FinetuneOptionsCheckParam(path_length_check_param=default_file_length_check_param,
                                                path_content_check_param=default_file_content_check_param,
                                                mode=PATH_MODE_LIMIT,
                                                path_including_file=True,
                                                force_quit=True,
                                                quiet=ctx.params.get('quiet', False))
        boot_file_path_option = FinetuneOptionsCheck(option_name=param.name, option_value=value)
        boot_file_path_option.check(check_param)

        if not self._is_py_file(boot_file_path_option.option_value):
            raise BadParameter('Param [boot_file_path] only support .py file path.')

        return boot_file_path_option.option_value
