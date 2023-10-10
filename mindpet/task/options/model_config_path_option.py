#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""model config path option"""

import click
from click.exceptions import BadParameter
from mindpet.utils.constants import PATH_MODE_LIMIT
from mindpet.task.finetune.finetune_options_check import FinetuneOptionsCheckParam, FinetuneOptionsCheck
from mindpet.task.options.path_check_param import default_file_length_check_param, default_file_content_check_param


class ModelConfigPathOption(click.core.Option):
    """ModelConfigPathOption class"""
    def __init__(self):
        super().__init__(
            param_decls=('-mc', '--model_config_path'),
            type=str,
            default=None,
            show_default=True,
            help='model config file local path',
            callback=self.model_config_path_callback
        )

    @staticmethod
    def _is_yaml_file(path):
        """
        路径是否是yaml文件
        :param path: 指定路径
        :return: 路径是否是yaml文件
        """
        return path.endswith('.yaml') or path.endswith('.yml')

    def model_config_path_callback(self, ctx, param, value):
        """
        model_config_path参数click回调方法

        :param ctx: 上下文信息
        :param param: 参数属性
        :param value: 输入值
        :return: 回调处理后的参数值
        """
        if value is None:
            return value

        check_param = FinetuneOptionsCheckParam(path_length_check_param=default_file_length_check_param,
                                                path_content_check_param=default_file_content_check_param,
                                                mode=PATH_MODE_LIMIT,
                                                path_including_file=True,
                                                quiet=ctx.params.get('quiet', False))
        model_config_path = FinetuneOptionsCheck(option_name=param.name, option_value=value)
        model_config_path.check(check_param)

        if not self._is_yaml_file(model_config_path.option_value):
            raise BadParameter('Param [model_config_path] only support .yaml or .yml file path.')

        return model_config_path.option_value
