#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""ckpt path option"""
import click
from click.exceptions import MissingParameter
from mindpet.utils.constants import PATH_MODE_LIMIT
from mindpet.task.finetune.finetune_options_check import FinetuneOptionsCheckParam, FinetuneOptionsCheck
from mindpet.task.options.path_check_param import default_folder_length_check_param, default_folder_content_check_param


class CkptPathOption(click.core.Option):
    """CkptPathOption class"""
    def __init__(self):
        super().__init__(
            param_decls=('-cp', '--ckpt_path'),
            type=str,
            help='checkpoint local path',
            callback=self.ckpt_path_callback
        )

    @staticmethod
    def ckpt_path_callback(ctx, param, value):
        """
        ckpt_path参数click回调方法

        :param ctx: 上下文信息
        :param param: 参数属性
        :param value: 输入值
        :return: 回调处理后的参数值
        """
        if value is None:
            raise MissingParameter('Param [ckpt_path] is required.')

        check_param = FinetuneOptionsCheckParam(path_length_check_param=default_folder_length_check_param,
                                                path_content_check_param=default_folder_content_check_param,
                                                mode=PATH_MODE_LIMIT,
                                                quiet=ctx.params.get('quiet', False))
        ckpt_path_option = FinetuneOptionsCheck(option_name=param.name, option_value=value)
        ckpt_path_option.check(check_param)

        return ckpt_path_option.option_value
