#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""base check module."""
from mindpet.security.param_check.option_check_utils import PathRightEscalationCheck, FileSizeCheck


class BaseCheckParam:
    """base param check class."""
    def __init__(self, mode, path_including_file, force_quit, quiet):
        """
        入参校验参数
        :param mode: 路径权限范围约束
        :param path_including_file: 路径是否为文件路径
        :param force_quit: SDK场景下, 路径权限提升校验异常时是否强制退出
        :param quiet: CLI场景下，路径权限提升校验是否忽略交互式过程
        """
        self.mode = mode
        self.path_including_file = path_including_file
        self.force_quit = force_quit
        self.quiet = quiet


class BaseCheck:
    """base check class."""
    def __init__(self, option_name, option_value):
        """
        基础校验项构造方法
        :param option_name: 入参名称
        :param option_value: 入参值
        """
        self.option_name = option_name
        self.option_value = option_value

    def check(self, check_param):
        """
        微调入参校验
        :param check_param: 微调入参校验参数
        """
        self.base_check_item(check_param)

    def base_check_item(self, check_param):
        """
        基础校验项
        :param check_param: 校验参数
        """
        # 路径权限提升校验
        PathRightEscalationCheck(option_name=self.option_name,
                                 option_value=self.option_value,
                                 mode=check_param.mode,
                                 force_quit=check_param.force_quit,
                                 quiet=check_param.quiet)

        # 路径文件大小校验
        FileSizeCheck(option_name=self.option_name,
                      option_value=self.option_value,
                      path_including_file=check_param.path_including_file)
