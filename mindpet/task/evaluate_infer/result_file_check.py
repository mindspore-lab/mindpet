#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved."""

from mindpet.security.param_check.base_check import BaseCheckParam, BaseCheck
from mindpet.security.param_check.option_check_utils import PathExistCheck, LinkPathCheck, get_real_path


class ResultFileCheckParam(BaseCheckParam):
    """ResultFileCheckParam class"""
    def __init__(self,
                 path_content_check_param,
                 mode,
                 path_including_file,
                 force_quit,
                 quiet):
        """
        微调入参校验参数
        :param path_content_check_param: 路径内容校验参数
        :param mode: 路径权限范围约束
        :param path_including_file: 路径是否为文件路径
        :param force_quit: SDK场景下, 路径权限提升校验异常时是否强制退出
        """
        super().__init__(mode, path_including_file, force_quit, quiet)
        self.path_content_check_param = path_content_check_param


# pylint: disable=W0246
class ResultFileCheck(BaseCheck):
    """ResultFileCheck class"""
    def check(self, check_param):
        """
        任务结果文件(eval_result.json/infer_result.json)合法性校验
        :param check_param: 任务结果文件校验参数
        """
        # 针对原始路径进行的校验
        self._origin_path_check_item()

        # 获取真实路径
        self.option_value = get_real_path(self.option_value)

        # 针对真实路径进行的校验
        self.base_check_item(check_param)

    def _origin_path_check_item(self):
        # 路径真实性校验
        PathExistCheck(option_name=self.option_name, option_value=self.option_value)

        # 路径软链接校验
        LinkPathCheck(option_name=self.option_name, option_value=self.option_value)
