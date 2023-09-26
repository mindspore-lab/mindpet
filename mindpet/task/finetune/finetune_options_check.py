#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved."""

from mindpet.utils.constants import GB_SIZE
from mindpet.security.param_check.base_check import BaseCheckParam, BaseCheck
from mindpet.security.param_check.option_check_utils import PathContentBlacklistCharactersCheck, AbsolutePathCheck, \
    PathExistCheck, LinkPathCheck, PathContentLengthCheck, PathContentCharacterCheck, PathGranularityCheck, \
    DiskFreeSpaceCheck, get_real_path


class FinetuneOptionsCheckParam(BaseCheckParam):
    """class FinetuneOptionsCheckParam class"""
    def __init__(self,
                 path_length_check_param,
                 path_content_check_param,
                 mode,
                 path_including_file=False,
                 force_quit=False,
                 quiet=False):
        """
        微调入参校验参数
        :param path_length_check_param: 路径长度校验参数
        :param path_content_check_param: 路径内容校验参数
        :param mode: 路径权限范围约束
        :param path_including_file: 路径是否为文件路径
        :param force_quit: SDK场景下, 路径权限提升校验异常时是否强制退出
        :param quiet: CLI场景下，路径权限提升校验是否忽略交互式过程
        """
        super().__init__(mode, path_including_file, force_quit, quiet)
        self.path_length_check_param = path_length_check_param
        self.path_content_check_param = path_content_check_param


class FinetuneOptionsCheck(BaseCheck):
    """FinetuneOptionsCheck class"""
    def __init__(self, option_name, option_value, disk_space_check=False):
        super().__init__(option_name, option_value)
        self.disk_space_check = disk_space_check

    def check(self, check_param):
        """
        微调入参校验
        :param check_param: 微调入参校验参数
        """
        # 针对原始路径进行的校验
        self._origin_path_check_item()

        # 获取真实路径
        self.option_value = get_real_path(self.option_value)

        # 针对真实路径进行的校验
        self._real_path_check_item(check_param)

    def _origin_path_check_item(self):
        """_origin_path_check_item"""
        # 路径内容黑名单校验
        PathContentBlacklistCharactersCheck(option_name=self.option_name, option_value=self.option_value)

        # 绝对路径校验
        AbsolutePathCheck(option_name=self.option_name, option_value=self.option_value)

        # 路径真实性校验
        PathExistCheck(option_name=self.option_name, option_value=self.option_value)

        # 路径软链接校验
        LinkPathCheck(option_name=self.option_name, option_value=self.option_value)

    def _real_path_check_item(self, check_param):
        """_real_path_check_item"""
        # 路径长度校验
        PathContentLengthCheck(option_name=self.option_name, option_value=self.option_value,
                               path_min_limit=check_param.path_length_check_param.path_min_limit,
                               path_max_limit=check_param.path_length_check_param.path_max_limit,
                               file_min_limit=check_param.path_length_check_param.file_min_limit,
                               file_max_limit=check_param.path_length_check_param.file_max_limit)

        # 路径内容白名单校验
        PathContentCharacterCheck(option_name=self.option_name, option_value=self.option_value,
                                  base_whitelist_mode=check_param.path_content_check_param.base_whitelist_mode,
                                  extra_whitelist=check_param.path_content_check_param.extra_whitelist)

        # 路径粒度校验
        PathGranularityCheck(option_name=self.option_name, option_value=self.option_value,
                             path_including_file=check_param.path_including_file)

        self.base_check_item(check_param)

        # 路径所在磁盘空间耗尽风险校验
        if self.disk_space_check:
            DiskFreeSpaceCheck(option_name=self.option_name,
                               option_value=self.option_value,
                               free_space_limit=GB_SIZE,
                               force_quit=check_param.force_quit,
                               quiet=check_param.quiet)
