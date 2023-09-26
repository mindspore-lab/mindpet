#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

"""
定义冻结功能相关接口
"""

from typing import Optional, List
from fnmatch import fnmatch
from mindspore import nn

from mindpet.log.log import logger
from mindpet.utils.constants import DELTA_LIST
from mindpet.utils.exceptions import ReadYamlFileError, ModelConfigFreezeInfoError, LinkPathError, UnexpectedError
from mindpet.utils.io_utils import read_yaml_file


def freeze_modules(model: nn.Cell,
                   include: Optional[List[str]] = None,
                   exclude: Optional[List[str]] = None) -> None:
    """
    根据指定模块列表冻结网络。

    :param model: 需要冻结的模型实例， 必填。
    :param include: 需要冻结的模块名列表， 选填。
                    模糊匹配列表中所有模块名，挨个将匹配到的模块的requires_grad设置为False。
                    列表项支持配置符号*，代表任意字符串，格式如 ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']
    :param exclude: 不冻结的模块名列表， 选填。
                    模糊匹配列表中所有模块名，挨个将匹配到的模块的requires_grad设置为True。
                    列表项支持配置符号*，代表任意字符串，格式如 ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']
                    当include和exclude列表项冲突时，对该项匹配到的模块不做任何处理。
    """
    logger.info("Start to freeze model, include list: %s, exclude list: %s", include, exclude)

    if not isinstance(model, nn.Cell):
        raise TypeError("A Cell is required for argument 'model'.")

    if include is None and exclude is None:
        raise ValueError("Argument 'include' (pos 2) and 'exclude' (pos 3) can't be None together.")

    if include is not None and (not isinstance(include, List) or not include):
        raise TypeError("A non-empty list is required for argument 'include' (pos 2).")

    if exclude is not None and (not isinstance(exclude, List) or not exclude):
        raise TypeError("A non-empty list is required for argument 'exclude' (pos 3).")

    _freeze_from_list(model, include, exclude)
    logger.info("End to freeze model.")


def _freeze_from_list(model, include, exclude):
    """
    根据include/exclude列表冻结网络。
    """
    for name, param in model.parameters_and_names():
        if _match_str_and_list(name, include) and not _match_str_and_list(name, exclude):
            param.requires_grad = False
        elif not _match_str_and_list(name, include) and _match_str_and_list(name, exclude):
            param.requires_grad = True


def freeze_delta(model: nn.Cell,
                 mode: str,
                 include: Optional[List[str]] = None,
                 exclude: Optional[List[str]] = None) -> None:
    """
    根据微调算法类型以及指定模块列表冻结网络。
    目前已实现lora和prefixtuning两种微调算法的冻结模式。

    :param model: 需要冻结的模型实例，必填。
    :param mode: 微调算法类型，必填。可选填'lora'或'prefixtuning'。
    :param include: 需要冻结的模块名列表， 选填。
                    模糊匹配列表中所有模块名，挨个将匹配到的模块的requires_grad设置为False。
                    列表项支持配置符号*，代表任意字符串，格式如 ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']
    :param exclude: 不冻结的模块名列表， 选填。
                    模糊匹配列表中所有模块名，挨个将匹配到的模块的requires_grad设置为True。
                    列表项支持配置符号*，代表任意字符串，格式如 ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']
                    当include和exclude列表项冲突时，对该项匹配到的模块不做任何处理。
    """
    logger.info("Start to freeze model for delta, mode: %s, include list: %s, exclude list: %s",
                mode, include, exclude)
    if not isinstance(model, nn.Cell):
        raise TypeError("A Cell is required for argument 'model'.")

    if isinstance(mode, str) and mode.lower() in DELTA_LIST:
        _freeze_for_mode(model, mode)
    else:
        raise ValueError(f"An argument of string type from {DELTA_LIST} is required.")

    if include or exclude:
        try:
            freeze_modules(model, include, exclude)
        except Exception as ex:
            raise UnexpectedError(f"Exception occurred when freeze model for delta, error message: {str(ex)}") from ex

    logger.info("End to freeze model for delta.")


def freeze_from_config(model: nn.Cell,
                       config_path: str) -> None:
    """
    根据配置文件中freeze关键词下的include和exclude列表冻结网络指定模块。

    :param model: 需要冻结的模型实例，必填。
    :param config_path: 配置文件路径，必填。

    配置文件内容实例：
    freeze:
      include: ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']
      exclude: ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']

    include: 需要冻结的模块名列表， 选填。
            模糊匹配列表中所有模块名，挨个将匹配到的模块的requires_grad设置为False。
            列表项支持配置符号*，代表任意字符串，格式如 ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']
    exclude: 不冻结的模块名列表， 选填。
            模糊匹配列表中所有模块名，挨个将匹配到的模块的requires_grad设置为True。
            列表项支持配置符号*，代表任意字符串，格式如 ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']
            当include和exclude列表项冲突时，对该项匹配到的模块不做任何处理。
    """
    logger.info("Start to freeze model from config file.")
    if not isinstance(model, nn.Cell):
        raise TypeError("A Cell is required for argument 'model'.")

    if not isinstance(config_path, str):
        raise TypeError("A string is required for argument 'config_path'.")

    if not config_path:
        raise ValueError("Argument 'config_path' is empty.")

    try:
        content_dict = read_yaml_file(config_path)
    except (FileNotFoundError, LinkPathError) as ex:
        raise ReadYamlFileError(f'Error occurred when freeze model from config, error message: {str(ex)}') from ex
    except Exception as ex:
        raise UnexpectedError(f'Unexpected error occurred when freeze model from config, '
                              f'error message: {str(ex)}') from ex

    include, exclude = _get_freeze_modules_list(content_dict)
    freeze_modules(model, include, exclude)
    logger.info("End to freeze model from config file.")


def _freeze_for_mode(model: nn.Cell, mode: str) -> None:
    """
    根据微调算法冻结网络。

    :param model: 需要冻结的模型实例，必填。
    """
    delta_name = '*mindpet_delta_' + mode + '*'
    if mode == 'bitfit':
        delta_name = '*bias'
    freeze_modules(model, include=['*'], exclude=[delta_name])


def _get_freeze_modules_list(content_dict) -> tuple:
    """
    从配置文件中获取include和exclude列表。

    :param content_dict: 配置文件内容字典
    """
    if not content_dict or 'freeze' not in content_dict.keys():
        raise ModelConfigFreezeInfoError("Missing required key 'freeze' in config file.")

    freeze_dict = content_dict.get('freeze')
    if not isinstance(freeze_dict, dict):
        raise ModelConfigFreezeInfoError("A dict is required for 'freeze' value in config file.")

    if freeze_dict is None:
        raise ModelConfigFreezeInfoError("'freeze' value in config file is empty.")

    # 校验freeze下是否有include和exclude字典，以及是否有其它非法字典
    for key in freeze_dict.keys():
        if key not in ['include', 'exclude']:
            raise ModelConfigFreezeInfoError("Invalid key in 'freeze' value, just config 'include'/'exclude' key.")

    include_list = []
    exclude_list = []
    if 'include' in freeze_dict.keys():
        include_list = freeze_dict.get('include')

    if 'exclude' in freeze_dict.keys():
        exclude_list = freeze_dict.get('exclude')

    return include_list, exclude_list


def _match_str_and_list(m_str, m_list: Optional[List[str]] = None) -> bool:
    """
    校验字符串是否与列表中某一项匹配。
    :param m_str: 完整字符串
    :param m_list: 有关键词的列表
    """
    if m_list is None:
        return False

    for key in m_list:
        if not isinstance(key, str):
            raise TypeError(f"List item '{key}' is not a string.")

        if fnmatch(m_str, key):
            return True

    return False
