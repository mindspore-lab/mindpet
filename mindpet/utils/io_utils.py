#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""IO APIs"""
import os
import json
import yaml
from mindpet.utils.exceptions import LinkPathError
from mindpet.utils.constants import DEFAULT_FLAGS, DEFAULT_MODES, EMPTY_STRING


def is_link_path(path):
    """
    判断路径是否是软链接
    :return: 是否是软链接
    """
    path = os.path.abspath(str(path))
    return os.path.islink(path)


def read_yaml_file(path, flags=None, modes=None):
    """
    读取文件内容

    :param path: 待读取文件路径
    :param flags 文件I/O flag标识
    :param modes 文件I/O modes标识
    :return: 读取内容
    """
    if flags is None:
        flags = DEFAULT_FLAGS

    if modes is None:
        modes = DEFAULT_MODES

    if not os.path.exists(path):
        raise FileNotFoundError('File is not found.')

    if is_link_path(path):
        raise LinkPathError('Detect link path, refuse reading file.')

    with os.fdopen(os.open(path, flags, modes), 'rb') as file:
        content = yaml.safe_load(file)

    return content


def read_json_file(path, flags=None, modes=None):
    """
    读取json文件内容

    :param path: 待读取文件路径
    :param flags 文件I/O flag标识
    :param modes 文件I/O modes标识
    :return: 读取内容
    """
    if flags is None:
        flags = DEFAULT_FLAGS

    if modes is None:
        modes = DEFAULT_MODES

    if not os.path.exists(path):
        raise FileNotFoundError('File is not found.')

    if is_link_path(path):
        raise LinkPathError('Detect link path, refuse reading file.')

    if os.path.getsize(path) == 0:
        return EMPTY_STRING

    with os.fdopen(os.open(path, flags, modes), 'rb') as file:
        content = json.load(file)

    return content
