#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""Version APIs"""
def is_version_le(current_version, base_version):
    """
    description: Check whether the current version is lower than or equal to the base version.
    args:
        current_version: string, like "1.10.1"
        base_version: string, like "2.0.0rc1"
    return:
        Bool: current_version <= base_version
    example:
        for current_version: 1.10.1, base_version: 2.0.0rc1, it return True.
    """
    version_split_char = '.'
    if version_split_char not in base_version or version_split_char not in current_version:
        raise ValueError("The version string will contain the `.`."
                         "For example, current_version 1.8.1, base_version: 2.0.0.")
    for x, y in zip(current_version.split(version_split_char), base_version.split(version_split_char)):
        if not x.isdigit() or not y.isdigit():
            continue
        if int(x) != int(y):
            return int(x) <= int(y)
    return True


def is_version_ge(current_version, base_version):
    """
    description: Check whether the current version is higher than or equal to the base version.
    args:
        current_version: string, like "1.10.1"
        base_version: string, like "2.0.0rc1"
    return:
        Bool: current_version >= base_version
    example:
        for current_version: 1.10.1, base_version: 2.0.0rc1, it return False.
    """
    version_split_char = '.'
    if version_split_char not in base_version or version_split_char not in current_version:
        raise ValueError("The version string will contain the `.`."
                         "For example, current_version 1.8.1， base_version: 2.0.0.")
    for x, y in zip(current_version.split(version_split_char), base_version.split(version_split_char)):
        if not x.isdigit() or not y.isdigit():
            continue
        if int(x) != int(y):
            return int(x) >= int(y)
    return True
