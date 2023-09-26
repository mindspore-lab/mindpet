#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved."""

from mindpet.task.options import DataPathOption, PretrainedModelPathOption, OutputPathOption, BootFilePathOption, \
    ModelConfigPathOption, QuietOption, CkptPathOption, TimeoutOption


def finetune_options():
    """
    微调功能入参集合
    :return: 装饰器
    """
    options = [
        QuietOption(),
        DataPathOption(),
        PretrainedModelPathOption(),
        OutputPathOption(),
        BootFilePathOption(),
        ModelConfigPathOption(),
        TimeoutOption()
    ]

    def decorator(func):
        if not hasattr(func, '__click_params__'):
            func.__click_params__ = []
        func.__click_params__ += options
        return func

    return decorator


def evaluate_options():
    """
    评估功能入参集合
    :return: 装饰器
    """
    options = [
        QuietOption(),
        DataPathOption(),
        CkptPathOption(),
        OutputPathOption(),
        BootFilePathOption(),
        ModelConfigPathOption(),
        TimeoutOption()
    ]

    def decorator(func):
        if not hasattr(func, '__click_params__'):
            func.__click_params__ = []
        func.__click_params__ += options
        return func

    return decorator


def infer_options():
    """
    推理功能入参集合
    :return: 装饰器
    """
    options = [
        QuietOption(),
        DataPathOption(),
        CkptPathOption(),
        OutputPathOption(),
        BootFilePathOption(),
        ModelConfigPathOption(),
        TimeoutOption()
    ]

    def decorator(func):
        if not hasattr(func, '__click_params__'):
            func.__click_params__ = []
        func.__click_params__ += options
        return func

    return decorator
