#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""MindPet SDK Module."""

from mindpet.mindpet_main import cli
from mindpet.log.log import logger, set_logger_property
from mindpet.utils.task_utils import handle_exception_log
from mindpet.utils.entrance_monitor import entrance_monitor
from mindpet.utils.constants import ENTRANCE_TYPE, EMPTY_STRING, ARG_NAMES, MINDPET_SDK_INTERFACE_NAMES

entrance_monitor.set_value(ENTRANCE_TYPE, 'SDK')
set_logger_property('SDK')


def finetune(*args, **kwargs):
    """
    SDK侧微调功能入口

    :param args: 参数args
    :param kwargs: 参数kwargs
    :return: 任务执行结果
    """
    return start_by_task_type(args, kwargs, task_type='finetune', ret_err_msg=False)


def evaluate(*args, **kwargs):
    """
    SDK侧评估功能入口

    :param args: 参数args
    :param kwargs: 参数kwargs
    :return: 任务执行结果
    """
    return start_by_task_type(args, kwargs, task_type='evaluate', ret_err_msg=EMPTY_STRING)


def infer(*args, **kwargs):
    """
    SDK侧推理功能入口

    :param args: 参数args
    :param kwargs: 参数kwargs
    :return: 任务执行结果
    """
    return start_by_task_type(args, kwargs, task_type='infer', ret_err_msg=EMPTY_STRING)


def start_by_task_type(args, kwargs, task_type, ret_err_msg):
    """
    根据任务类型启动任务

    :param task_type: 任务类型(finetune/evaluate/infer)
    :param ret_err_msg: 接口失败返回信息
    :param args: 参数args
    :param kwargs: 参数kwargs
    :return: 任务执行结果
    """
    if task_type not in MINDPET_SDK_INTERFACE_NAMES:
        logger.error('Invalid task_type for starting task.')
        return ret_err_msg

    try:
        commands = commands_generator(task_type, args, kwargs)
        return cli.main(commands, standalone_mode=False)
    # pylint: disable=W0719
    # pylint: disable=W0703
    except Exception as ex:
        handle_exception_log(ex)
        return ret_err_msg


def commands_generator(header, args, kwargs):
    """
    命令行生成器, 组装命令行参数

    :param header: 待执行功能
    :param args: 待组装args参数
    :param kwargs: 待组装kwargs参数
    :return: 组装后命令
    """
    output = []

    # 避免恶意传入过多参数导致下标越界
    args_length = min(len(args), len(ARG_NAMES.get(header)))

    for idx in range(args_length):
        output.append(f"--{(ARG_NAMES.get(header)[idx])}")
        output.append(f"--{(args[idx])}")

    for key_item, val_item in kwargs.items():
        # 安静模式仅允许CLI场景使用, SDK场景不允许配置该值, 给予错误警告
        if str(key_item) == 'quiet':
            raise ValueError('Param [quiet] is not supported by SDK.')
        # SDK场景参数值传None, 等价于CLI场景未传参数, 不应拼接到命令中
        if val_item is None:
            continue

        output.append(f"--{key_item}")
        output.append(f"--{val_item}")

    return [header] + output
