#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""MindPet Tuning Main."""

import sys
import click
from mindpet.utils.task_utils import handle_exception_log
from mindpet.utils.entrance_monitor import entrance_monitor
from mindpet.task.finetune.finetune_task import FinetuneTask
from mindpet.task.evaluate_infer.evaluate_infer_task import EvaluateInferTask
from mindpet.log.log import set_logger_property, record_operation_and_service_info_log
from mindpet.task.option_decorators import finetune_options, evaluate_options, infer_options
from mindpet.utils.constants import ENTRANCE_TYPE, EVALUATE_TASK_NAME, INFER_TASK_NAME, ABNORMAL_EXIT_CODE


def cli_wrapper():
    """
    CLI侧入口
    """
    entrance_monitor.set_value(ENTRANCE_TYPE, 'CLI')
    # 设置日志内容发起端标识
    set_logger_property("CLI")

    try:
        cli(standalone_mode=False)
    # pylint: disable=W0719
    # pylint: disable=W0703
    except Exception as ex:
        handle_exception_log(ex)
        sys.exit(ABNORMAL_EXIT_CODE)


@click.group()
def cli():
    pass


@cli.command()
@finetune_options()
def finetune(*args, **kwargs):
    """
    command line finetune entrance
    """
    record_operation_and_service_info_log('Start finetune.')
    finetune_task = FinetuneTask(*args, **kwargs)
    return finetune_task.start()


@cli.command()
@infer_options()
def infer(*args, **kwargs):
    """
    command line infer entrance
    """
    record_operation_and_service_info_log('Start infer.')
    infer_task = EvaluateInferTask(task_type=INFER_TASK_NAME, *args, **kwargs)
    return infer_task.start()


@cli.command()
@evaluate_options()
def evaluate(*args, **kwargs):
    """
    command line evaluate entrance
    """
    record_operation_and_service_info_log('Start evaluate.')
    evaluate_task = EvaluateInferTask(task_type=EVALUATE_TASK_NAME, *args, **kwargs)
    return evaluate_task.start()
