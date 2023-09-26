#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""Task APIs"""
import os
import signal
import time
import uuid
import multiprocessing
from pathlib import Path
import click
from mindpet.security.param_check.model_config_params_check_util import ModelConfigParamsChecker
from mindpet.utils.exceptions import MakeDirError, PathLoopError, ModelConfigKeysInfoError, ModelConfigParamsInfoError
from mindpet.log.log import logger, logger_without_std, record_operation_and_service_info_log
from mindpet.utils.constants import OUTPUT_PATH_RANDOM_DIR_MODE, TIME_MONITOR_SLEEP_TIME, SECONDS_PER_HOUR, HOURS_PER_DAY


def monitor_process_rsp_code(task_name, task_process, timeout=None):
    """
    含超时机制的子进程监测器, 并获取子进程结束状态码
    :param task_name: 任务名称
    :param task_process: 待监听进程
    :param timeout: 超时时间(秒数)
    """
    timeout_monitor_process = None

    # 启动异步进程做超时监听
    if timeout is not None:
        timeout_monitor_process = multiprocessing.Process(target=timeout_monitor,
                                                          args=(task_name, task_process, timeout))
        timeout_monitor_process.start()

    # 持续监听子进程标准输出管道, 捕捉管道内容并落盘
    while task_process.poll() is None:
        process_log = str(task_process.stdout.readline().decode('utf-8')).rstrip()
        if process_log:
            logger.info(process_log, extra={'Model': True})

    # 子进程正常退出时, 如超时监听进程未结束, 结束超时监听
    if timeout is not None and timeout_monitor_process is not None and timeout_monitor_process.is_alive():
        timeout_monitor_process.terminate()

    return task_process.poll()


def timeout_monitor(task_name, task_process, timeout):
    """
    超时监听, 超时后终止被监听进程
    :param task_name: 任务名称
    :param task_process: 被监听进程
    :param timeout: 超时时间(秒数)
    """
    begin_time = time.time()
    while True:
        if time.time() - begin_time > timeout + 1:
            days = int(timeout // SECONDS_PER_HOUR // HOURS_PER_DAY)
            hours = int((timeout // SECONDS_PER_HOUR) % HOURS_PER_DAY)

            logger.warning('Task exceeds the time limit, it is forcibly terminated [%ddays %dhours].', days, hours)
            record_operation_and_service_info_log('%s failed because of time limit.', str(task_name).capitalize())
            task_process.kill()
            task_process.terminate()

            # 避免子进程无法全部结束, 向同进程组进程发送结束信号
            try:
                os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
            except (TypeError, ProcessLookupError):
                logger_without_std.warning('Task exceeds the time limit, it is forcibly terminated [%ddays %dhours].',
                                           days, hours)
                break

        time.sleep(TIME_MONITOR_SLEEP_TIME)


def create_output_path_subdir_with_uuid(output_path):
    """
    在output_path下生成包含uuid随机片段的子文件夹名称
    :param output_path: 入参output_path值
    """
    uuid_name = str(uuid.uuid4()).replace('-', '')
    random_dir = f'mindpet_{uuid_name}'.upper()
    full_output_path = os.path.join(output_path, random_dir).replace('\\', '/')

    while os.path.exists(full_output_path):
        uuid_name = str(uuid.uuid4()).replace('-', '')
        random_dir = f'mindpet_{uuid_name}'.upper()
        full_output_path = os.path.join(output_path, random_dir).replace('\\', '/')

    try:
        os.makedirs(full_output_path, exist_ok=True, mode=OUTPUT_PATH_RANDOM_DIR_MODE)
    except Exception as ex:
        raise MakeDirError(f'Failed to create directory from param [output_path], error message: {str(ex)}') from ex

    # pylint: disable=W1203
    logger.info(f'Create output directory successfully, directory name: {random_dir}')
    return full_output_path


def model_config_keys_check_item(content, config_keys):
    """
    校验model_config.yaml文件中的配置头名称是否合法
    :param content: yaml文件中配置内容
    :param config_keys: 允许配置的头名称
    """
    if not isinstance(config_keys, tuple):
        config_keys = (config_keys,)

    try:
        for key_item in content.keys():
            if key_item not in config_keys:
                raise ModelConfigKeysInfoError("Invalid config in model config file, "
                                               f"only support [{'/'.join(config_keys)}]")
    except AttributeError as ex:
        raise ModelConfigKeysInfoError('Invalid key in model config file.') from ex


def model_config_params_check_item(task_object, content):
    """
    model_config自定义参数内容黑名单校验及参数拼装
    :param task_object: 任务对象
    :param content: 待校验内容
    """
    if 'params' in content.keys():
        params_config = content.get('params')

        if params_config is None:
            raise ModelConfigParamsInfoError('[params] attribute in model config file is empty.')

        params_check = ModelConfigParamsChecker(task_object=task_object, params_config=params_config)
        params_check.check()


def extend_model_config_freeze_command(task_object, model_config_path, content):
    """
    如果有freeze关键词，则拼接--advanced_config参数传递文件路径
    :param content: 待校验内容
    :param task_object: 参数对象
    :param model_config_path: model_config配置文件路径
    """
    if 'freeze' in content.keys():
        task_object.command_params.extend(['--advanced_config', model_config_path])


def path_in_out_loop_check(param_dict):
    """
    用于判断输入、输出路径是否存在回环可能
    :param param_dict: 输入参数集合
    """
    # 从参数集合中提取输出路径
    output_path = Path(param_dict.get('output_path'))

    # 从参数集合中提取输入路径
    input_path_dict = {'data_path': Path(param_dict.get('data_path'))}
    if param_dict.get('pretrained_model_path') is not None:
        input_path_dict['pretrained_model_path'] = Path(param_dict.get('pretrained_model_path'))
    if param_dict.get('ckpt_path') is not None:
        input_path_dict['ckpt_path'] = Path(param_dict.get('ckpt_path'))

    # 判断输入路径/输出径是否相等或输出路径是否是输入路径的子路径, 如果是, 则存在回环路径的可能
    for input_path_name, input_path_value in input_path_dict.items():
        if (output_path == input_path_value) or (input_path_value in output_path.parents):
            raise PathLoopError(f'Param [output_path] can not be the same as, '
                                f'or a subdirectory of param [{input_path_name}]')


def handle_exception_log(exception):
    """
    根据异常场景, 给出对应异常日志
    :param exception: 异常场景
    """
    if isinstance(exception, click.exceptions.Abort):
        logger.info('Current command is artificially canceled.')
    elif isinstance(exception, click.exceptions.NoSuchOption):
        # pylint: disable=W1203
        logger.error(f'Invalid param detected, error message: {str(exception)}')
    elif isinstance(exception, click.exceptions.MissingParameter):
        # pylint: disable=W1203
        logger.error(f'Necessary param is missing, error message: {str(exception)}')
    elif isinstance(exception, Exception):
        if exception is None or not str(exception):
            logger.error('Exception occurred, no error message available.')
        elif str(exception).isdigit():
            # pylint: disable=W1203
            logger.error(f'Exception occurred, error code: {str(exception)}')
        # pylint: disable=W1203
        logger.error(f'Exception occurred, error message: {str(exception)}')
