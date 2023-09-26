#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved."""

import os
import json
import subprocess
from mindpet.utils.io_utils import read_yaml_file, read_json_file
from mindpet.security.param_check.option_check_utils import PathContentCheckParam
from mindpet.task.evaluate_infer.result_file_check import ResultFileCheckParam, ResultFileCheck
from mindpet.log.log import logger, operation_logger_without_std, record_operation_and_service_info_log, \
    record_operation_and_service_error_log, record_operation_and_service_warning_log
from mindpet.utils.constants import MODEL_CONFIG_ROOT_KEYS, EVAL_RESULT_FILE_NAME, INFER_RESULT_FILE_NAME, SPACE_CHARACTER, \
    PATH_MODE_LIMIT, EVAL_INFER_TASK_NAMES, EVALUATE_TASK_NAME, EMPTY_STRING
from mindpet.utils.task_utils import create_output_path_subdir_with_uuid, model_config_params_check_item, \
    monitor_process_rsp_code, model_config_keys_check_item, path_in_out_loop_check
from mindpet.utils.exceptions import LinkPathError, ReadYamlFileError, UnexpectedError, MonitorProcessRspError, \
    CreateProcessError, TaskError, FileOversizeError, PathRightEscalationError


class EvaluateInferTask:
    """EvaluateInferTask class"""
    def __init__(self, task_type, *args, **kwargs):
        """
        评估/推理任务构造方法
        task_type: 任务类型(evaluate/infer)
        *args: 元组参数
        **kwargs: 字典参数
        """
        if task_type not in EVAL_INFER_TASK_NAMES:
            raise ValueError(f'Invalid task_type, only support [{EVAL_INFER_TASK_NAMES}].')

        self.task_type = task_type
        self.result_name = EVAL_RESULT_FILE_NAME if self.task_type == EVALUATE_TASK_NAME else INFER_RESULT_FILE_NAME
        self.args = args
        self.kwargs = kwargs
        self.command = []
        self.command_params = []

        try:
            self._process_param_and_command()
        except KeyboardInterrupt as ex:
            record_operation_and_service_error_log(
                f'{str(self.task_type).capitalize()} task is terminated by current user, task has stopped and exited.')
            raise ex
        except Exception as ex:
            record_operation_and_service_error_log(f'{str(self.task_type).capitalize()} failed.')
            raise ex

    # pylint: disable=R1732
    def start(self):
        """
        启动命令
        :return: 评估/推理JSON形式结果
        """
        # 启动评估/推理任务
        record_operation_and_service_info_log(
            f'{str(self.task_type).capitalize()} task is running.')

        try:
            process = subprocess.Popen(self.command, env=os.environ, shell=False,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except KeyboardInterrupt as ex:
            record_operation_and_service_error_log(
                f'{str(self.task_type).capitalize()} task is terminated by current user, task has stopped and exited.')
            raise ex

        except Exception as ex:
            if ex is None or not str(ex):
                raise CreateProcessError(f'Exception occurred when creating {self.task_type} task process, '
                                         f'no error message available.') from ex
            raise CreateProcessError(f'Exception occurred when creating {self.task_type} task process, '
                                     f'error message: {str(ex)}.') from ex

        # 启动超时监听程序
        try:
            rsp_code = monitor_process_rsp_code(task_name=self.task_type, task_process=process,
                                                timeout=self.kwargs.get('timeout'))
        except KeyboardInterrupt as ex:
            record_operation_and_service_error_log(
                f'{str(self.task_type).capitalize()} task is terminated by current user, task has stopped and exited.')
            raise ex
        except Exception as ex:
            if ex is None or not str(ex):
                raise MonitorProcessRspError(f'Exception occurred when monitoring {self.task_type} task process, '
                                             f'no error message available.') from ex
            raise MonitorProcessRspError(f'Exception occurred when monitoring {self.task_type} task process, '
                                         f'error message: {str(ex)}.') from ex

        if rsp_code != 0:
            message = f'{str(self.task_type).capitalize()} failed.'
            operation_logger_without_std.error(message)
            raise TaskError(message)

        # 获取评估结果
        result = self._get_task_result()

        if result.get('status') == 0:
            record_operation_and_service_info_log(f'{str(self.task_type).capitalize()} successfully.')
        else:
            record_operation_and_service_warning_log(
                f'Completed {self.task_type} task, but failed to get {self.result_name} file.')
            logger.warning(result.get("error_message"))

        return result

    # pylint: disable=W0718
    def _get_task_result(self):
        """
        读取并返回用户落盘的评估/推理结果
        :return: json格式的评估/推理结果
        """

        # 读取用户落盘结果，默认状态为失败
        result = {"status": -1,
                  "task_result": EMPTY_STRING}

        # 校验用户代码生成的评估/推理结果文件合法性
        full_output_path = self.command_params[self.command_params.index('--output_path') + 1]
        result_path = os.path.join(full_output_path, self.result_name)
        error_message = self._get_check_result_file_error_msg(result_path)

        # 如果未通过结果文件生成校验，直接返回评估/推理结果
        if error_message:
            result["error_message"] = error_message
            return result

        # 读取并解析评估/推理结果, 读取成功以json串形式并返回，无error_message；读取失败返回空字符串，有相应error_message。
        try:
            task_result = read_json_file(result_path)
        except (json.JSONDecodeError, TypeError):
            # 文件为空但文件大小不为0、格式错误
            result["error_message"] = f'File {self.result_name} should follow JSON format.'
        # pylint: disable=W0703
        except Exception as ex:
            result["error_message"] = f'An error occurred during reading {self.result_name}: {str(ex)}'
        else:
            if task_result:
                result["status"] = 0
                result["task_result"] = task_result
            else:
                result["error_message"] = f'File {self.result_name} is empty.'

        return result

    # pylint: disable=W0718
    def _get_check_result_file_error_msg(self, result_path):
        """
        获取结果文件校验结果与对应error_message
        :result_path: 用户落盘结果文件路径
        :return:
           check_flag, 校验是否合格
           error_message：对应错误信息
        """
        error_message = EMPTY_STRING
        try:
            self._check_task_result(result_path)
        except FileNotFoundError:
            error_message = f'Cannot find {self.result_name} in [output_path].'
        except PathRightEscalationError:
            error_message = f'Permission denied when reading {self.result_name}.'
        except FileOversizeError:
            error_message = f'File {self.result_name} is too large.'
        except LinkPathError:
            error_message = f'Detect link path, reject reading file: {self.result_name}.'
        except ValueError:
            error_message = f'Invalid file: {self.result_name}.'
        # pylint: disable=W0703
        except Exception as ex:
            error_message = f'An error occurred during reading {self.result_name}: {str(ex)}'

        return error_message

    def _process_param_and_command(self):
        """
        评估/推理任务启动前的准备工作
        """
        # 命令行参数准备
        self._prepare_command_params()

        # 命令准备
        self._prepare_command()

    def _check_task_result(self, eval_result_path):
        """
        校验评估/推理结果文件合法性
        """
        base_whitelist_mode = 'ALL'
        extra_whitelist = ['.', '/', '-', '_', SPACE_CHARACTER]

        result_file_name = EVAL_RESULT_FILE_NAME if self.task_type == EVALUATE_TASK_NAME else INFER_RESULT_FILE_NAME
        result_file_path = ResultFileCheck(option_name=result_file_name, option_value=eval_result_path)

        path_content_check_param = PathContentCheckParam(base_whitelist_mode=base_whitelist_mode,
                                                         extra_whitelist=extra_whitelist)

        check_param = ResultFileCheckParam(path_content_check_param=path_content_check_param,
                                           mode=PATH_MODE_LIMIT,
                                           path_including_file=True,
                                           force_quit=False,
                                           quiet=self.kwargs.get('quiet'))

        result_file_path.check(check_param)

    def _prepare_command_params(self):
        """
        准备命令行参数
        """
        # 输入、输出回环路径判断
        path_in_out_loop_check(self.kwargs)

        self._wrap_model_config_params()
        self._wrap_required_params()

    def _wrap_required_params(self):
        """
        组装必填参数
        """
        output_path = create_output_path_subdir_with_uuid(self.kwargs.get('output_path'))

        self.command_params.extend(['--output_path', output_path])
        self.command_params.extend(['--data_path', self.kwargs.get('data_path')])
        self.command_params.extend(['--ckpt_path', self.kwargs.get('ckpt_path')])

    def _wrap_model_config_params(self):
        """
        针对model_config配置文件内容，组装相关参数
        """
        model_config_option = self.kwargs.get('model_config_path')

        if model_config_option is not None:
            try:
                content = read_yaml_file(model_config_option)
            except (FileNotFoundError, LinkPathError) as ex:
                raise ReadYamlFileError(
                    f'Error occurred when getting {self.task_type} params info from param [model_config_path], '
                    f'error message: {str(ex)}') from ex
            except Exception as ex:
                raise UnexpectedError(
                    f'Unexpected error occurred when getting {self.task_type} params info from '
                    f'param [model_config_path], error message: {str(ex)}') from ex

            model_config_keys_check_item(content, MODEL_CONFIG_ROOT_KEYS[0])
            model_config_params_check_item(task_object=self, content=content)

    def _prepare_command(self):
        """
        组装命令
        """
        self.command.append('python')
        self.command.append(self.kwargs.get('boot_file_path'))
        self.command.extend(self.command_params)
