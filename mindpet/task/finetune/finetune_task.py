#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved."""

import os
import subprocess
from mindpet.utils.io_utils import read_yaml_file
from mindpet.utils.constants import MODEL_CONFIG_ROOT_KEYS
from mindpet.utils.exceptions import LinkPathError, TaskError, ReadYamlFileError, UnexpectedError, CreateProcessError, \
    MonitorProcessRspError
from mindpet.log.log import record_operation_and_service_error_log, record_operation_and_service_info_log, \
    operation_logger_without_std
from mindpet.utils.task_utils import create_output_path_subdir_with_uuid, model_config_params_check_item, \
    extend_model_config_freeze_command, monitor_process_rsp_code, model_config_keys_check_item, path_in_out_loop_check


class FinetuneTask:
    """FinetuneTask class"""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.command = []
        self.command_params = []

        try:
            self._process_param_and_command()
        except KeyboardInterrupt as ex:
            record_operation_and_service_error_log(
                'Finetune task is terminated by current user, task has stopped and exited.')
            raise ex
        except Exception as ex:
            record_operation_and_service_error_log('Finetune failed.')
            raise ex

    # pylint: disable=R1732
    def start(self):
        """
        启动命令
        :return: 微调执行状态
        """
        record_operation_and_service_info_log('Finetune task is running.')

        try:
            process = subprocess.Popen(self.command, env=os.environ, shell=False,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except KeyboardInterrupt as ex:
            record_operation_and_service_error_log('Finetune task is terminated by current user, '
                                                   'task has stopped and exited.')
            raise ex
        except Exception as ex:
            if ex is None or not str(ex):
                raise CreateProcessError('Exception occurred when creating finetune task process, '
                                         'no error message available.') from ex
            raise CreateProcessError(f'Exception occurred when creating finetune task process, '
                                     f'error message: {str(ex)}.') from ex

        try:
            rsp_code = monitor_process_rsp_code(task_name='finetune', task_process=process,
                                                timeout=self.kwargs.get('timeout'))
        except KeyboardInterrupt as ex:
            record_operation_and_service_error_log('Finetune task is terminated by current user, '
                                                   'task has stopped and exited.')
            raise ex
        except Exception as ex:
            if ex is None or not str(ex):
                raise MonitorProcessRspError('Exception occurred when monitoring finetune task process, '
                                             'no error message available.') from ex
            raise MonitorProcessRspError('Exception occurred when monitoring finetune task process, '
                                         'error message: {str(ex)}.') from ex

        if rsp_code != 0:
            operation_logger_without_std.error('Finetune failed.')
            raise TaskError('Finetune failed.')

        record_operation_and_service_info_log('Finetune successfully.')
        return True

    def _process_param_and_command(self):
        """
        调优任务启动前的准备工作
        """
        # 命令行参数准备
        self._prepare_command_params()

        # 命令准备
        self._prepare_command()

    def _prepare_command_params(self):
        """
        准备命令行参数
        """
        # 输入、输出回环路径判断
        path_in_out_loop_check(self.kwargs)

        self._wrap_optional_params()
        self._wrap_model_config_params()
        self._wrap_required_params()

    def _wrap_optional_params(self):
        """
        组装选填参数
        """
        pretrained_model_path = self.kwargs.get('pretrained_model_path')
        if pretrained_model_path is not None:
            self.command_params.extend(['--pretrained_model_path', str(pretrained_model_path)])

    def _wrap_required_params(self):
        """
        组装必填参数
        """
        output_path = create_output_path_subdir_with_uuid(self.kwargs.get('output_path'))

        self.command_params.extend(['--output_path', output_path])
        self.command_params.extend(['--data_path', self.kwargs.get('data_path')])

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
                    f'Error occurred when getting finetune params info from param [model_config_path], '
                    f'error message: {str(ex)}') from ex
            except Exception as ex:
                raise UnexpectedError(
                    f'Unexpected error occurred when getting finetune params info from param [model_config_path], '
                    f'error message: {str(ex)}') from ex

            model_config_keys_check_item(content, MODEL_CONFIG_ROOT_KEYS)
            model_config_params_check_item(task_object=self, content=content)
            extend_model_config_freeze_command(content=content, task_object=self,
                                               model_config_path=self.kwargs.get('model_config_path'))

    def _prepare_command(self):
        """
        组装命令
        """
        self.command.append('python')
        self.command.append(self.kwargs.get('boot_file_path'))
        self.command.extend(self.command_params)
