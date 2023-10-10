#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""Model Config check module."""

from mindpet.log.log import logger
from mindpet.utils.exceptions import ModelConfigParamsInfoError
from mindpet.utils.constants import MODEL_CONFIG_PARAMS_CHARACTER_BLACKLIST, MODEL_CONFIG_LEN_LIMIT, \
    MINDPET_DEFINED_PARAM_NAME, INVALID_CUSTOM_PARAM_KEY_PREFIX, INVALID_CUSTOM_PARAM_VAL_PREFIX


class ModelConfigParamsChecker:
    """
    Define model config params checker.
    """
    def __init__(self, task_object, params_config=None):
        """
        model_config配置文件内容校验构造方法
        :param task_object: 任务对象
        :param params_config: params部分参数
        """
        if task_object is None:
            raise ValueError('Param [task_object] is required.')

        if not params_config:
            raise ValueError('Param [params_config] is required.')

        if params_config is not None and not isinstance(params_config, dict):
            raise TypeError('Invalid type for param [params_config], which must be key-value pairs.')

        self.task_object = task_object
        self.params = params_config

    def check(self):
        """
        model_config配置文件params部分参数校验
        """
        for param in self.params.items():
            param_key, param_val = param[0], param[1]

            if param_key is None or param_val is None:
                raise ModelConfigParamsInfoError('Config in [params] part in model config file is empty.')

            # params的value部分是否为单层键值对
            if isinstance(param_val, dict):
                raise ModelConfigParamsInfoError(
                    'Config in [params] part in model config file only supports single-level nesting.')

            param_key, param_val = str(param_key), str(param_val)

            # 接口预定义参数与params的key重复校验
            if param_key in MINDPET_DEFINED_PARAM_NAME:
                # pylint: disable=W1203
                logger.warning(
                    f'Find duplicate key [{param_key}] from config in [params] part in model config file.')
                continue

            # params键值对长度及字符合法性
            if len(param_key) > MODEL_CONFIG_LEN_LIMIT or len(param_val) > MODEL_CONFIG_LEN_LIMIT:
                raise ModelConfigParamsInfoError('Config in [params] part in model config file is too long.')

            for k_item in param_key:
                if k_item in MODEL_CONFIG_PARAMS_CHARACTER_BLACKLIST:
                    raise ModelConfigParamsInfoError('Config in [params] part in model config file '
                                                     'contains invalid character(s).')

            for v_item in param_val:
                if v_item in MODEL_CONFIG_PARAMS_CHARACTER_BLACKLIST:
                    raise ModelConfigParamsInfoError('Config in [params] part in model config file '
                                                     'contains invalid character(s).')

            # params键值对不合法前缀校验
            if param_key.startswith(INVALID_CUSTOM_PARAM_KEY_PREFIX) or param_val.startswith(
                    INVALID_CUSTOM_PARAM_VAL_PREFIX):
                raise ModelConfigParamsInfoError('Invalid character(s) in [params] part in model config file.')

            self.task_object.command_params.extend([f'--{param_key}', param_val])
