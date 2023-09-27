#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""Utils Constants"""
import os
import stat


# 空字符串
EMPTY_STRING = ''

# 入参本地路径黑名单字符过滤
BLACKLIST_CHARACTERS = (
    '\'', '"',
    '..', '../', '%2e%2e', '%2e./', '.%2e/', '..%2f', '%2e%2e/', '%2e.%2f', '.%2e%2f', '%2e%2e%2f'
)

# 入参选项字符粒度参数校验
SPACE_CHARACTER = ' '
UPPER_CASE_LETTERS = [chr(i) for i in range(65, 91)]
LOWER_CASE_LETTERS = [chr(i) for i in range(97, 123)]
NUMBERS = [chr(i) for i in range(48, 58)]
BASE_WHITELIST_CHARACTERS = {
    'UPPER': UPPER_CASE_LETTERS,
    'LOWER': LOWER_CASE_LETTERS,
    'LETTERS': UPPER_CASE_LETTERS + LOWER_CASE_LETTERS,
    'NUMBERS': NUMBERS,
    'ALL': UPPER_CASE_LETTERS + LOWER_CASE_LETTERS + NUMBERS
}

# 入参长度默认限制
DEFAULT_PATH_LEN_MIN_LIMIT = 1
DEFAULT_PATH_LEN_MAX_LIMIT = 4096
DEFAULT_FILE_LEN_MIN_LIMIT = 1
DEFAULT_FILE_LEN_MAX_LIMIT = 255

# 文件大小限制
MB_SIZE = 1024 * 1024
GB_SIZE = 1024 * 1024 * 1024
MAX_FILE_MB_SIZE = 10

# 文件I/O默认权限配置
DEFAULT_FLAGS = os.O_RDWR | os.O_CREAT  # 允许读写, 文件不存在时新建
DEFAULT_MODES = stat.S_IWUSR | stat.S_IRUSR  # 所有者读写

# finetune model_config配置文件相关配置
MODEL_CONFIG_ROOT_KEYS = ('params', 'freeze')
MODEL_CONFIG_PARAMS_CHARACTER_BLACKLIST = ('|', '&', ',', ';', '`', '"', '\'', '$', '(', ')', '{', '}', '>', '<', '?',
                                           '\\', '!', '\n', '*', '#')
MODEL_CONFIG_LEN_LIMIT = pow(2, 14)

# 微调基础包入口标识
ENTRANCE_TYPE = 'ENTRANCE_TYPE'

# 路径权限最小范围
MIN_MODE = 0
MAX_MODE = 7
PATH_MODE_LIMIT = '750'
DEFAULT_MAX_LOG_FILE_SIZE = 10
DEFAULT_MAX_LOG_FILE_NUM = 10

# output_path内部随机命名文件夹权限控制
OUTPUT_PATH_RANDOM_DIR_MODE = 0o750

# 评估结果文件名
EVAL_RESULT_FILE_NAME = 'eval_result.json'

# 推理结果文件名
INFER_RESULT_FILE_NAME = 'infer_result.json'

# 超时机制相关超参数
TIME_MONITOR_SLEEP_TIME = 1
TIMEOUT_REGEX = '^([0-9]+d)?([0-9]+h)?$'
SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24

# 命令行模式异常退出状态码
ABNORMAL_EXIT_CODE = 1

# MindPet定义入参
MINDPET_DEFINED_PARAM_NAME = (
    'dp', 'data_path', 'op', 'output_path', 'bp', 'boot_file_path', 'mc', 'model_config_path', 'pm',
    'pretrained_model_path', 'cp', 'ckpt_path', 'q', 'quiet', 't', 'timeout', 'advanced_config')

# finetune接口入参顺序约束
ARG_NAMES = {
    'finetune': (
        'data_path', 'output_path', 'boot_file_path', 'pretrained_model_path', 'model_config_path', 'timeout'),
    'evaluate': ('data_path', 'ckpt_path', 'output_path', 'boot_file_path', 'model_config_path', 'timeout'),
    'infer': ('data_path', 'ckpt_path', 'output_path', 'boot_file_path', 'model_config_path', 'timeout')
}

# 自定义超参数不合法前缀
INVALID_CUSTOM_PARAM_KEY_PREFIX = '-'
INVALID_CUSTOM_PARAM_VAL_PREFIX = '--'

# 评估推理任务相关常量
FINETUNE_TASK_NAME = 'finetune'
EVALUATE_TASK_NAME = 'evaluate'
INFER_TASK_NAME = 'infer'
EVAL_INFER_TASK_NAMES = [EVALUATE_TASK_NAME, INFER_TASK_NAME]

# 微调工具包SDK接口清单
MINDPET_SDK_INTERFACE_NAMES = [FINETUNE_TASK_NAME, EVALUATE_TASK_NAME, INFER_TASK_NAME]

# 微调算法清单
DELTA_LIST = ['lora', 'prefixtuning', 'adapter', 'low_rank_adapter', 'bitfit', "ptuning2"]
