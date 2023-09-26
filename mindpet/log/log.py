#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""log module."""

import logging
import logging.config
import logging.handlers
import os
import sys
from typing import Dict, List, Tuple, Union
import traceback

try:
    from concurrent_log_handler import ConcurrentRotatingFileHandler as RFHandler
except ImportError:
    from logging.handlers import RotatingFileHandler as RFHandler

from mindpet.log.log_utils import check_list, const, convert_nodes_devices_input, create_dirs, \
    generate_rank_list, get_num_nodes_devices, get_rank_info, log_args_black_list_characters_replace, check_link_path
from mindpet.utils.constants import DEFAULT_MAX_LOG_FILE_NUM, DEFAULT_MAX_LOG_FILE_SIZE, ABNORMAL_EXIT_CODE
from mindpet.utils.exceptions import MakeDirError, UnsupportedPlatformError, PathOwnerError, PathModeError


logger_list = {}
stream_handler_list = {}
file_handler_list = {}
LOG_RECORD_MAX_LEN = 2048
LEVEL_CONVERTER = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
LOGGER_SOURCE_LIST = ('SDK', 'CLI')
UNKNOWN = 'unknown'
MODEL_LOG_FORMATTER = logging.Formatter(const.default_model_log_format)
MODE_640 = 0o640
MODE_440 = 0o440
MODE_750 = 0o750


def judge_stdout(rank_id: int,
                 rank_size: int,
                 is_output: bool,
                 nodes: Union[List, Tuple, Dict[str, int], None] = None,
                 devices: Union[List, Tuple, Dict[str, int], None] = None) -> bool:
    """判断日志内容是否需要输出到屏幕.

    Args:
        rank_id (int): 节点Rank id.
        rank_size (int): Rank size.
        is_output (bool): 是否需要输出到屏幕
        nodes (list or tuple or dict or None): 节点列表，列表中的节点的日志会输出到屏幕
        devices (list or tuple or dict or None): 设备列表，列表中的设备的日志会输出到屏幕

    Returns:
        is_output (bool): 最终判定是否需要输出到屏幕
    """
    nodes_or_devices = (nodes is not None or devices is not None)
    if is_output and rank_size > 1 and nodes_or_devices:
        num_nodes, num_devices = get_num_nodes_devices(rank_size)
        stdout_nodes = convert_nodes_devices_input(nodes, num_nodes)
        stdout_devices = convert_nodes_devices_input(devices, num_devices)
        check_list('nodes', stdout_nodes, num_nodes)
        check_list('devices', stdout_devices, num_devices)
        rank_list = generate_rank_list(stdout_nodes, stdout_devices)
        if rank_id not in rank_list:
            is_output = False

    return is_output


def validate_nodes_devices_input(var_name: str, var):
    """检查节点或者设备的参数输入

    Args:
        var_name (str): 参数名，节点或设备
        var: 参数

    Returns:
        None
    """
    if not (var is None or isinstance(var, (list, tuple, dict))):
        raise TypeError(f'The value of {var_name} can be None or a value of type tuple, list, or dict.')
    if isinstance(var, (list, tuple)):
        for item in var:
            if not isinstance(item, int):
                raise TypeError('The elements of a variable of type list or tuple must be of type int.')


def validate_level(var_name: str, var):
    """验证日志的等级是否正确

    Args:
        var_name (str): 参数名称
        var: 参数

    Returns:
        None
    """
    if not isinstance(var, str):
        raise TypeError(f'The format of {var_name} must be of type str.')
    if var not in const.level:
        raise ValueError(f'{var_name}={var} needs to be in {const.level}')


def validate_std_input_format(to_std: bool, stdout_nodes: Union[List, Tuple, None],
                              stdout_devices: Union[List, Tuple, None], stdout_level: str):
    """验证日志内容输出的节点、设备、等级"""

    if not isinstance(to_std, bool):
        raise TypeError('The format of the to_std must be of type bool.')

    validate_nodes_devices_input('stdout_nodes', stdout_nodes)
    validate_nodes_devices_input('stdout_devices', stdout_devices)
    validate_level('stdout_level', stdout_level)


def validate_file_input_format(file_level: Union[List, Tuple], file_save_dir: str, append_rank_dir: bool,
                               file_name: Union[List, Tuple]):
    """验证日志内容落盘的参数"""

    if not isinstance(file_level, (tuple, list)):
        raise TypeError('The value of file_level should be list or a tuple.')
    for level in file_level:
        validate_level('level in file_level', level)

    if not isinstance(file_name, (tuple, list)):
        raise TypeError('The value of file_name should be a list or a tuple.')

    if not len(file_level) == len(file_name):
        raise ValueError('The length of file_level and file_name should be equal.')

    if not isinstance(file_save_dir, str):
        raise TypeError('The type of file_save_dir should be a value of type str.')

    if not isinstance(append_rank_dir, bool):
        raise TypeError('The value of append_rank_dir should be a value of type bool.')

    for name in file_name:
        if not isinstance(name, str):
            raise TypeError('The value of name in file_name should be a value of type str.')
        if name.startswith('/'):
            raise ValueError('The file name cannot start with "/".')
        if name.startswith('../'):
            raise ValueError('The file name cannot start with "../".')


def _convert_level(level: str) -> int:
    """转换日志等级参数

    Args:
        level (str): 用户定义的日志等级

    Returns:
        level (str): 转换后的日志等级
    """
    logging_level = LEVEL_CONVERTER.get(level, logging.INFO)

    return logging_level


def get_stream_handler(stdout_format: str, stdout_level: str):
    """获得日志的流处理器"""
    if not stdout_format:
        stdout_format = const.default_stdout_format
    handler_name = f'{stdout_format}.{stdout_level}'
    if handler_name in stream_handler_list:
        return stream_handler_list.get(handler_name)

    stream_handler = CustomedStreamHandler(sys.stdout)
    stream_handler.setLevel(_convert_level(stdout_level))
    stream_formatter = logging.Formatter(stdout_format)
    stream_handler.setFormatter(stream_formatter)

    stream_handler_list[handler_name] = stream_handler

    return stream_handler


class CustomedStreamHandler(logging.StreamHandler):
    def format(self, record):
        return log_format(self, record)


def log_format(handler, record):
    if record.__dict__.get("Model"):
        fmt = MODEL_LOG_FORMATTER
    else:
        fmt = handler.formatter
    return fmt.format(record)


def get_file_path_list(base_save_dir: str,
                       append_rank_dir: bool,
                       server_id: int,
                       rank_id: int,
                       file_name: Union[Tuple, List],
                       mode: int = MODE_750) -> List:
    """获得日志落盘的日志文件路径"""
    if not base_save_dir:
        base_save_dir = os.path.expanduser(const.local_default_log_file_dir)

    file_save_dir = base_save_dir
    if append_rank_dir:
        rank_str = const.rank_dir_format.format(rank_id)
        node_str = const.node_dir_format.format(server_id)
        temp_save_dir = os.path.join(base_save_dir, node_str)
        file_save_dir = os.path.join(temp_save_dir, rank_str)

    file_path = []
    for name in file_name:
        path = os.path.join(file_save_dir, name)
        path = os.path.realpath(path)
        base_dir = os.path.dirname(path)
        create_dirs(base_dir)
        file_path.append(path)

    for root, dirs, _ in os.walk(base_save_dir):
        for dir_name in dirs:
            path = os.path.join(root, dir_name)
            if not check_link_path(path):
                os.chmod(path, mode)

    return file_path


class CustomizedRotatingFileHandler(RFHandler):
    """继承concurrent_log_handler包下的ConcurrentRotatingFileHandler类
    1.实现控制日志文件的并发写入和读取
    2.实现日志文件的权限控制
    3.日志文件的绕接
    """
    def doRollover(self) -> None:
        """重写doRoller，实现绕接，并且对日志文件的权限进行控制"""
        super().doRollover()
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                file_name = self.rotation_filename(f'{self.baseFilename}.{i}')
                if os.path.exists(file_name) and not check_link_path(str(file_name)):
                    os.chmod(file_name, MODE_440)

    def emit(self, record) -> None:
        """重写emit方法，实现日志内容长度限制"""
        tmp_out = record.getMessage()
        if tmp_out is not None and len(tmp_out) > LOG_RECORD_MAX_LEN:
            record.msg = tmp_out[:LOG_RECORD_MAX_LEN]
            record.args = ()
        super().emit(record)
        if not check_link_path(str(self.baseFilename)) and os.path.exists(self.baseFilename):
            os.chmod(self.baseFilename, MODE_640)

    def format(self, record):
        return log_format(self, record)

    def _open_lockfile(self):
        super()._open_lockfile()
        self._do_chmod(MODE_640)

    def _do_chmod(self, mode):
        if mode is not None and os.path.exists(self.lockFilename):
            os.chmod(self.lockFilename, mode)


def get_file_handler_list(file_level: Union[List, Tuple], file_path: Union[List, Tuple], max_file_size: int,
                          max_num_of_files: int) -> List:
    """获得日志的文件处理器"""
    logging_level = []
    for level in file_level:
        logging_level.append(_convert_level(level))

    max_file_size = max_file_size * 1024 * 1024

    file_formatter = logging.Formatter(const.default_filehandler_format)

    file_handlers = []
    for path, level in zip(file_path, logging_level):
        handler_name = f'{path}.{max_file_size}.{max_num_of_files}.{level}'

        if handler_name in file_handler_list:
            file_handlers.append(file_handler_list.get(handler_name))
        else:
            file_handler = CustomizedRotatingFileHandler(filename=path,
                                                         maxBytes=max_file_size,
                                                         backupCount=max_num_of_files,
                                                         delay=True)
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            file_handlers.append(file_handler)

            file_handler_list[handler_name] = file_handler
    return file_handlers


class MxLogger(logging.Logger):
    """Define Mx logger"""
    def __init__(self, name, **kwargs):
        super().__init__(name)
        self.source = None
        self.method = None
        self.propagate = False
        self.to_std = kwargs.get('to_std', True)
        self.stdout_nodes = kwargs.get('stdout_nodes', None)
        self.stdout_devices = kwargs.get('stdout_devices', (0,))
        self.stdout_level = kwargs.get('stdout_level', 'INFO')
        self.stdout_format = kwargs.get('stdout_format', '')
        self.file_level = kwargs.get('file_level', ('INFO', 'ERROR'))
        self.append_rank_dir = kwargs.get('append_rank_dir', True)
        self.file_name = kwargs.get('file_name', ('mx_logger.log', 'error.log'))
        self.max_file_size = kwargs.get('max_file_size', DEFAULT_MAX_LOG_FILE_SIZE)
        self.max_num_of_files = kwargs.get('max_num_of_files', DEFAULT_MAX_LOG_FILE_NUM)
        self.set_flag = False

    def makeRecord(
            self,
            name,
            level,
            fn,
            lno,
            msg,
            args,
            exc_info,
            func=None,
            extra=None,
            sinfo=None,
    ):
        """重写makeRecord方法，日志内容中增加发起端标识和pid信息"""
        if extra is None:
            extra = {}
        extra['source'] = self.source if hasattr(self, 'source') else UNKNOWN
        extra['pid'] = self.pid if hasattr(self, 'pid') else UNKNOWN
        args = log_args_black_list_characters_replace(args)
        return super().makeRecord(name, level=level, fn=fn, lno=lno, msg=msg, args=args,
                                                exc_info=exc_info,
                                                func=func, extra=extra, sinfo=sinfo)

    def set_logger(self):
        """设置logger输出格式以及日志输出路径"""
        if const.local_default_log_file_dir is None:
            const.get_local_default_log_file_dir()
        file_save_dir = os.path.expanduser(const.local_default_log_file_dir)

        validate_std_input_format(self.to_std, self.stdout_nodes, self.stdout_devices, self.stdout_level)
        validate_file_input_format(self.file_level, file_save_dir, self.append_rank_dir, self.file_name)

        rank_info = get_rank_info()

        self.to_std = judge_stdout(rank_info[0], rank_info[1], self.to_std, self.stdout_nodes, self.stdout_devices)
        if self.to_std:
            stream_handler = get_stream_handler(self.stdout_format, self.stdout_level)
            self.addHandler(stream_handler)

        file_path = get_file_path_list(file_save_dir, self.append_rank_dir, rank_info[2], rank_info[3], self.file_name)
        file_handlers = get_file_handler_list(self.file_level, file_path, self.max_file_size, self.max_num_of_files)
        for file_handler in file_handlers:
            self.addHandler(file_handler)
        self.setLevel(_convert_level('DEBUG'))
        self.set_flag = True

    # pylint: disable=W0221
    def _log(self,
             level,
             msg,
             args,
             **kwargs):
        """日志接口"""
        if not self.set_flag:
            try:
                self.set_logger()
            except (PathOwnerError, PathModeError, ValueError, MakeDirError, UnsupportedPlatformError, TypeError) as ex:
                if str(ex):
                    exc_type = sys.exc_info()
                    error_msg = f'An {str(exc_type[0])} occurred, {str(ex)}'
                    logging.error(error_msg)
                else:
                    logging.error(traceback.format_exc())
                sys.exit(ABNORMAL_EXIT_CODE)
            # pylint: disable=W0703
            except Exception:
                logging.error(traceback.format_exc())
                sys.exit(ABNORMAL_EXIT_CODE)
        super()._log(level, msg, args, **kwargs)


def set_logger_property(source):
    """设置日志实例的自定义属性"""
    for logger_name in logger_list:
        if source in LOGGER_SOURCE_LIST:
            logger_item = logger_list.get(logger_name)
            logger_item.source = source
            logger_item.pid = os.getpid()


def get_logger(logger_name: str = 'mx_logger', **kwargs) -> logging.Logger:
    """获取日志的实例

    Args:
        logger_name (str): 日志实例名称
        kwargs: 参数dict
            to_std (bool): 是否需要将日志内容输出到屏幕
            stdout_nodes (list[int] or tuple[int] or optional):
                需要输出日志内容到屏幕的节点列表
                默认: None, 表示所有节点的日志内容都会输出
                eg: [0, 1, 2, 3] or (0, 1, 2, 3): 表示0,1,2,3节点的日志内容都会输出到屏幕
            stdout_devices (list[int] or tuple[int] or optional):
                需要输出日志内容到屏幕的设备列表
                默认: None, 表示所有设备的日志内容都会输出
                eg: [0, 1, 2, 3] or (0, 1, 2, 3): 表示0,1,2,3设备的日志内容都会输出到屏幕
            stdout_level (str): 输出到屏幕的日志的等级，DEBUG, INFO, WARNING, ERROR, CRITICAL.
            stdout_format (str): 输出到屏幕的日志内容模板
            file_level (list[str] or tuple[str]): 落盘的日志的等级
                eg: ['INFO', 'ERROR'] 表示INFO以及ERROR等级以上的日志内容会落盘到对应的文件中
                列表的长度需要和文件名参数的长度保持一致
            append_rank_dir (bool): 是否需要按照rank进行日志区分
            file_name (list[str] or tuple[str]): 日志文件的名称列表
            max_file_size (int): 单个日志文件的大小阈值（MB）
            max_num_of_files (int): 日志绕接的文件数量的阈值

    Returns:
        mx_logger (logging.Logger): Logger
    """
    if logger_name in logger_list:
        return logger_list.get(logger_name)

    mx_logger = MxLogger(name=logger_name, **kwargs)
    logger_list[logger_name] = mx_logger

    return mx_logger


logger = get_logger('logger', to_std=True, file_name=['service.log'], file_level=['INFO'], append_rank_dir=True)
logger_without_std = get_logger('logger_without_std', to_std=False, file_name=['service.log'], file_level=['INFO'],
                                append_rank_dir=True)
operation_logger = get_logger('operation_logger', to_std=True, file_name=['operation.log'], file_level=['INFO'],
                              append_rank_dir=True)
operation_logger_without_std = get_logger('operation_logger_without_std', to_std=False, file_name=['operation.log'],
                                          file_level=['INFO'], append_rank_dir=True)


def record_operation_and_service_info_log(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)
    operation_logger_without_std.info(msg, *args, **kwargs)


def record_operation_and_service_error_log(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs)
    operation_logger_without_std.error(msg, *args, **kwargs)


def record_operation_and_service_warning_log(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)
    operation_logger_without_std.warning(msg, *args, **kwargs)
