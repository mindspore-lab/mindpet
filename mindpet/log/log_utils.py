#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""Log APIs"""
import os
import stat
import platform
from typing import Dict, List, Tuple, Union

from mindpet.utils.exceptions import MakeDirError, UnsupportedPlatformError, PathOwnerError
from mindpet.utils.constants import EMPTY_STRING

LOG_CONTENT_BLACK_LIST = ('\r', '\n', '\t', '\f', '\v', '\b', '\u000A', '\u000D', '\u000C',
                          '\u000B', '\u0008', '\u007F', '\u0009')
BLANKS = '    '


def check_list(var_name: str, list_var: Union[Tuple, List], num: int):
    """确定节点或者设备的参数的合法性

    Args:
        var_name (str): 参数名称
        list_var (tuple or list): list参数列表
        num (int): 节点数或设备数

    Returns:
        None
    """
    for value in list_var:
        if value >= num:
            raise ValueError(f'The index of the {var_name} needs to be less than the number of nodes {num}.')


def generate_rank_list(stdout_nodes: Union[List, Tuple], stdout_devices: Union[List, Tuple]):
    """生成需要输出的日志内容的rank列表

    Args:
        stdout_nodes (list or tuple): 需要输出日志内容的节点列表
        stdout_devices (list or tuple): 需要输出日志内筒的设备列表

    Returns:
        rank_list (list): 日志内容输出的rank列表
    """
    rank_list = []
    for node in stdout_nodes:
        for device in stdout_devices:
            rank_list.append(8 * node + device)

    return rank_list


def convert_nodes_devices_input(var: Union[List, Tuple, Dict[str, int], None], num: int) -> Union[List, Tuple]:
    """转换节点和设备的参数，转换为列表形式

    Args:
        var (list[int] or tuple[int] or dict[str, int] or optional):
            需要转换的参数名
        num (str): 节点数或者设备数

    Returns:
        var (list[int] or tuple[int]): list形式的节点或者设备
    """
    if var is None:
        var = tuple(range(num))
    elif isinstance(var, dict):
        var = tuple(range(var['start'], var['end']))

    return var


def create_dirs(path, mode=0o750):
    """递归创建文件夹"""
    p_path = os.path.join(path, "../..")
    abspath = os.path.abspath(p_path)
    if not os.path.exists(abspath):
        os.makedirs(abspath, mode, exist_ok=True)
    if not os.path.exists(path):
        os.makedirs(path, mode, exist_ok=True)


def get_env_info(env_name, default_value):
    env_info = os.getenv(env_name, default_value)
    try:
        res = int(env_info)
    except ValueError:
        res = default_value
    if res < 0:
        res = default_value
    return res


def get_rank_info() -> Tuple[int, int, int, int]:
    """获得环境中的rank信息

    Returns:
        server_id (int): 设备id
        device_id (int): 在本地的卡id
        rank_id (int): 全局的卡id
        rank_size (int): 总卡数
    """
    rank_id = get_env_info('RANK_ID', 0)
    rank_size = get_env_info('RANK_SIZE', 1)
    server_id = get_env_info('SERVER_ID', 0)
    device_id = get_env_info('DEVICE_ID', 0)
    device_id = device_id % 8
    rank_info = rank_id, rank_size, server_id, device_id

    return rank_info


def get_num_nodes_devices(rank_size: int) -> Tuple[int, int]:
    """根据rank大小计算节点数和设备数

    Args:
        rank_size (int): rank 大小

    Returns:
       num_nodes (int):节点数
       num_devices (int): 卡数
    """
    if rank_size in (2, 4, 8):
        num_nodes = 1
        num_devices = rank_size
    else:
        num_nodes = rank_size // 8
        num_devices = 8

    return num_nodes, num_devices


def log_args_black_list_characters_replace(args):
    """日志内容参数黑名单校验"""
    res = []
    if args is None or len(args) == 0:
        return args
    if isinstance(args, (list, tuple)):
        for arg in args:
            replace = character_replace(arg)
            res.append(replace)
        args = tuple(res)
    else:
        args = character_replace(args)
    return args


def character_replace(content):
    """字符串黑名单过滤"""
    if not isinstance(content, str):
        return content

    content = str(content)
    for forbidden_str in LOG_CONTENT_BLACK_LIST:
        if forbidden_str in content:
            content = content.replace(forbidden_str, '')
    while BLANKS in content:
        content = content.replace(BLANKS, ' ')
    return content


def wrap_local_working_directory(file_name, specific_path_config=None):
    """
    根据当前用户身份, 组装对应本地缓存文件路径
    默认文件存放位置$HOME/.cache/Huawei/mxTuningKit
    需要存放特定文件夹需指定specific_path_config = {'path': 路径, 'rule': 权限范围}
    """

    def home_path_check(path):
        """
        HOME路径属主校验以及其他用户不可写校验
        """
        path_owner = os.stat(path).st_uid

        current_login_user = os.geteuid()

        if path_owner != current_login_user:
            raise PathOwnerError('The owner of $HOME path is not current login user.')

    if file_name is None or file_name == '':
        raise ValueError('[file_name] is None or empty.')

    # 针对Linux环境, 在当前用户$HOME目录下指定文件夹存放文件, 同时约束文件夹访问权限
    if platform.system().lower() == 'linux':
        home_path = os.getenv('HOME')

        if home_path is None or home_path == EMPTY_STRING:
            raise ValueError('Wrong environment variables')

        home_path_check(home_path)

        full_path = os.path.join(home_path, '.cache', 'Huawei', 'mxTuningKit')

        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path, exist_ok=True, mode=stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
            except Exception as ex:
                raise MakeDirError('Failed to create directory.') from ex
    else:
        raise UnsupportedPlatformError('Current platform is not supported, only support Linux.')

    if specific_path_config is not None:
        specific_path, specific_path_rule = specific_path_config_legality_check(specific_path_config)

        full_path = os.path.join(full_path, specific_path)

        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path, exist_ok=True, mode=specific_path_rule)
            except Exception as ex:
                raise MakeDirError('Failed to create directory.') from ex

    wrap_path = os.path.join(full_path, file_name).replace('\\', '/')

    if not os.path.exists(wrap_path):
        try:
            os.makedirs(wrap_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP, exist_ok=True)
        except Exception as ex:
            raise MakeDirError('Failed to create directory.') from ex

    return wrap_path


def specific_path_config_legality_check(specific_path_config):
    """specific_path_config_legality_check"""
    specific_path = specific_path_config.get('path')

    if specific_path is None:
        raise ValueError('[specific_path] is None.')

    specific_path_rule = specific_path_config.get('rule')

    if specific_path_rule is None:
        raise ValueError('[specific_path_rule] is None.')

    return [specific_path, specific_path_rule]


class Const:
    """Const"""
    def __init__(self):
        self.level = ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        self.local_default_log_file_dir = None
        self.node_dir_format = 'node_{}/'
        self.rank_dir_format = 'device_{}/'
        self.default_filehandler_format = \
            '[%(levelname)s] %(asctime)s [%(pid)s] [%(source)s] : %(message)s'
        self.default_stdout_format = '[%(levelname)s] %(asctime)s [%(pid)s] [%(source)s] : %(message)s'
        self.default_model_log_format = '%(asctime)s [%(pid)s] [Model]: %(message)s'

    def __setattr__(self, var, value):
        if not var.islower():
            raise ValueError(f'Const name {var} is not all lowercase.')
        self.__dict__[var] = value

    def get_local_default_log_file_dir(self):
        self.local_default_log_file_dir = wrap_local_working_directory('log')


const = Const()


def check_link_path(path):
    """
    校验传入的路径是否为软链接
    """
    path = os.path.abspath(path)
    return os.path.islink(path)
