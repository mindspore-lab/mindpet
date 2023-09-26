#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""command option check utils."""

import os
from mindpet.log.log import logger
from mindpet.utils.io_utils import is_link_path
from mindpet.utils.entrance_monitor import entrance_monitor
from mindpet.utils.constants import EMPTY_STRING, BLACKLIST_CHARACTERS, BASE_WHITELIST_CHARACTERS, MAX_FILE_MB_SIZE, \
    MB_SIZE, ENTRANCE_TYPE, MIN_MODE, MAX_MODE, GB_SIZE
from mindpet.utils.exceptions import FileOversizeError, LinkPathError, LowDiskFreeSizeRiskError, PathContentError, \
    AbsolutePathError, PathLengthError, FileNameLengthError, PathGranularityError, PathOwnerError, PathModeError, \
    PathRightEscalationError, ManualCancelError


def get_real_path(path):
    """
    获取给定路径对应的真实路径

    :param path: 给定路径
    :return: 给定路径对应真实路径
    """
    if path is None or path == EMPTY_STRING:
        raise ValueError('Path is None or empty when getting real path.')

    return os.path.realpath(path)


class OptionBase:
    """Define command option."""
    def __init__(self, option_name, option_value):
        """
        构造方法
        :param option_name: 入参名称
        :param option_value: 入参值
        """
        if option_name is None or option_name == EMPTY_STRING:
            raise ValueError('Param [option_name] is None or empty.')

        if option_value is None or option_value == EMPTY_STRING:
            raise ValueError('Param [option_value] is None or empty.')

        self.option_name = option_name
        self.option_value = option_value

    def _is_file_path(self):
        """
        判断路径是否是一个文件
        :return: 是否是一个文件
        """
        return os.path.isfile(self.option_value)

    def _is_dir_path(self):
        """
        判断路径是否是一个文件夹
        :return: 是否是一个文件夹
        """
        return os.path.isdir(self.option_value)


class PathLengthCheckParam:
    """Define file path length check params."""
    def __init__(self, path_min_limit, path_max_limit, file_min_limit, file_max_limit):
        """
        路径长度校验项参数构造方法
        :param path_min_limit: 路径最小长度约束
        :param path_max_limit: 路径最大长度约束
        :param file_min_limit: 文件名最小长度约束
        :param file_max_limit: 文件名最大长度约束
        """
        self.path_min_limit = path_min_limit
        self.path_max_limit = path_max_limit
        self.file_min_limit = file_min_limit
        self.file_max_limit = file_max_limit


class PathContentCheckParam:
    """Define file path check params."""
    def __init__(self, base_whitelist_mode, extra_whitelist):
        """
        路径内容校验项参数构造方法
        :param base_whitelist_mode: 路径基础白名单模式
        :param extra_whitelist: 路径额外字符白名单
        """
        self.base_whitelist_mode = base_whitelist_mode
        self.extra_whitelist = extra_whitelist


class InteractionByEntrance(OptionBase):
    """Define interaction."""
    def __init__(self, option_name, option_value, notice_msg, notice_accept_msg, exception_type):
        """
        根据使用入口(CLI/SDK)执行不同的交互逻辑构造方法
        :param option_name: 参数名称
        :param option_value: 参数值
        :param notice_msg: 通知消息
        :param notice_accept_msg: 用户接受通知消息
        :param exception_type: 异常类型
        """
        super().__init__(option_name, option_value)

        if notice_msg is None or notice_msg == EMPTY_STRING:
            raise ValueError('Param [notice_msg] is None or empty.')

        if not isinstance(notice_msg, str):
            raise TypeError('Invalid type for param [notice_msg] when initializing InteractionByEntrance.')

        if notice_accept_msg is None or notice_accept_msg == EMPTY_STRING:
            raise ValueError('Param [notice_accept_msg] is None or empty.')

        if not isinstance(notice_accept_msg, str):
            raise TypeError('Invalid type for param [notice_accept_msg] when initializing InteractionByEntrance.')

        if exception_type is None:
            raise ValueError('Param [exception_type] is None.')

        if not issubclass(exception_type, Exception):
            raise TypeError('Invalid type for param [exception_type] when initializing InteractionByEntrance.')

        self.notice_msg = notice_msg
        self.notice_accept_msg = notice_accept_msg
        self.exception_type = exception_type

    def interact_by_entrance(self, force_quit, quiet):
        """
        根据不同的入口(CLI/SDK), 进行不同类型权限校验交互
        :param force_quit: 触发校验异常时，是否强制结束进程
        :param quiet: 是否启用安静模式
        """
        if force_quit is None:
            raise ValueError('Param [force_quit] is None.')

        if quiet is None:
            raise ValueError('Param [quiet] is None.')

        if not isinstance(force_quit, bool):
            raise TypeError('Invalid type for param [force_quit] in interact_by_entrance.')

        if not isinstance(quiet, bool):
            raise TypeError('Invalid type for param [quiet] in interact_by_entrance.')

        entrance_type = entrance_monitor.get_value(ENTRANCE_TYPE)

        if entrance_type == 'CLI':
            self._interact_by_cli(force_quit, quiet)
        elif entrance_type == 'SDK':
            self._interact_by_sdk(force_quit)
        else:
            raise ValueError('Invalid param [entrance_type].')

    def _interact_by_cli(self, force_quit, quiet):
        """
        CLI场景下的交互方式
        :param force_quit: 是否强制终止请求
        :param quiet: 是否启动安静模式
        """
        if quiet:
            if not force_quit:
                logger.warning(self.notice_msg)
            else:
                raise self.exception_type(self.notice_msg)
        else:
            ans = input(self.notice_msg + ' would you like to continue? (y/n)')
            if str(ans).lower() != 'y':
                raise ManualCancelError('Current command is cancelled.')
            logger.info(self.notice_accept_msg)

    def _interact_by_sdk(self, force_quit):
        """
        SDK场景下的交互方式
        :param force_quit: 是否强制终止请求
        """
        if force_quit:
            raise self.exception_type(self.notice_msg)
        logger.warning(self.notice_msg)


class PathContentBlacklistCharactersCheck(OptionBase):
    """Define path black list check"""
    def __init__(self, option_name, option_value):
        """
        路径内容黑名单字符校验构造方法
        :param option_name: 参数名称
        :param option_value: 参数值
        """
        super().__init__(option_name, option_value)
        self._check()

    def _check(self):
        """
        路径内容黑名单字符校验
        """
        self.option_value = str(self.option_value).strip()

        for item in BLACKLIST_CHARACTERS:
            if item in self.option_value:
                raise PathContentError(f'Invalid character(s) in param [{self.option_name}].')


class AbsolutePathCheck(OptionBase):
    """Define absolute path check."""
    def __init__(self, option_name, option_value):
        """
        绝对路径校验构造方法
        :param option_name: 参数名称
        :param option_value: 参数值
        """
        super().__init__(option_name, option_value)
        self._check()

    def _check(self):
        """
        绝对路径校验
        """
        if not os.path.isabs(self.option_value):
            raise AbsolutePathError(f'Param [{self.option_name}] is not an absolute path.')


class PathExistCheck(OptionBase):
    """Define path exist check."""
    def __init__(self, option_name, option_value):
        """
        路径存在校验构造方法
        :param option_name: 参数名称
        :param option_value: 参数值
        """
        super().__init__(option_name, option_value)
        self._check()

    def _check(self):
        """
        路径存在校验
        """
        if not os.path.exists(self.option_value):
            raise FileNotFoundError(f'Path of param [{self.option_name}] does not exist.')


class LinkPathCheck(OptionBase):
    """Define link path exist check."""
    def __init__(self, option_name, option_value):
        """
        链接路径校验构造方法
        :param option_name: 参数名称
        :param option_value: 参数值
        """
        super().__init__(option_name, option_value)
        self._check()

    def _check(self):
        """
        链接路径校验
        """
        if is_link_path(self.option_value):
            raise LinkPathError(f'Path of param [{self.option_name}] is a link path.')


class PathContentLengthCheck(OptionBase):
    """Define path content option length check."""
    def __init__(self, option_name, option_value, path_min_limit, path_max_limit, file_min_limit, file_max_limit):
        """
        路径内容长度校验构造方法
        :param option_name: 参数值
        :param option_value: 参数方法
        :param path_min_limit: 路径最小长度限制
        :param path_max_limit: 路径最大长度限制
        :param file_min_limit: 路径中文件名最小长度限制
        :param file_max_limit: 路径中文件名最大长度限制
        """
        if path_min_limit is not None and not isinstance(path_min_limit, int):
            raise TypeError('Invalid type for param [path_min_limit] when initializing PathContentLengthCheck.')

        if path_max_limit is not None and not isinstance(path_max_limit, int):
            raise TypeError('Invalid type for param [path_max_limit] when initializing PathContentLengthCheck.')

        if file_min_limit is not None and not isinstance(file_min_limit, int):
            raise TypeError('Invalid type for param [file_min_limit] when initializing PathContentLengthCheck.')

        if file_max_limit is not None and not isinstance(file_max_limit, int):
            raise TypeError('Invalid type for param [file_max_limit] when initializing PathContentLengthCheck.')

        super().__init__(option_name, option_value)
        self._check(path_min_limit, path_max_limit, file_min_limit, file_max_limit)

    def _check(self, path_min_limit, path_max_limit, file_min_limit, file_max_limit):
        """
        路径内容长度校验
        :param path_min_limit: 路径最小长度, 默认为0
        :param path_max_limit: 路径最大长度
        :param file_min_limit: 文件最小长度
        :param file_max_limit: 文件最大长度
        """
        if not self._path_length_check_item(path_min_limit=path_min_limit, path_max_limit=path_max_limit):
            raise PathLengthError(f'Length of path in param [{self.option_name}] is invalid.')

        if self._is_file_path():
            if not self._file_length_check_item(file_min_limit=file_min_limit, file_max_limit=file_max_limit):
                raise FileNameLengthError(f'Length of file name in param [{self.option_name}] is invalid.')

    def _path_length_check_item(self, path_min_limit, path_max_limit):
        """
        路径长度校验
        :param path_min_limit: 路径最小长度限制, 默认值为0
        :param path_max_limit: 路径最大长度限制
        :return: 路径粒度长度是否合规
        """
        path_min_limit = 0 if path_min_limit is None else path_min_limit

        if path_min_limit is not None and path_min_limit < 0:
            logger.error('Param [path_min_limit] should be non-negative integer.')
            return False

        if path_max_limit is not None and path_max_limit < 0:
            logger.error('Param [path_max_limit] should be non-negative integer.')
            return False

        if path_min_limit is not None and path_max_limit is not None and path_min_limit > path_max_limit:
            logger.error('Param [path_min_limit] should be less than or equal to path_max_limit.')
            return False

        if path_min_limit is not None and len(self.option_value) < path_min_limit:
            return False

        if path_max_limit is not None and len(self.option_value) > path_max_limit:
            return False

        return True

    def _file_length_check_item(self, file_min_limit, file_max_limit):
        """
        路径中文件名长度校验
        :param file_min_limit: 文件名最小长度限制, 默认值为0
        :param file_max_limit: 文件名最大长度限制
        :return: 文件名粒度长度是否合规
        """
        file_name = os.path.basename(self.option_value)
        file_min_limit = 0 if file_min_limit is None else file_min_limit

        if file_min_limit is not None and file_min_limit < 0:
            logger.error('Param [file_min_limit] should be non-negative integer.')
            return False

        if file_max_limit is not None and file_max_limit < 0:
            logger.error('Param [file_max_limit] should be non-negative integer.')
            return False

        if file_min_limit is not None and file_max_limit is not None and file_min_limit > file_max_limit:
            logger.error('Param [file_min_limit] should be less than or equal to file_max_limit.')
            return False

        if file_min_limit is not None and len(file_name) < file_min_limit:
            return False

        if file_max_limit is not None and len(file_name) > file_max_limit:
            return False

        return True


class PathContentCharacterCheck(OptionBase):
    """Define path content option character check."""
    def __init__(self, option_name, option_value, base_whitelist_mode, extra_whitelist):
        """
        路径内容校验构造方法
        :param option_name: 参数名称
        :param option_value: 参数值
        :param base_whitelist_mode: 基础白名单模式
        :param extra_whitelist: 额外白名单列表
        """
        super().__init__(option_name, option_value)

        if base_whitelist_mode is None or base_whitelist_mode == EMPTY_STRING:
            raise ValueError('Param [base_whitelist_mode] is None or empty.')

        if not isinstance(base_whitelist_mode, str):
            raise TypeError('Invalid type for param [base_whitelist_mode] when initializing PathContentCharacterCheck.')

        if extra_whitelist is not None and not isinstance(extra_whitelist, list):
            raise TypeError('Invalid type for param [extra_whitelist] when initializing PathContentCharacterCheck.')

        self._check(base_whitelist_mode, extra_whitelist)

    @staticmethod
    def get_base_whitelist_by_mode(base_whitelist_mode):
        """
        根据base_whitelist_mode获得对应字符粒度参数校验基础白名单
        :param base_whitelist_mode: 模式类型，包括全大写/全小写/数字/大小写字母/大小写字母和数字
        :return: 基础白名单字符列表
        """
        if base_whitelist_mode not in BASE_WHITELIST_CHARACTERS:
            raise ValueError('Invalid param [base_whitelist_mode], only support %s' %
                             list(BASE_WHITELIST_CHARACTERS.keys()))

        return BASE_WHITELIST_CHARACTERS.get(base_whitelist_mode)

    def _check(self, base_whitelist_mode, extra_whitelist):
        """
        路径内容校验
        :param base_whitelist_mode: 基础白名单模式
        :param extra_whitelist: 额外补充白名单字符
        """
        base_whitelist_mode = self.get_base_whitelist_by_mode(base_whitelist_mode)
        extra_whitelist = [] if extra_whitelist is None else extra_whitelist
        whitelist = base_whitelist_mode + extra_whitelist

        for val in self.option_value:
            if val not in whitelist:
                raise PathContentError(f'Invalid character(s) in param [{self.option_name}].')


class PathGranularityCheck(OptionBase):
    """Define path option granular check."""
    def __init__(self, option_name, option_value, path_including_file):
        """
        路径粒度校验(文件夹/文件粒度)构造方法
        :param option_name: 参数名称
        :param option_value: 参数值
        :param path_including_file: 路径是否包含文件名, 传None表示同时兼容
        """
        super().__init__(option_name, option_value)

        if path_including_file is not None and not isinstance(path_including_file, bool):
            raise TypeError('Invalid type for param [path_including_file] when initializing PathGranularityCheck.')

        self.path_including_file = path_including_file
        self._check()

    def _check(self):
        """
        路径粒度校验(文件夹/文件粒度)
        """
        status = True

        if self.path_including_file is not None:
            status = self._is_file_path() if self.path_including_file else self._is_dir_path()

        if not status:
            raise PathGranularityError(
                f"Param [{self.option_name}] should be a {'file' if self.path_including_file else 'dir'} path.")


class PathRightEscalationCheck(OptionBase):
    """Define path right option escalation check."""
    def __init__(self, option_name, option_value, mode, force_quit, quiet):
        """
        路径权限提升校验构造方法
        :param option_name: 参数名称
        :param option_value: 参数值
        :param mode: 权限约束
        :param force_quit: 异常是否强制退出
        :param quiet: 安静模式
        """
        super().__init__(option_name, option_value)
        self._check(mode, force_quit, quiet)

    @staticmethod
    def mode_property_check(mode):
        """
        mode属性合法性检查, 返回合法的路径权限
        :param mode: 路径要求的权限范围, 要求必须是字符串类型, 包含3位数字, 数字范围[0,7], 例如'755'
        :return 路径权限值
        """
        if mode is None:
            raise ValueError('Param [mode] is None.')

        if not isinstance(mode, str):
            raise TypeError('Invalid type for param [mode] in mode_property_check.')

        mode = str(mode)
        if len(mode) != 3:
            raise ValueError('Invalid param [mode] for mode_property_check.')

        for pri in mode:
            if not pri.isdigit():
                raise TypeError('Invalid param [mode] for mode_property_check.')

            pri = int(pri)
            if pri < MIN_MODE or pri > MAX_MODE:
                raise ValueError('Invalid param [mode] for mode_property_check.')

        ret = (int(mode[0]), int(mode[1]), int(mode[2]))

        return ret

    def _check(self, mode, force_quit, quiet):
        """
        路径权限提升校验
        :param mode:路径权限约束
        :param force_quit: 发现权限提升问题是否强制退出
        :param quiet: 安静模式
        """
        is_legal_mode = True

        # 路径属主一致性校验
        try:
            self._path_owner_check()
        except PathOwnerError:
            is_legal_mode = False

        # 路径权限范围校验
        try:
            self._path_mode_check(mode)
        except PathModeError:
            is_legal_mode = False

        if not is_legal_mode:
            notice_msg = f'Param [{self.option_name}] may have risk of rights escalation ' \
                         f'(owner of path from param [{self.option_name}] is not current login user, ' \
                         f'or other users have unnecessary permissions of path from param [{self.option_name}]).'

            notice_accept_msg = f'Current login user has accepted the risk of rights escalation of param ' \
                                f'[{self.option_name}].'

            interaction_op = InteractionByEntrance(option_name=self.option_name,
                                                   option_value=self.option_value,
                                                   notice_msg=notice_msg,
                                                   notice_accept_msg=notice_accept_msg,
                                                   exception_type=PathRightEscalationError)
            interaction_op.interact_by_entrance(force_quit=force_quit, quiet=quiet)

    def _path_owner_check(self):
        """
        路径属主一致性校验, 判断路径属主和当前用户是否一致
        """
        path_owner = os.stat(self.option_value).st_uid
        current_login_user = os.geteuid()

        if path_owner != current_login_user:
            raise PathOwnerError(f'Invalid owner of path of param [{self.option_name}].')

    def _path_mode_check(self, mode):
        """
        路径权限校验
        :param mode: 路径要求的权限范围
        """
        required_mode = self.mode_property_check(mode)
        actual_mode = self._get_path_mode()

        mode_size = len(required_mode)
        for idx in range(mode_size):
            if required_mode[idx] | actual_mode[idx] != required_mode[idx]:
                raise PathModeError(f'Invalid mode of path of param [{self.option_name}].')

    def _get_path_mode(self):
        """
        获得当前路径对应的权限, 分用户权限/同组权限/其他权限, 使用十进制表示
        :return: tuple元素, (用户权限, 同组权限, 其他权限)
        """
        mode = oct(os.stat(self.option_value).st_mode)[-3:]
        ret = (int(mode[0]), int(mode[1]), int(mode[2]))
        return ret


class FileSizeCheck(OptionBase):
    """Define file size check."""
    def __init__(self, option_name, option_value, path_including_file):
        """
        文件大小校验构造方法
        :param option_name: 参数名称
        :param option_value: 参数值
        :param path_including_file: 路径是否包含文件名
        """
        super().__init__(option_name, option_value)

        if path_including_file is None:
            raise ValueError('Param [path_including_file] is None.')

        if not isinstance(path_including_file, bool):
            raise TypeError('Invalid type for param [path_including_file] when initializing FileSizeCheck.')

        self._check(path_including_file)

    def _check(self, path_including_file):
        """
        文件大小校验
        """
        if path_including_file:
            size = os.stat(self.option_value).st_size / MB_SIZE
            if size > MAX_FILE_MB_SIZE:
                raise FileOversizeError(f'File of param [{self.option_name}] is too large.')


class DiskFreeSpaceCheck(OptionBase):
    """Define disk free space check."""
    def __init__(self, option_name, option_value, free_space_limit, force_quit, quiet):
        """
        路径所在磁盘剩余空间校验构造方法
        :param option_name: 参数名称
        :param option_value: 参数值
        :param free_space_limit: 剩余最小空间约束
        :param force_quit: 异常是否强制退出
        :param quiet: 安静模式
        """
        super().__init__(option_name, option_value)

        if free_space_limit is None:
            raise ValueError('Param [free_space_limit] is None.')

        if not isinstance(free_space_limit, int):
            raise TypeError('Invalid type for param [free_space_limit] when initializing DiskFreeSpaceCheck.')

        if free_space_limit < 0:
            raise ValueError('Invalid param [free_space_limit] when initializing DiskFreeSpaceCheck.')

        self.free_space_limit = free_space_limit
        self._check(force_quit, quiet)

    def _check(self, force_quit, quiet):
        """
        路径所在磁盘剩余空间校验
        :param force_quit: 是否强制退出
        :param quiet: 是否安静模式
        """
        info = os.statvfs(self.option_value)
        free_size = info.f_bsize * info.f_bavail

        # pylint: disable=W1203
        logger.info(
            f'Disk where param {self.option_name} is located has {round(free_size / GB_SIZE, 2)} GB free space.')

        if free_size <= self.free_space_limit:
            notice_msg = f'Free space of disk where param {self.option_name} is located is no greater than 1 GB, ' \
                         f'there is risk of disk exhaustion.'

            notice_accept_msg = 'Current login user has accepted the risk of disk exhaustion.'

            interaction_op = InteractionByEntrance(option_name=self.option_name,
                                                   option_value=self.option_value,
                                                   notice_msg=notice_msg,
                                                   notice_accept_msg=notice_accept_msg,
                                                   exception_type=LowDiskFreeSizeRiskError)
            interaction_op.interact_by_entrance(force_quit=force_quit, quiet=quiet)
