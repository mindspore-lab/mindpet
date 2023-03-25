#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from tk.utils.constants import EMPTY_STRING


# mxTuningKit包基础异常
class TKError(Exception):
    def __init__(self, error_info=None):
        super(TKError, self).__init__()
        self.error_info = error_info

    def __str__(self):
        return self.error_info if self.error_info else EMPTY_STRING


# 基础操作异常
class UnexpectedError(TKError):
    def __init__(self, error_info=None):
        super(UnexpectedError, self).__init__(error_info)


class MakeDirError(TKError):
    def __init__(self, error_info=None):
        super(MakeDirError, self).__init__(error_info)


class ReadYamlFileError(TKError):
    def __init__(self, error_info=None):
        super(ReadYamlFileError, self).__init__(error_info)


class ManualCancelError(TKError):
    def __init__(self, error_info=None):
        super(ManualCancelError, self).__init__(error_info)


# 任务异常
class CreateProcessError(TKError):
    def __init__(self, error_info=None):
        super(CreateProcessError, self).__init__(error_info)


class MonitorProcessRspError(TKError):
    def __init__(self, error_info=None):
        super(MonitorProcessRspError, self).__init__(error_info)


class TaskError(TKError):
    def __init__(self, error_info=None):
        super(TaskError, self).__init__(error_info)


class UnsupportedPlatformError(TKError):
    def __init__(self, error_info=None):
        super(UnsupportedPlatformError, self).__init__(error_info)


# 接口参数校验异常
class LinkPathError(TKError):
    def __init__(self, error_info=None):
        super(LinkPathError, self).__init__(error_info)


class LowDiskFreeSizeRiskError(TKError):
    def __init__(self, error_info=None):
        super(LowDiskFreeSizeRiskError, self).__init__(error_info)


class FileOversizeError(TKError):
    def __init__(self, error_info=None):
        super(FileOversizeError, self).__init__(error_info)


class PathContentError(TKError):
    def __init__(self, error_info=None):
        super(PathContentError, self).__init__(error_info)


class PathLengthError(TKError):
    def __init__(self, error_info=None):
        super(PathLengthError, self).__init__(error_info)


class FileNameLengthError(TKError):
    def __init__(self, error_info=None):
        super(FileNameLengthError, self).__init__(error_info)


class AbsolutePathError(TKError):
    def __init__(self, error_info=None):
        super(AbsolutePathError, self).__init__(error_info)


class PathGranularityError(TKError):
    def __init__(self, error_info=None):
        super(PathGranularityError, self).__init__(error_info)


class PathOwnerError(TKError):
    def __init__(self, error_info=None):
        super(PathOwnerError, self).__init__(error_info)


class PathModeError(TKError):
    def __init__(self, error_info=None):
        super(PathModeError, self).__init__(error_info)


class PathRightEscalationError(TKError):
    def __init__(self, error_info=None):
        super(PathRightEscalationError, self).__init__(error_info)


class PathLoopError(TKError):
    def __init__(self, error_info=None):
        super(PathLoopError, self).__init__(error_info)


# model_config配置文件校验异常
class ModelConfigKeysInfoError(TKError):
    def __init__(self, error_info=None):
        super(ModelConfigKeysInfoError, self).__init__(error_info)


class ModelConfigParamsInfoError(TKError):
    def __init__(self, error_info=None):
        super(ModelConfigParamsInfoError, self).__init__(error_info)


class ModelConfigFreezeInfoError(TKError):
    def __init__(self, error_info=None):
        super(ModelConfigFreezeInfoError, self).__init__(error_info)
