#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""MindPet Exceptions"""
from mindpet.utils.constants import EMPTY_STRING


# MindPet包基础异常
class MindPetError(Exception):
    def __init__(self, error_info=None):
        super().__init__()
        self.error_info = error_info

    def __str__(self):
        return self.error_info if self.error_info else EMPTY_STRING


# 基础操作异常
class UnexpectedError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class MakeDirError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class ReadYamlFileError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class ManualCancelError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


# 任务异常
class CreateProcessError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class MonitorProcessRspError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class TaskError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class UnsupportedPlatformError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


# 接口参数校验异常
class LinkPathError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class LowDiskFreeSizeRiskError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class FileOversizeError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class PathContentError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class PathLengthError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class FileNameLengthError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class AbsolutePathError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class PathGranularityError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class PathOwnerError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class PathModeError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class PathRightEscalationError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class PathLoopError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


# model_config配置文件校验异常
class ModelConfigKeysInfoError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class ModelConfigParamsInfoError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)


class ModelConfigFreezeInfoError(MindPetError):
    # pylint: disable=W0235
    def __init__(self, error_info=None):
        super().__init__(error_info)
