#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""path check param"""

from mindpet.utils.constants import SPACE_CHARACTER
from mindpet.security.param_check.option_check_utils import PathLengthCheckParam, PathContentCheckParam
from mindpet.utils.constants import DEFAULT_PATH_LEN_MIN_LIMIT, DEFAULT_PATH_LEN_MAX_LIMIT, DEFAULT_FILE_LEN_MIN_LIMIT, \
    DEFAULT_FILE_LEN_MAX_LIMIT

default_folder_length_check_param = PathLengthCheckParam(path_min_limit=DEFAULT_PATH_LEN_MIN_LIMIT,
                                                         path_max_limit=DEFAULT_PATH_LEN_MAX_LIMIT,
                                                         file_min_limit=None,
                                                         file_max_limit=None)
default_file_length_check_param = PathLengthCheckParam(path_min_limit=DEFAULT_PATH_LEN_MIN_LIMIT,
                                                       path_max_limit=DEFAULT_PATH_LEN_MAX_LIMIT,
                                                       file_min_limit=DEFAULT_FILE_LEN_MIN_LIMIT,
                                                       file_max_limit=DEFAULT_FILE_LEN_MAX_LIMIT)
default_folder_content_check_param = PathContentCheckParam(base_whitelist_mode='ALL',
                                                           extra_whitelist=['/', '-', '_', SPACE_CHARACTER])
default_file_content_check_param = PathContentCheckParam(base_whitelist_mode='ALL',
                                                         extra_whitelist=['.', '/', '-', '_', SPACE_CHARACTER])
