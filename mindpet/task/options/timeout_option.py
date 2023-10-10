#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""timeout option"""

import re
import click
from click.exceptions import BadParameter
from mindpet.utils.constants import TIMEOUT_REGEX, HOURS_PER_DAY, SPACE_CHARACTER, EMPTY_STRING, SECONDS_PER_HOUR


class TimeoutOption(click.core.Option):
    """TimeoutOption class"""
    def __init__(self):
        super().__init__(
            param_decls=('-t', '--timeout'),
            type=str,
            default=None,
            help='if the task exceeds the time limit, it will be forcibly terminated.',
            callback=self.timeout_callback
        )

    @staticmethod
    def get_timeout_hours(match_result):
        """get_timeout_hours"""
        res = 0

        extract_days_str = match_result.group(1)
        if extract_days_str:
            res += (int(extract_days_str.strip('d')) * HOURS_PER_DAY)

        extract_hours_str = match_result.group(2)
        if extract_hours_str:
            extract_hours = int(extract_hours_str.strip('h'))
            if extract_hours >= HOURS_PER_DAY:
                raise BadParameter('Invalid param [timeout].')

            res += extract_hours

        if res == 0:
            raise BadParameter('Invalid param [timeout].')

        res *= SECONDS_PER_HOUR

        return res

    # pylint: disable=W0613
    def timeout_callback(self, ctx, param, value):
        """
        timeout参数click回调方法

        :param ctx: 上下文信息
        :param param: 参数属性
        :param value: 输入值
        :return: 回调处理后的参数值
        """
        if value is None:
            return value

        value = value.strip().replace(SPACE_CHARACTER, EMPTY_STRING).lower()
        match_result = re.match(TIMEOUT_REGEX, value)

        if not match_result:
            raise BadParameter('Invalid param [timeout].')

        # 将输入的timeout参数转化为秒数
        res = self.get_timeout_hours(match_result)

        return res
