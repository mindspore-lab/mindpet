#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import click
from click.exceptions import BadParameter


class QuietOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-q', '--quiet'),
            type=bool,
            is_flag=True,
            help='enable quiet mode, ignore the risk of permission rules of incoming parameters.',
            callback=self.quiet_callback
        )

    @staticmethod
    def quiet_callback(ctx, params, value):
        # --quiet参数仅允许被定义在首位, 其ctx上下文属性必须为空
        if value is True and ctx.params:
            raise BadParameter('Param [--quiet] should be set first.')

        return value
