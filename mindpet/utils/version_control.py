#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""Version Control APIs"""
import mindspore as ms
from mindspore import nn
from .version_utils import is_version_ge
from ..layers.activation import get_activation, _activation

# pylint: disable=W0127
_activation = _activation
# pylint: disable=W0127
get_activation = get_activation


def get_dropout(dropout_prob):
    if is_version_ge(ms.__version__, '1.11.0'):
        dropout = nn.Dropout(p=dropout_prob)
    else:
        dropout = nn.Dropout(keep_prob=1 - dropout_prob)
    return dropout
