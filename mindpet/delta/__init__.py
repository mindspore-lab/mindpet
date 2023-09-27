#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""Mindpet delta init"""
from mindpet.delta.adapter import AdapterDense, AdapterLayer
from mindpet.delta.lora import LoRADense
from mindpet.delta.low_rank_adapter import LowRankAdapterDense, LowRankAdapterLayer
from mindpet.delta.prefix_layer import PrefixLayer
from mindpet.delta.ptuning2 import PrefixEncoder
from mindpet.delta.r_drop import RDropLoss, rdrop_repeat

__all__ = [
    "LoRADense",
    "PrefixLayer",
    "LowRankAdapterDense",
    "LowRankAdapterLayer",
    "AdapterDense",
    "AdapterLayer",
    "RDropLoss",
    "rdrop_repeat",
    "PrefixEncoder",
]
