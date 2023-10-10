#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""Entrance Monitor"""
class EntranceMonitor:
    """Entrance Monitor"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    @staticmethod
    def init():
        global MONITOR_DICT
        MONITOR_DICT = {}

    @staticmethod
    def set_value(key, value):
        MONITOR_DICT[key] = value

    @staticmethod
    def get_value(key):
        return MONITOR_DICT.get(key)


MONITOR_DICT = {}
entrance_monitor = EntranceMonitor()
entrance_monitor.init()
