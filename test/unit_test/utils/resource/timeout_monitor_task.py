#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import time
import logging

logging.getLogger().setLevel(logging.INFO)

TIMEOUT = 20


def timeout_monitor_task():
    logging.info('Start timeout_monitor_task time count.')
    for i in range(TIMEOUT):
        time.sleep(1)
        logging.info(f'Sleep {i + 1} second.')
    logging.info('Finish timeout_monitor_task time count finish.')


if __name__ == '__main__':
    timeout_monitor_task()
