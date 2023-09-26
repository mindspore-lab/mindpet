#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import multiprocessing
import os
import time
import argparse
import logging

"""
测试方式

export TEST_PATH=/home/xxx/mxTuningKit/test/developer_test/task/  修改为实际服务器存储mxTuningKit/test/developer_test/task/代码的目录

cd $TEST_PATH

mkdir dataset outputs pretrained_models

chmod 750 dataset outputs pretrained_models finetune_launcher.py model_config_finetune.yaml

mindpet finetune --data_path $TEST_PATH/dataset/ --output_path $TEST_PATH/outputs/  \
--pretrained_model_path $TEST_PATH/pretrained_models/ \
--model_config_path $TEST_PATH/model_config_finetune.yaml --boot_file_path $TEST_PATH/finetune_launcher.py
"""

TEST_MODE = 0  # 0是单进程场景, 1是多进程场景


def test_func():
    env_id = os.getenv('DEVICE_ID', default=0)

    logging.info(f'device : {env_id} model training starts')
    for i in range(5):
        time.sleep(1)
        logging.info(f'device: {env_id} model training {i + 1}')
    logging.info(f'device: {env_id} model training ends')


def single_process(process_num, sleep_time):
    logging.info('create new process, process number: %s', process_num)
    time.sleep(sleep_time)
    logging.info('process number: %s finish.', process_num)


def test_func_multi_processes():
    logging.info('multi-processing scenario start.')

    for i in range(5):
        cur_process = multiprocessing.Process(target=single_process, args=(i, 8))
        cur_process.start()

    logging.info('multi-processing scenario finish.')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', type=str, required=True)
    parser.add_argument('-ac', '--advanced_config', type=str, required=False)
    parser.add_argument('-op', '--output_path', type=str, required=True)
    parser.add_argument('-pm', '--pretrained_model_path', type=str, required=False)
    parser.add_argument('-lr', '--learning_rate', type=float, required=False)
    parser.add_argument('-bs', '--batch_size', type=int, required=False)
    args = parser.parse_args()

    logging.info('----------accepting params starts----------')
    logging.info(args)
    logging.info('----------accepting params ends----------')

    logging.info('----------task starts----------')

    if TEST_MODE == 0:
        test_func()
    elif TEST_MODE == 1:
        test_func_multi_processes()
    else:
        logging.error('wrong param: [test_mode]')
        raise ValueError('wrong param: [test_mode]')

    logging.info('----------task ends----------')
