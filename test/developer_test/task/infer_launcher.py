#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os
import stat
import time
import json
import logging
import argparse

"""
测试方式

export TEST_PATH=/home/xxx/mxTuningKit/test/developer_test/task/  修改为实际服务器存储mxTuningKit/test/developer_test/task/代码的目录

cd $TEST_PATH

mkdir dataset outputs pretrained_models ckpt

chmod 750 dataset outputs ckpt infer_launcher.py model_config_infer.yaml

mindpet infer --data_path $TEST_PATH/dataset/ --ckpt_path $TEST_PATH/ckpt/ --output_path $TEST_PATH/outputs/  \
--model_config_path $TEST_PATH/model_config_infer.yaml  --boot_file_path $TEST_PATH/infer_launcher.py
"""

LOCAL_INFER_RESULT = 'infer_result.json'
DEFAULT_FLAGS = os.O_RDWR | os.O_CREAT
DEFAULT_MODES = stat.S_IWUSR | stat.S_IRUSR


def test_infer_func(output_path):
    # 模拟推理过程
    for i in range(3):
        time.sleep(1)
        logging.info('inference has been going %ss', (i + 1))
    logging.info('inference has been done.')

    # 落盘推理结果
    infer_result = json.dumps({'infer_result1': 123, 'infer_result2': 456})
    infer_result_path = os.path.join(output_path, LOCAL_INFER_RESULT)
    with os.fdopen(os.open(infer_result_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
        json.dump(infer_result, file)
    logging.info('infer_result.json has been saved.')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', type=str, required=True)
    parser.add_argument('-cp', '--ckpt_path', type=str, required=True)
    parser.add_argument('-op', '--output_path', type=str, required=True)

    # model_config_infer.yaml
    parser.add_argument('-ip1', '--infer_param1', type=float, required=False)
    parser.add_argument('-ip2', '--infer_param2', type=int, required=False)

    args = parser.parse_args()

    logging.info('----------accepting params starts----------')
    logging.info('params1: %s', args.infer_param1)
    logging.info('params2: %s', args.infer_param2)
    logging.info('----------accepting params ends----------')
    logging.info('----------task starts----------')
    test_infer_func(args.output_path)
    logging.info('----------task ends----------')
