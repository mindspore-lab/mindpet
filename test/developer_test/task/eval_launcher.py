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

chmod 750 dataset outputs ckpt eval_launcher.py model_config_eval.yaml

mindpet evaluate --data_path $TEST_PATH/dataset/ --ckpt_path $TEST_PATH/ckpt/ --output_path $TEST_PATH/outputs/  \
--model_config_path $TEST_PATH/model_config_eval.yaml  --boot_file_path $TEST_PATH/eval_launcher.py
"""

LOCAL_DATA_PATH = './dataset/'
LOCAL_OUTPUT_PATH = './outputs/'
LOCAL_CKPT_PATH = './ckpt/'
LOCAL_EVAL_RESULT = 'eval_result.json'
DEFAULT_FLAGS = os.O_RDWR | os.O_CREAT
DEFAULT_MODES = stat.S_IWUSR | stat.S_IRUSR


def test_eval_func(output_path):
    for i in range(3):
        time.sleep(1)
        logging.info('evaluation has been going %ss', (i + 1))
    logging.info('evaluation has been done.')
    eval_result = json.dumps({'A': 123, 'B': 234, 'C': 'mx is the best'})
    eval_result_path = os.path.join(output_path, LOCAL_EVAL_RESULT)
    with os.fdopen(os.open(eval_result_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
        json.dump(eval_result, file)
    logging.info('eval_result.json has been saved.')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', type=str, required=True)
    parser.add_argument('-cp', '--ckpt_path', type=str, required=True)
    parser.add_argument('-op', '--output_path', type=str, required=True)

    # model_config_eval.yaml
    parser.add_argument('-ep1', '--eval_param1', type=float, required=False)
    parser.add_argument('-ep2', '--eval_param2', type=int, required=False)

    args = parser.parse_args()

    logging.info('----------accepting params starts----------')
    logging.info('params1:%s', args.eval_param1)
    logging.info('params2:%s', args.eval_param2)
    logging.info('----------accepting params ends----------')
    logging.info('----------task starts----------')
    test_eval_func(args.output_path)
    logging.info('----------task ends----------')
