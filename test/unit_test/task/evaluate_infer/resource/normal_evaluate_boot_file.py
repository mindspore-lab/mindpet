#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os
import stat
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--lr', type=str)
    args = parser.parse_args()

    content = 'evaluate task success'

    output_path = args.output_path
    result_file_path = os.path.join(output_path, 'eval_result.json')

    flag = os.O_RDWR | os.O_CREAT  # 允许读写, 文件不存在时新建
    mode = stat.S_IWUSR | stat.S_IRUSR  # 所有者读写

    with os.fdopen(os.open(result_file_path, flag, mode), 'w') as file:
        json.dump(content, file)


if __name__ == '__main__':
    main()
