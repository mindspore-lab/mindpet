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
    src_result_file_path = os.path.join(output_path, 'src_eval_result.json')

    flag = os.O_RDWR | os.O_CREAT  # 允许读写, 文件不存在时新建
    mode = stat.S_IWUSR | stat.S_IRUSR  # 所有者读写

    with os.fdopen(os.open(src_result_file_path, flag, mode), 'w') as file:
        json.dump(content, file)

    tar_result_file_path = os.path.join(output_path, 'eval_result.json')

    os.symlink(src_result_file_path, tar_result_file_path)


if __name__ == '__main__':
    main()
