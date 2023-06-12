#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import argparse
import logging

logging.getLogger().setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--lr', type=str)
    args = parser.parse_args()
    logging.info(args)


if __name__ == '__main__':
    main()
