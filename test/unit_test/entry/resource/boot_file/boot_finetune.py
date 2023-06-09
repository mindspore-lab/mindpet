#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--pretrained_model_path', type=str)
    parser.add_argument('--learning_rate', type=str)
    parser.add_argument('--batch_size', type=str)
    parser.add_argument('--advanced_config', type=str)
    args = parser.parse_args()

    logging.info(args)


if __name__ == '__main__':
    main()
