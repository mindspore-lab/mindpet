#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
import sys
sys.path.append('.')
import os
import json
import shutil
import pathlib
import unittest
import logging
import yaml
import pytest
from mindpet.utils.constants import DEFAULT_FLAGS, DEFAULT_MODES, EMPTY_STRING
from mindpet.utils.io_utils import read_json_file, read_yaml_file
from mindpet.utils.exceptions import LinkPathError

logging.getLogger().setLevel(logging.INFO)


class TestIoUtils(unittest.TestCase):
    root_path = None
    yaml_file_path = None
    json_file_path = None

    @classmethod
    def setUpClass(cls):
        cls.root_path = os.path.join('/', 'tmp', 'test_io_utils')
        if not os.path.exists(cls.root_path):
            os.makedirs(cls.root_path)

        cls.yaml_file_path = os.path.join(cls.root_path, 'yaml_file.yaml')
        with os.fdopen(os.open(cls.yaml_file_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            yaml.dump('yaml file content', file)

        cls.json_file_path = os.path.join(cls.root_path, 'json_file.json')
        with os.fdopen(os.open(cls.json_file_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            json.dump('json file content', file)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.json_file_path):
            os.remove(cls.json_file_path)

        if os.path.exists(cls.yaml_file_path):
            os.remove(cls.yaml_file_path)

        if os.path.exists(cls.root_path):
            shutil.rmtree(cls.root_path)

    def test_read_yaml_file_with_specific_flag_and_mode(self):
        """
        测试调用read_yaml_file方法时传入指定的flag和mode
        """
        logging.info('Start test_read_yaml_file_with_specific_flag_and_mode.')

        content = read_yaml_file(self.yaml_file_path, flags=DEFAULT_FLAGS, modes=DEFAULT_MODES)
        self.assertEqual(content, 'yaml file content')

        logging.info('Finish test_read_yaml_file_with_specific_flag_and_mode.')

    def test_read_yaml_file_with_link_path(self):
        """
        测试读取软链接路径的yaml文件
        """
        logging.info('Start test_read_yaml_file_with_link_path.')

        link_yaml_path = os.path.join(self.root_path, 'link_yaml.yaml')
        os.symlink(self.yaml_file_path, link_yaml_path)

        with self.assertRaises(LinkPathError):
            read_yaml_file(link_yaml_path)

        if os.path.exists(link_yaml_path):
            os.remove(link_yaml_path)

        logging.info('Finish test_read_yaml_file_with_link_path.')

    def test_read_json_file_with_specific_flag_and_mode(self):
        """
        测试调用read_json_file方法时传入指定flag和mode
        """
        logging.info('Start test_read_json_file_with_specific_flag_and_mode.')

        content = read_json_file(self.json_file_path, flags=DEFAULT_FLAGS, modes=DEFAULT_MODES)
        self.assertEqual(content, 'json file content')

        logging.info('Finish test_read_json_file_with_specific_flag_and_mode.')

    def test_read_json_file_with_link_path(self):
        """
        测试读取软链接路径的json文件
        """
        logging.info('Start test_read_json_file_with_link_path.')

        link_json_path = os.path.join(self.root_path, 'link_json.json')
        os.symlink(self.json_file_path, link_json_path)

        with self.assertRaises(LinkPathError):
            read_json_file(link_json_path)

        if os.path.exists(link_json_path):
            os.remove(link_json_path)

        logging.info('Finish test_read_json_file_with_link_path.')

    def test_read_json_file_with_invalid_file_path(self):
        """
        测试读取不存在的json文件
        """
        logging.info('Start test_read_json_file_with_invalid_file_path.')

        temp_json_file = os.path.join(self.root_path, 'temp_json.json')
        if os.path.exists(temp_json_file):
            os.remove(temp_json_file)

        with self.assertRaises(FileNotFoundError):
            read_json_file(temp_json_file)

        logging.info('Finish test_read_json_file_with_invalid_file_path.')

    def test_read_json_file_with_empty_file(self):
        """
        测试读取空json文件
        """
        logging.info('Start test_read_json_file_with_empty_file.')

        temp_json_path = os.path.join(self.root_path, 'temp_json.json')
        pathlib.Path(temp_json_path).touch()

        content = read_json_file(temp_json_path)

        self.assertEqual(content, EMPTY_STRING)

        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)

        logging.info('Finish test_read_json_file_with_empty_file.')


if __name__ == '__main__':
    pytest.main(['-s', os.path.abspath(__file__)])
