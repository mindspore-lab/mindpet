#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
"""Set up"""
from distutils.core import setup
from setuptools import find_packages

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

cmd_class = {}


def get_version():
    version = '1.0.4'
    return version


def do_setup(packages_data):
    setup(
        name='mindpet',

        version=get_version(),

        description='Parameter-Efficient Tuning',

        keywords='Parameter-Efficient Tuning',

        # 详细的程序描述
        long_description=readme,
        long_description_content_type='text/markdown',

        # 编包依赖
        setup_requires=[
            'python_version>=3.7', 'setuptools>=61.2.0', 'pyyaml>=6.0', 'wheel'
        ],

        # 项目依赖的Python库
        install_requires=[
            'click', 'pyyaml'
        ],

        # 需要打包的目录列表
        packages=find_packages(
            # 不需要打包的目录列表
            exclude=[
                'test', 'test.*'
            ]
        ),

        package_data=packages_data,
        entry_points={
            'console_scripts': [
                'mindpet = mindpet.mindpet_main:cli_wrapper'
            ],
        },
        cmdclass=cmd_class
    )


if __name__ == '__main__':
    package_data = {
        'mindpet': [
            '*.py',
            '*/*.py',
            '*/*/*.py',
            '*/*/*/*.py',
            '*/*/*/*/*.py'
        ]
    }
    do_setup(package_data)
