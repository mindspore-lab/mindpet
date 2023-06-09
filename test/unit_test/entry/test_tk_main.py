#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging
import os
import stat
import shutil
import unittest
import subprocess
import pytest

FLAG = os.O_RDWR | os.O_CREAT  # 允许读写, 文件不存在时新建
MODE = stat.S_IWUSR | stat.S_IRUSR  # 所有者读写

logging.getLogger().setLevel(logging.INFO)


class TestTkMain(unittest.TestCase):
    root_path = None
    data_path = None
    output_path = None
    pretrained_model_path = None
    ckpt_path = None
    finetune_model_config_path = None
    evaluate_infer_model_config_path = None
    finetune_boot_path = None
    evaluate_boot_path = None
    infer_boot_path = None

    @staticmethod
    def _create_subprocess(cmd):
        """
        使用cmd创建subprocess子进程
        :param cmd: subprocess命令
        :return: subprocess进程对象
        """
        return subprocess.Popen(cmd, env=os.environ, shell=False, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

    @classmethod
    def setUpClass(cls):
        cls.root_path = os.path.join('/', 'tmp', 'test_tk_main')
        if not os.path.exists(cls.root_path):
            os.makedirs(cls.root_path)

        cls.data_path = os.path.join(cls.root_path, 'data_path')
        if not os.path.exists(cls.data_path):
            os.makedirs(cls.data_path)

        cls.output_path = os.path.join(cls.root_path, 'output_path')
        if not os.path.exists(cls.output_path):
            os.makedirs(cls.output_path)

        cls.pretrained_model_path = os.path.join(cls.root_path, 'pretrained_model_path')
        if not os.path.exists(cls.pretrained_model_path):
            os.makedirs(cls.pretrained_model_path)

        cls.ckpt_path = os.path.join(cls.root_path, 'ckpt_path')
        if not os.path.exists(cls.ckpt_path):
            os.makedirs(cls.ckpt_path)

        src_finetune_model_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      'resource/model_config/model_config_finetune.yaml')

        src_evaluate_infer_model_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                            'resource/model_config/model_config_evaluate_infer.yaml')

        src_finetune_boot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              'resource/boot_file/boot_finetune.py')

        src_evaluate_boot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              'resource/boot_file/boot_evaluate.py')

        src_infer_boot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'resource/boot_file/boot_infer.py')

        cls.finetune_model_config_path = os.path.join(cls.root_path, 'model_config_finetune.yaml')
        cls.evaluate_infer_model_config_path = os.path.join(cls.root_path, 'model_config_evaluate_infer.yaml')
        cls.finetune_boot_path = os.path.join(cls.root_path, 'boot_finetune.py')
        cls.evaluate_boot_path = os.path.join(cls.root_path, 'boot_evaluate.py')
        cls.infer_boot_path = os.path.join(cls.root_path, 'boot_infer.py')

        shutil.copyfile(src_finetune_model_config_path, cls.finetune_model_config_path)
        shutil.copyfile(src_evaluate_infer_model_config_path, cls.evaluate_infer_model_config_path)
        shutil.copyfile(src_finetune_boot_path, cls.finetune_boot_path)
        shutil.copyfile(src_evaluate_boot_path, cls.evaluate_boot_path)
        shutil.copyfile(src_infer_boot_path, cls.infer_boot_path)

        cls._chmod(cls.data_path)
        cls._chmod(cls.output_path)
        cls._chmod(cls.pretrained_model_path)
        cls._chmod(cls.ckpt_path)
        cls._chmod(cls.finetune_model_config_path)
        cls._chmod(cls.evaluate_infer_model_config_path)
        cls._chmod(cls.finetune_boot_path)
        cls._chmod(cls.evaluate_boot_path)
        cls._chmod(cls.infer_boot_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.data_path):
            shutil.rmtree(cls.data_path)

        if os.path.exists(cls.output_path):
            shutil.rmtree(cls.output_path)

        if os.path.exists(cls.pretrained_model_path):
            shutil.rmtree(cls.pretrained_model_path)

        if os.path.exists(cls.ckpt_path):
            shutil.rmtree(cls.ckpt_path)

        if os.path.exists(cls.finetune_model_config_path):
            os.remove(cls.finetune_model_config_path)

        if os.path.exists(cls.evaluate_infer_model_config_path):
            os.remove(cls.evaluate_infer_model_config_path)

        if os.path.exists(cls.finetune_boot_path):
            os.remove(cls.finetune_boot_path)

        if os.path.exists(cls.evaluate_boot_path):
            os.remove(cls.evaluate_boot_path)

        if os.path.exists(cls.infer_boot_path):
            os.remove(cls.infer_boot_path)

        if os.path.exists(cls.root_path):
            shutil.rmtree(cls.root_path)

    @classmethod
    def _chmod(cls, path):
        """
        修正path的权限
        :param path:待修正路径
        """
        os.chmod(path, 0o750)

    def test_finetune_cli(self):
        """
        测试CLI侧finetune接口的功能
        """
        logging.info('Start test_finetune_cli.')

        cmd = self._create_finetune_command()
        process = self._create_subprocess(cmd)
        std_out, std_err = process.communicate(timeout=60)

        self.assertIsNone(std_err)
        self.assertIn('Finetune successfully', str(std_out, encoding='utf-8'))

        logging.info('Finish test_finetune_cli.')

    def test_evaluate_cli(self):
        """
        测试CLI侧evaluate接口的功能
        """
        logging.info('Start test_evaluate_cli.')

        cmd = self._create_evaluate_infer_command(task_type='evaluate')
        process = self._create_subprocess(cmd)
        std_out, std_err = process.communicate(timeout=60)

        self.assertIn('Evaluate successfully', str(std_out, encoding='utf-8'))

        logging.info('Finish test_evaluate_cli.')

    def test_infer_cli(self):
        """
        测试CLI侧infer接口的功能
        """
        logging.info('Start test_infer_cli.')

        cmd = self._create_evaluate_infer_command(task_type='infer')
        process = self._create_subprocess(cmd)
        std_out, std_err = process.communicate(timeout=60)

        self.assertIn('Infer successfully', str(std_out, encoding='utf-8'))

        logging.info('Finish test_infer_cli.')

    def test_handle_exception_log_with_click_abort_exception(self):
        """
        测试CLI侧finetune接口抛出click.exception.Abort异常时的功能
        """
        logging.info('Start test_handle_exception_log_with_click_abort_exception.')

        boot_file_path = os.path.join(self.root_path, 'tmp_boot_file.py')
        with os.fdopen(os.open(boot_file_path, FLAG, MODE), 'w') as file:
            file.write('import click; raise click.exceptions.Abort')

        cmd = self._create_finetune_command()
        cmd[cmd.index('--boot_file_path') + 1] = boot_file_path

        process = self._create_subprocess(cmd)
        std_out, _ = process.communicate(timeout=60)

        self.assertIn('Finetune failed', str(std_out, encoding='utf-8'))

        logging.info('Finish test_handle_exception_log_with_click_abort_exception.')

    def test_handle_exception_log_with_click_no_such_option_exception(self):
        """
        测试CLI侧finetune接口抛出click.exception.NoSuchOption异常时的功能
        """
        logging.info('Start test_handle_exception_log_with_click_no_such_option_exception.')

        boot_file_path = os.path.join(self.root_path, 'tmp_boot_file.py')
        with os.fdopen(os.open(boot_file_path, FLAG, MODE), 'w') as file:
            file.write('import click; raise click.exceptions.NoSuchOption')

        cmd = self._create_finetune_command()
        cmd[cmd.index('--boot_file_path') + 1] = boot_file_path

        process = self._create_subprocess(cmd)
        std_out, _ = process.communicate(timeout=60)

        self.assertIn('Finetune failed', str(std_out, encoding='utf-8'))

        logging.info('Finish test_handle_exception_log_with_click_no_such_option_exception.')

    def test_handle_exception_log_with_click_missing_parameter_exception(self):
        """
        测试CLI侧finetune接口抛出click.exception.MissingParameter异常时的功能
        """
        logging.info('Start test_handle_exception_log_with_click_missing_parameter_exception.')

        boot_file_path = os.path.join(self.root_path, 'tmp_boot_file.py')
        with os.fdopen(os.open(boot_file_path, FLAG, MODE), 'w') as file:
            file.write('import click; raise click.exceptions.MissingParameter')

        cmd = self._create_finetune_command()
        cmd[cmd.index('--boot_file_path') + 1] = boot_file_path

        process = self._create_subprocess(cmd)
        std_out, _ = process.communicate(timeout=60)

        self.assertIn('Finetune failed', str(std_out, encoding='utf-8'))

        logging.info('Finish test_handle_exception_log_with_click_missing_parameter_exception.')

    def test_handle_exception_log_with_exception_without_err_msg(self):
        """
        测试CLI侧finetune接口抛出不包含异常信息的Exception时的功能
        """
        logging.info('Start test_handle_exception_log_with_exception_without_err_msg.')

        boot_file_path = os.path.join(self.root_path, 'tmp_boot_file.py')
        with os.fdopen(os.open(boot_file_path, FLAG, MODE), 'w') as file:
            file.write('raise RuntimeError')

        cmd = self._create_finetune_command()
        cmd[cmd.index('--boot_file_path') + 1] = boot_file_path

        process = self._create_subprocess(cmd)
        std_out, _ = process.communicate(timeout=60)

        self.assertIn('Finetune failed', str(std_out, encoding='utf-8'))

        logging.info('Finish test_handle_exception_log_with_exception_without_err_msg.')

    def test_handle_exception_log_with_exception_with_err_msg(self):
        """
        测试CLI侧finetune接口抛出包含异常信息的Exception时的功能
        """
        logging.info('Start test_handle_exception_log_with_exception_with_err_msg.')

        boot_file_path = os.path.join(self.root_path, 'tmp_boot_file.py')
        with os.fdopen(os.open(boot_file_path, FLAG, MODE), 'w') as file:
            file.write('raise RuntimeError(\'runtime error occurred\')')

        cmd = self._create_finetune_command()
        cmd[cmd.index('--boot_file_path') + 1] = boot_file_path

        process = self._create_subprocess(cmd)
        std_out, _ = process.communicate(timeout=60)

        self.assertIn('Finetune failed', str(std_out, encoding='utf-8'))

        logging.info('Finish test_handle_exception_log_with_exception_with_err_msg.')

    def test_finetune_cli_without_quiet(self):
        """
        测试CLI侧非安静模式的finetune接口的功能
        """
        logging.info('Start test_finetune_cli_without_quiet.')

        cmd = ['tk',
               'finetune',
               '--data_path', self.data_path,
               '--output_path', self.output_path,
               '--pretrained_model_path', self.pretrained_model_path,
               '--model_config_path', self.finetune_model_config_path,
               '--boot_file_path', self.finetune_boot_path,
               '--timeout', '1d1h']

        process = self._create_subprocess(cmd)
        std_out, std_err = process.communicate(timeout=60)

        self.assertIsNone(std_err)
        self.assertIn('Finetune successfully', str(std_out, encoding='utf-8'))

        logging.info('Finish test_finetune_cli_without_quiet.')

    def test_finetune_cli_with_quiet_wrong_position(self):
        """
        测试CLI侧finetune接口中--quiet参数在非首位的情况
        """
        logging.info('Start test_finetune_cli_with_quiet_wrong_position.')

        cmd = ['tk',
               'finetune',
               '--data_path', self.data_path,
               '--output_path', self.output_path,
               '--pretrained_model_path', self.pretrained_model_path,
               '--quiet',
               '--model_config_path', self.finetune_model_config_path,
               '--boot_file_path', self.finetune_boot_path,
               '--timeout', '1d1h']

        process = self._create_subprocess(cmd)
        std_out, std_err = process.communicate(timeout=60)

        self.assertIsNone(std_err)
        self.assertIn('Param [--quiet] should be set first.', str(std_out, encoding='utf-8'))

        logging.info('Finish test_finetune_cli_with_quiet_wrong_position.')

    def test_finetune_cli_with_invalid_timeout_hour(self):
        """
        测试CLI侧finetune接口传入--timeout参数中包含不合法小时数
        """
        logging.info('Start test_finetune_cli_with_invalid_timeout_hour.')

        cmd = ['tk',
               'finetune',
               '--quiet',
               '--data_path', self.data_path,
               '--output_path', self.output_path,
               '--pretrained_model_path', self.pretrained_model_path,
               '--model_config_path', self.finetune_model_config_path,
               '--boot_file_path', self.finetune_boot_path,
               '--timeout', '1d30h']

        process = self._create_subprocess(cmd)
        std_out, std_err = process.communicate(timeout=60)

        self.assertIsNone(std_err)
        self.assertIn('Invalid param [timeout].', str(std_out, encoding='utf-8'))

        logging.info('Finish test_finetune_cli_with_invalid_timeout_hour.')

    def test_finetune_cli_with_zero_timeout_hour(self):
        """
        测试CLI侧finetune接口传入--timeout参数值为0的情况
        """
        logging.info('Start test_finetune_cli_with_zero_timeout_hour.')

        cmd = ['tk',
               'finetune',
               '--quiet',
               '--data_path', self.data_path,
               '--output_path', self.output_path,
               '--pretrained_model_path', self.pretrained_model_path,
               '--model_config_path', self.finetune_model_config_path,
               '--boot_file_path', self.finetune_boot_path,
               '--timeout', '0d0h']

        process = self._create_subprocess(cmd)
        std_out, std_err = process.communicate(timeout=60)

        self.assertIsNone(std_err)
        self.assertIn('Invalid param [timeout].', str(std_out, encoding='utf-8'))

        logging.info('Finish test_finetune_cli_with_zero_timeout_hour.')

    def test_finetune_cli_with_timeout_day_limit(self):
        """
        测试CLI侧finetune接口传入--timeout参数值仅包含天数限制的情况
        """
        logging.info('Start test_finetune_cli_with_timeout_day_limit.')

        cmd = ['tk',
               'finetune',
               '--quiet',
               '--data_path', self.data_path,
               '--output_path', self.output_path,
               '--pretrained_model_path', self.pretrained_model_path,
               '--model_config_path', self.finetune_model_config_path,
               '--boot_file_path', self.finetune_boot_path,
               '--timeout', '1d']

        process = self._create_subprocess(cmd)
        std_out, std_err = process.communicate(timeout=60)

        self.assertIsNone(std_err)
        self.assertIn('Finetune successfully.', str(std_out, encoding='utf-8'))

        logging.info('Finish test_finetune_cli_with_timeout_day_limit.')

    def test_finetune_cli_with_timeout_hour_limit(self):
        """
        测试CLI侧finetune接口传入--timeout参数值仅包含小时数限制的情况
        """
        logging.info('Start test_finetune_cli_with_timeout_hour_limit.')

        cmd = ['tk',
               'finetune',
               '--quiet',
               '--data_path', self.data_path,
               '--output_path', self.output_path,
               '--pretrained_model_path', self.pretrained_model_path,
               '--model_config_path', self.finetune_model_config_path,
               '--boot_file_path', self.finetune_boot_path,
               '--timeout', '1h']

        process = self._create_subprocess(cmd)
        std_out, std_err = process.communicate(timeout=60)

        self.assertIsNone(std_err)
        self.assertIn('Finetune successfully.', str(std_out, encoding='utf-8'))

        logging.info('Finish test_finetune_cli_with_timeout_hour_limit.')

    def test_finetune_cli_with_invalid_timeout_format(self):
        """
        测试CLI侧finetune接口传入--timeout参数值不符合约束格式的情况
        """
        logging.info('Start test_finetune_cli_with_invalid_timeout_format.')

        cmd = ['tk',
               'finetune',
               '--quiet',
               '--data_path', self.data_path,
               '--output_path', self.output_path,
               '--pretrained_model_path', self.pretrained_model_path,
               '--model_config_path', self.finetune_model_config_path,
               '--boot_file_path', self.finetune_boot_path,
               '--timeout', 'd1h1']

        process = self._create_subprocess(cmd)
        std_out, std_err = process.communicate(timeout=60)

        self.assertIsNone(std_err)
        self.assertIn('Invalid param [timeout].', str(std_out, encoding='utf-8'))

        logging.info('Finish test_finetune_cli_with_invalid_timeout_format.')

    def _create_finetune_command(self):
        """
        组装finetune cmd指令
        :return: cmd指令
        """
        return ['tk',
                'finetune',
                '--quiet',
                '--data_path', self.data_path,
                '--output_path', self.output_path,
                '--pretrained_model_path', self.pretrained_model_path,
                '--model_config_path', self.finetune_model_config_path,
                '--boot_file_path', self.finetune_boot_path,
                '--timeout', '1d1h']

    def _create_evaluate_infer_command(self, task_type):
        """
        组装evaluate/infer cmd指令
        :return: cmd指令
        """
        return ['tk',
                task_type,
                '--quiet',
                '--data_path', self.data_path,
                '--output_path', self.output_path,
                '--ckpt_path', self.pretrained_model_path,
                '--model_config_path', self.evaluate_infer_model_config_path,
                '--boot_file_path', self.evaluate_boot_path if task_type == 'evaluate' else self.infer_boot_path,
                '--timeout', '1d1h']


if __name__ == '__main__':
    pytest.main(['-s', os.path.abspath(__file__)])
