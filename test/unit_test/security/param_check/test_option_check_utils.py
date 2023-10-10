#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
import sys
sys.path.append('.')
import os
import shutil
import unittest
import logging
import pytest
from mindpet.utils.entrance_monitor import entrance_monitor
from mindpet.utils.constants import ENTRANCE_TYPE, DEFAULT_MODES, DEFAULT_FLAGS, BLACKLIST_CHARACTERS, GB_SIZE
from mindpet.utils.exceptions import FileOversizeError, LinkPathError, AbsolutePathError, PathContentError, \
    FileNameLengthError, PathLengthError, PathGranularityError, PathRightEscalationError, LowDiskFreeSizeRiskError
from mindpet.security.param_check.option_check_utils import OptionBase, PathContentBlacklistCharactersCheck, \
    AbsolutePathCheck, PathExistCheck, LinkPathCheck, PathContentLengthCheck, PathContentCharacterCheck, \
    PathGranularityCheck, PathRightEscalationCheck, FileSizeCheck, InteractionByEntrance, DiskFreeSpaceCheck, \
    get_real_path

logging.getLogger().setLevel(logging.INFO)

LOCAL_PATH = os.path.join('/', 'tmp', 'ut').replace('\\', '/')
LOCAL_FILE = os.path.join(LOCAL_PATH, 'ut_sample.txt')
EMPTY_STRING = ''


class OptionCheckError(Exception):
    def __init__(self, error_info=None):
        super(OptionCheckError, self).__init__()
        self.error_info = error_info

    def __str__(self):
        return self.error_info if self.error_info else EMPTY_STRING


class TestOptionCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(LOCAL_PATH):
            os.makedirs(LOCAL_PATH, exist_ok=True)

        if not os.path.exists(LOCAL_FILE):
            with os.fdopen(os.open(LOCAL_FILE, DEFAULT_FLAGS, DEFAULT_MODES), 'w+') as file:
                file.write('ut test sample')

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(LOCAL_PATH):
            shutil.rmtree(LOCAL_PATH)

    def test_option_base_init_with_none_option_name(self):
        """
        测试OptionBase类构造方法中, 参数option_name传None的情况
        """
        logging.info('Start test_option_base_init_with_none_option_name.')
        with self.assertRaises(ValueError):
            OptionBase(option_name=None, option_value='option_value')
        logging.info('Finish test_option_base_init_with_none_option_name finish.')

    def test_option_base_init_with_empty_option_name(self):
        """
        测试OptionBase类构造方法中, 参数option_name传空字符串的情况
        """
        logging.info('Start test_option_base_init_with_empty_option_name.')
        with self.assertRaises(ValueError):
            OptionBase(option_name=EMPTY_STRING, option_value='option_value')
        logging.info('Finish test_option_base_init_with_empty_option_name.')

    def test_option_base_init_with_none_option_value(self):
        """
        测试OptionBase类构造方法中, 参数option_value传None的情况
        """
        logging.info('Start test_option_base_init_with_none_option_value.')
        with self.assertRaises(ValueError):
            OptionBase(option_name='option_name', option_value=None)
        logging.info('Finish test_option_base_init_with_none_option_value.')

    def test_option_base_init_with_empty_option_value(self):
        """
        测试OptionBase类构造方法中, 参数option_value传空字符串的情况
        """
        logging.info('Start test_option_base_init_with_empty_option_value.')
        with self.assertRaises(ValueError):
            OptionBase(option_name='option_name', option_value=EMPTY_STRING)
        logging.info('Finish test_option_base_init_with_empty_option_value.')

    def test_option_base_init(self):
        """
        测试OptionBase类构造方法正常构建过程
        """
        logging.info('Start test_option_base_init.')
        option = OptionBase(option_name='option_name', option_value='option_value')
        self.assertEqual(option.option_name, 'option_name')
        self.assertEqual(option.option_value, 'option_value')
        logging.info('Finish test_option_base_init.')

    def test_interaction_by_entrance_with_none_notice_msg(self):
        """
        测试InteractionByEntrance类中, 参数notice_msg传None值的情况
        """
        logging.info('Start test_interaction_by_entrance_with_none_notice_msg.')
        with self.assertRaises(ValueError):
            InteractionByEntrance(option_name='option_name',
                                  option_value='option_value',
                                  notice_msg=None,
                                  notice_accept_msg='notice_accept_msg',
                                  exception_type=OptionCheckError)
        logging.info('Finish test_interaction_by_entrance_with_none_notice_msg.')

    def test_interaction_by_entrance_with_empty_notice_msg(self):
        """
        测试InteractionByEntrance类中, 参数notice_msg传空字符串的情况
        """
        logging.info('Start test_interaction_by_entrance_with_empty_notice_msg.')
        with self.assertRaises(ValueError):
            InteractionByEntrance(option_name='option_name',
                                  option_value='option_value',
                                  notice_msg=EMPTY_STRING,
                                  notice_accept_msg='notice_accept_msg',
                                  exception_type=OptionCheckError)
        logging.info('Finish test_interaction_by_entrance_with_empty_notice_msg.')

    def test_interaction_by_entrance_with_invalid_notice_msg_type(self):
        """
        测试InteractionByEntrance类中, 参数notice_msg传不合理类型值的情况
        """
        logging.info('Start test_interaction_by_entrance_with_invalid_notice_msg_type.')
        with self.assertRaises(TypeError):
            InteractionByEntrance(option_name='option_name',
                                  option_value='option_value',
                                  notice_msg=True,
                                  notice_accept_msg='notice_accept_msg',
                                  exception_type=OptionCheckError)
        logging.info('Finish test_interaction_by_entrance_with_invalid_notice_msg_type.')

    def test_interaction_by_entrance_with_none_notice_accept_msg(self):
        """
        测试InteractionByEntrance类中, 参数notice_accept_msg传None值的情况
        """
        logging.info('Start test_interaction_by_entrance_with_none_notice_accept_msg.')
        with self.assertRaises(ValueError):
            InteractionByEntrance(option_name='option_name',
                                  option_value='option_value',
                                  notice_msg='notice_msg',
                                  notice_accept_msg=None,
                                  exception_type=OptionCheckError)
        logging.info('Finish test_interaction_by_entrance_with_none_notice_accept_msg.')

    def test_interaction_by_entrance_with_empty_notice_accept_msg(self):
        """
        测试InteractionByEntrance类中, 参数notice_accept_msg传空字符串的情况
        """
        logging.info('Start test_interaction_by_entrance_with_empty_notice_accept_msg.')
        with self.assertRaises(ValueError):
            InteractionByEntrance(option_name='option_name',
                                  option_value='option_value',
                                  notice_msg='notice_msg',
                                  notice_accept_msg=EMPTY_STRING,
                                  exception_type=OptionCheckError)
        logging.info('Finish test_interaction_by_entrance_with_empty_notice_accept_msg.')

    def test_interaction_by_entrance_with_invalid_notice_accept_msg_type(self):
        """
        测试InteractionByEntrance类中, 参数notice_accept_msg传不合理类型值的情况
        """
        logging.info('Start test_interaction_by_entrance_with_invalid_notice_accept_msg_type.')
        with self.assertRaises(TypeError):
            InteractionByEntrance(option_name='option_name',
                                  option_value='option_value',
                                  notice_msg='notice_msg',
                                  notice_accept_msg=True,
                                  exception_type=OptionCheckError)
        logging.info('Finish test_interaction_by_entrance_with_invalid_notice_accept_msg_type.')

    def test_interaction_by_entrance_with_none_exception_type(self):
        """
        测试InteractionByEntrance类中, 参数exception_type传None值的情况
        """
        logging.info('Start test_interaction_by_entrance_with_none_exception_type.')
        with self.assertRaises(ValueError):
            InteractionByEntrance(option_name='option_name',
                                  option_value='option_value',
                                  notice_msg='notice_msg',
                                  notice_accept_msg='notice_accept_msg',
                                  exception_type=None)
        logging.info('Finish test_interaction_by_entrance_with_none_exception_type.')

    def test_interaction_by_entrance_with_invalid_exception_type(self):
        """
        测试InteractionByEntrance类中, 参数exception_type传不合理异常类型值的情况
        """
        logging.info('Start test_interaction_by_entrance_with_invalid_exception_type.')
        with self.assertRaises(TypeError):
            InteractionByEntrance(option_name='option_name',
                                  option_value='option_value',
                                  notice_msg='notice_msg',
                                  notice_accept_msg='notice_accept_msg',
                                  exception_type=SystemExit)
        logging.info('Finish test_interaction_by_entrance_with_invalid_exception_type.')

    def test_interaction_by_entrance_with_none_force_quit(self):
        """
        测试InteractionByEntrance类中, 参数force_quit传None值的情况
        """
        logging.info('Start test_interaction_by_entrance_with_none_force_quit.')
        interaction_op = InteractionByEntrance(option_name='option_name',
                                               option_value='option_value',
                                               notice_msg='notice_msg',
                                               notice_accept_msg='notice_accept_msg',
                                               exception_type=OptionCheckError)
        with self.assertRaises(ValueError):
            interaction_op.interact_by_entrance(force_quit=None, quiet=False)
        logging.info('Finish test_interaction_by_entrance_with_none_force_quit.')

    def test_interaction_by_entrance_with_invalid_force_quit_type(self):
        """
        测试InteractionByEntrance类中, 参数force_quit传不合理类型值的情况
        """
        logging.info('Start test_interaction_by_entrance_with_invalid_force_quit_type.')
        interaction_op = InteractionByEntrance(option_name='option_name',
                                               option_value='option_value',
                                               notice_msg='notice_msg',
                                               notice_accept_msg='notice_accept_msg',
                                               exception_type=OptionCheckError)
        with self.assertRaises(TypeError):
            interaction_op.interact_by_entrance(force_quit='True', quiet=False)
        logging.info('Finish test_interaction_by_entrance_with_invalid_force_quit_type.')

    def test_interaction_by_entrance_with_none_quiet(self):
        """
        测试InteractionByEntrance类中, 参数quiet传None值的情况
        """
        logging.info('Start test_interaction_by_entrance_with_none_quiet.')
        interaction_op = InteractionByEntrance(option_name='option_name',
                                               option_value='option_value',
                                               notice_msg='notice_msg',
                                               notice_accept_msg='notice_accept_msg',
                                               exception_type=OptionCheckError)
        with self.assertRaises(ValueError):
            interaction_op.interact_by_entrance(force_quit=True, quiet=None)
        logging.info('Finish test_interaction_by_entrance_with_none_quiet.')

    def test_interaction_by_entrance_with_invalid_quiet_type(self):
        """
        测试InteractionByEntrance类中, 参数quiet传不合理类型值的情况
        """
        logging.info('Start test_interaction_by_entrance_with_invalid_quiet_type.')
        interaction_op = InteractionByEntrance(option_name='option_name',
                                               option_value='option_value',
                                               notice_msg='notice_msg',
                                               notice_accept_msg='notice_accept_msg',
                                               exception_type=OptionCheckError)
        with self.assertRaises(TypeError):
            interaction_op.interact_by_entrance(force_quit=True, quiet='False')
        logging.info('Finish test_interaction_by_entrance_with_invalid_quiet_type.')

    def test_interaction_by_entrance_with_force_quit(self):
        """
        测试InteractionByEntrance类中, 参数force_quit传True的情况
        """
        logging.info('Start test_interaction_by_entrance_with_force_quit.')
        interaction_op = InteractionByEntrance(option_name='option_name',
                                               option_value='option_value',
                                               notice_msg='notice_msg',
                                               notice_accept_msg='notice_accept_msg',
                                               exception_type=OptionCheckError)
        with self.assertRaises(OptionCheckError):
            interaction_op.interact_by_entrance(force_quit=True, quiet=True)
        logging.info('Finish test_interaction_by_entrance_with_force_quit.')

    def test_interaction_by_entrance_with_cli_entry(self):
        """
        测试InteractionByEntrance类中, 在CLI侧场景安静模式下force_quit为True的情况
        """
        logging.info('Start test_interaction_by_entrance_with_cli_entry.')
        entrance_monitor.set_value(ENTRANCE_TYPE, 'CLI')
        interaction_op = InteractionByEntrance(option_name='option_name',
                                               option_value='option_value',
                                               notice_msg='notice_msg',
                                               notice_accept_msg='notice_accept_msg',
                                               exception_type=OptionCheckError)
        with self.assertRaises(OptionCheckError):
            interaction_op.interact_by_entrance(force_quit=True, quiet=True)
        logging.info('Finish test_interaction_by_entrance_with_cli_entry.')

    def test_interaction_by_entrance_with_sdk_entry(self):
        """
        测试InteractionByEntrance类中, 在SDK侧场景非安静模式下force_quit为True的情况
        """
        logging.info('Start test_interaction_by_entrance_with_sdk_entry.')
        entrance_monitor.set_value(ENTRANCE_TYPE, 'SDK')
        interaction_op = InteractionByEntrance(option_name='option_name',
                                               option_value='option_value',
                                               notice_msg='notice_msg',
                                               notice_accept_msg='notice_accept_msg',
                                               exception_type=OptionCheckError)
        with self.assertRaises(OptionCheckError):
            interaction_op.interact_by_entrance(force_quit=True, quiet=False)
        logging.info('Finish test_interaction_by_entrance_with_sdk_entry.')

    def test_interaction_by_entrance_with_invalid_entrance_type(self):
        """
        测试InteractionByEntrance类中, 参数entrance_type传不合理类型值的情况
        """
        logging.info('Start test_interaction_by_entrance_with_invalid_entrance_type.')
        entrance_monitor.set_value(ENTRANCE_TYPE, 'OTHERS')
        interaction_op = InteractionByEntrance(option_name='option_name',
                                               option_value='option_value',
                                               notice_msg='notice_msg',
                                               notice_accept_msg='notice_accept_msg',
                                               exception_type=OptionCheckError)
        with self.assertRaises(ValueError):
            interaction_op.interact_by_entrance(force_quit=True, quiet=True)
        logging.info('Finish test_interaction_by_entrance_with_invalid_entrance_type.')

    def test_interaction_by_entrance_in_cli_with_quiet_and_not_force_quit(self):
        """
        测试InteractionByEntrance类中, 在CLI侧安静模式下force_quit为False的情况
        """
        logging.info('Start test_interaction_by_entrance_in_cli_with_quiet_and_not_force_quit.')
        entrance_monitor.set_value(ENTRANCE_TYPE, 'CLI')
        interaction_op = InteractionByEntrance(option_name='option_name',
                                               option_value='option_value',
                                               notice_msg='notice_msg',
                                               notice_accept_msg='notice_accept_msg',
                                               exception_type=OptionCheckError)
        interaction_op.interact_by_entrance(force_quit=False, quiet=True)
        logging.info('Finish test_interaction_by_entrance_in_cli_with_quiet_and_not_force_quit.')

    def test_interaction_by_entrance_in_sdk_without_force_quit(self):
        """
        测试InteractionByEntrance类中, 在SDK侧安静模式下force_quit为False的情况
        """
        logging.info('Start test_interaction_by_entrance_in_sdk_without_force_quit.')
        entrance_monitor.set_value(ENTRANCE_TYPE, 'SDK')
        interaction_op = InteractionByEntrance(option_name='option_name',
                                               option_value='option_value',
                                               notice_msg='notice_msg',
                                               notice_accept_msg='notice_accept_msg',
                                               exception_type=OptionCheckError)
        interaction_op.interact_by_entrance(force_quit=False, quiet=True)
        logging.info('Finish test_interaction_by_entrance_in_sdk_without_force_quit.')

    def test_path_content_blacklist_characters_check(self):
        """
        测试路径内容黑名单字符校验
        """
        logging.info('Start test_path_content_blacklist_characters_check.')

        path = os.path.join('/', '$tmp', '$ut_sample.txt')
        for black_char in BLACKLIST_CHARACTERS:
            tmp_path = path.replace('$', black_char)
            logging.info('Current temp path: %s', tmp_path)

            with self.assertRaises(PathContentError):
                PathContentBlacklistCharactersCheck(option_name='path_content_blacklist_characters_check',
                                                    option_value=tmp_path)
        logging.info('Finish test_path_content_blacklist_characters_check.')

    def test_absolute_path_check(self):
        """
        测试绝对路径校验
        """
        logging.info('Start test_absolute_path_check.')
        path = os.path.join('./', 'tmp', 'tmp_file.txt')
        with self.assertRaises(AbsolutePathError):
            AbsolutePathCheck(option_name='absolute_path_check', option_value=path)
        logging.info('Finish test_absolute_path_check.')

    def test_path_exist_check(self):
        """
        测试路径存在性校验
        """
        logging.info('Start test_path_exist_check.')
        path = os.path.join('/', 'tmp', 'unknown_dir')
        with self.assertRaises(FileNotFoundError):
            PathExistCheck(option_name='path_exist_check', option_value=path)
        logging.info('Finish test_path_exist_check.')

    def test_link_path_check(self):
        """
        测试软链接路径校验
        """
        logging.info('Start test_link_path_check.')
        link_path = LOCAL_PATH + 'link_file.txt'
        os.symlink(LOCAL_FILE, link_path)

        with self.assertRaises(LinkPathError):
            LinkPathCheck(option_name='link_path_check', option_value=link_path)

        if os.path.exists(link_path):
            os.remove(link_path)
        logging.info('Finish test_link_path_check.')

    def test_path_content_length_check_with_invalid_path_min_limit_type(self):
        """
        测试路径内容长度校验, 参数path_min_limit传不合理类型值的情况
        """
        logging.info('Start test_path_content_length_check_with_invalid_path_min_limit_type.')

        path = os.path.join(LOCAL_PATH, 'path_length_check')

        if not os.path.exists(path):
            os.makedirs(path)

        with self.assertRaises(TypeError):
            PathContentLengthCheck(option_name='path_length_check',
                                   option_value=path,
                                   path_min_limit='100',
                                   path_max_limit=1024,
                                   file_min_limit=None,
                                   file_max_limit=None)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_length_check_with_invalid_path_min_limit_type.')

    def test_path_content_length_check_with_invalid_path_max_limit_type(self):
        """
        测试路径内容长度校验, 参数path_max_limit传不合理类型值的情况
        """
        logging.info('Start test_path_content_length_check_with_invalid_path_max_limit_type.')

        path = os.path.join(LOCAL_PATH, 'path_length_check')

        if not os.path.exists(path):
            os.makedirs(path)

        with self.assertRaises(TypeError):
            PathContentLengthCheck(option_name='path_length_check',
                                   option_value=path,
                                   path_min_limit=100,
                                   path_max_limit='1024',
                                   file_min_limit=None,
                                   file_max_limit=None)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_length_check_with_invalid_path_max_limit_type.')

    def test_path_content_length_check_with_invalid_file_min_limit_type(self):
        """
        测试路径内容长度校验, 参数file_min_limit传不合理类型值的情况
        """
        logging.info('Start test_path_content_length_check_with_invalid_file_min_limit_type.')

        path = os.path.join(LOCAL_PATH, 'path_length_check')

        if not os.path.exists(path):
            os.makedirs(path)

        file_name = 'check_file.txt'
        full_path = os.path.join(path, file_name)

        with os.fdopen(os.open(full_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('check_content')

        with self.assertRaises(TypeError):
            PathContentLengthCheck(option_name='path_length_check',
                                   option_value=full_path,
                                   path_min_limit=100,
                                   path_max_limit=1024,
                                   file_min_limit='100',
                                   file_max_limit=1024)

        if os.path.exists(full_path):
            os.remove(full_path)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_length_check_with_invalid_file_min_limit_type.')

    def test_path_content_length_check_with_invalid_file_max_limit_type(self):
        """
        测试路径内容长度校验, 参数file_max_limit传不合理类型值的情况
        """
        logging.info('Start test_path_content_length_check_with_invalid_file_max_limit_type.')

        path = os.path.join(LOCAL_PATH, 'path_length_check')

        if not os.path.exists(path):
            os.makedirs(path)

        file_name = 'check_file.txt'
        full_path = os.path.join(path, file_name)

        with os.fdopen(os.open(full_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('check_content')

        with self.assertRaises(TypeError):
            PathContentLengthCheck(option_name='path_length_check',
                                   option_value=full_path,
                                   path_min_limit=100,
                                   path_max_limit=1024,
                                   file_min_limit=100,
                                   file_max_limit='1024')

        if os.path.exists(full_path):
            os.remove(full_path)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_length_check_with_invalid_file_max_limit_type.')

    def test_path_content_length_check_with_negative_path_min_limit(self):
        """
        测试路径内容长度校验, 参数path_min_limit传负数值的情况
        """
        logging.info('Start test_path_content_length_check_with_negative_path_min_limit.')
        path = os.path.join(LOCAL_PATH, 'path_length_check')

        if not os.path.exists(path):
            os.makedirs(path)

        with self.assertRaises(PathLengthError):
            PathContentLengthCheck(option_name='path_length_check',
                                   option_value=path,
                                   path_min_limit=-1,
                                   path_max_limit=1024,
                                   file_min_limit=None,
                                   file_max_limit=None)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_length_check_with_negative_path_min_limit.')

    def test_path_content_length_check_with_negative_path_max_limit(self):
        """
        测试路径内容长度校验, 参数path_max_limit传负数值的情况
        """
        logging.info('Start test_path_content_length_check_with_negative_path_max_limit.')
        path = os.path.join(LOCAL_PATH, 'path_length_check')

        if not os.path.exists(path):
            os.makedirs(path)

        with self.assertRaises(PathLengthError):
            PathContentLengthCheck(option_name='path_length_check',
                                   option_value=path,
                                   path_min_limit=100,
                                   path_max_limit=-1,
                                   file_min_limit=None,
                                   file_max_limit=None)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_length_check_with_negative_path_max_limit.')

    def test_path_content_length_check_with_negative_file_min_limit(self):
        """
        测试路径内容长度校验, 参数file_min_limit传负数值的情况
        """
        logging.info('Start test_path_content_length_check_with_negative_file_min_limit.')

        path = os.path.join(LOCAL_PATH, 'path_length_check')

        if not os.path.exists(path):
            os.makedirs(path)

        file_name = 'check_file.txt'
        full_path = os.path.join(path, file_name)

        with os.fdopen(os.open(full_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('check_content')

        with self.assertRaises(FileNameLengthError):
            PathContentLengthCheck(option_name='path_length_check',
                                   option_value=full_path,
                                   path_min_limit=1,
                                   path_max_limit=1024,
                                   file_min_limit=-1,
                                   file_max_limit=1024)

        if os.path.exists(full_path):
            os.remove(full_path)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_length_check_with_negative_file_min_limit.')

    def test_path_content_length_check_with_negative_file_max_limit(self):
        """
        测试路径内容长度校验, 参数file_max_limit传负数值的情况
        """
        logging.info('Start test_path_content_length_check_with_negative_file_max_limit.')

        path = os.path.join(LOCAL_PATH, 'path_length_check')

        if not os.path.exists(path):
            os.makedirs(path)

        file_name = 'check_file.txt'
        full_path = os.path.join(path, file_name)

        with os.fdopen(os.open(full_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('check_content')

        with self.assertRaises(FileNameLengthError):
            PathContentLengthCheck(option_name='path_length_check',
                                   option_value=full_path,
                                   path_min_limit=1,
                                   path_max_limit=1024,
                                   file_min_limit=100,
                                   file_max_limit=-1)

        if os.path.exists(full_path):
            os.remove(full_path)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_length_check_with_negative_file_max_limit.')

    def test_path_content_length_check_with_inverse_path_min_max_limit(self):
        """
        测试路径内容长度校验, path_min_limit比path_max_limit还要大的颠倒情况
        """
        logging.info('Start test_path_content_length_check_with_inverse_path_min_max_limit.')
        path = os.path.join(LOCAL_PATH, 'path_length_check')

        if not os.path.exists(path):
            os.makedirs(path)

        with self.assertRaises(PathLengthError):
            PathContentLengthCheck(option_name='path_length_check',
                                   option_value=path,
                                   path_min_limit=1024,
                                   path_max_limit=100,
                                   file_min_limit=None,
                                   file_max_limit=None)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_length_check_with_inverse_path_min_max_limit.')

    def test_path_content_length_check_with_short_path(self):
        """
        测试路径内容长度校验, 路径长度过短的情况
        """
        logging.info('Start test_path_content_length_check_with_short_path.')
        path = os.path.join(LOCAL_PATH, 'path_length_check')

        if not os.path.exists(path):
            os.makedirs(path)

        with self.assertRaises(PathLengthError):
            PathContentLengthCheck(option_name='path_length_check',
                                   option_value=path,
                                   path_min_limit=100,
                                   path_max_limit=1024,
                                   file_min_limit=None,
                                   file_max_limit=None)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_length_check_with_short_path.')

    def test_path_content_length_check_with_long_path(self):
        """
        测试路径内容长度校验, 路径长度过长的情况
        """
        logging.info('Start test_path_content_length_check_with_long_path.')
        path = os.path.join(LOCAL_PATH, 'path_length_check')

        if not os.path.exists(path):
            os.makedirs(path)

        with self.assertRaises(PathLengthError):
            PathContentLengthCheck(option_name='path_length_check',
                                   option_value=path,
                                   path_min_limit=1,
                                   path_max_limit=10,
                                   file_min_limit=None,
                                   file_max_limit=None)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_length_check_with_long_path.')

    def test_path_content_length_check_with_inverse_file_min_max_limit(self):
        """
        测试路径内容长度校验, file_min_limit比file_max_limit还要大的颠倒情况
        """
        logging.info('Start test_path_content_length_check_with_inverse_file_min_max_limit.')

        path = os.path.join(LOCAL_PATH, 'path_length_check')

        if not os.path.exists(path):
            os.makedirs(path)

        file_name = 'check_file.txt'
        full_path = os.path.join(path, file_name)

        with os.fdopen(os.open(full_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('check_content')

        with self.assertRaises(FileNameLengthError):
            PathContentLengthCheck(option_name='path_length_check',
                                   option_value=full_path,
                                   path_min_limit=1,
                                   path_max_limit=1024,
                                   file_min_limit=1024,
                                   file_max_limit=100)

        if os.path.exists(full_path):
            os.remove(full_path)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_length_check_with_inverse_file_min_max_limit.')

    def test_path_content_length_check_with_short_file_name(self):
        """
        测试路径内容长度校验, 文件名长度过短的情况
        """
        logging.info('Start test_path_content_length_check_with_short_file_name.')

        path = os.path.join(LOCAL_PATH, 'path_length_check')

        if not os.path.exists(path):
            os.makedirs(path)

        file_name = 'check_file.txt'
        full_path = os.path.join(path, file_name)

        with os.fdopen(os.open(full_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('check_content')

        with self.assertRaises(FileNameLengthError):
            PathContentLengthCheck(option_name='path_length_check',
                                   option_value=full_path,
                                   path_min_limit=1,
                                   path_max_limit=1024,
                                   file_min_limit=100,
                                   file_max_limit=1024)

        if os.path.exists(full_path):
            os.remove(full_path)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_length_check_with_short_file_name.')

    def test_path_content_length_check_with_long_file_name(self):
        """
        测试路径内容长度校验, 文件名长度过长的情况
        """
        logging.info('Start test_path_content_length_check_with_long_file_name.')

        path = os.path.join(LOCAL_PATH, 'path_length_check')

        if not os.path.exists(path):
            os.makedirs(path)

        file_name = 'check_file.txt'
        full_path = os.path.join(path, file_name)

        with os.fdopen(os.open(full_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('check_content')

        with self.assertRaises(FileNameLengthError):
            PathContentLengthCheck(option_name='path_length_check',
                                   option_value=full_path,
                                   path_min_limit=1,
                                   path_max_limit=1024,
                                   file_min_limit=1,
                                   file_max_limit=10)

        if os.path.exists(full_path):
            os.remove(full_path)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_length_check_with_long_file_name.')

    def test_path_content_character_check_with_none_base_whitelist_mode(self):
        """
        测试路径内容字符白名单校验, 入参base_whitelist_mode传None值的情况
        """
        logging.info('Start test_path_content_character_check_with_none_base_whitelist_mode.')

        path = os.path.join(LOCAL_PATH, 'path_content_character_check')
        if not os.path.exists(path):
            os.makedirs(path)

        with self.assertRaises(ValueError):
            PathContentCharacterCheck(option_name='path_content_character_check',
                                      option_value=path,
                                      base_whitelist_mode=None,
                                      extra_whitelist=['/', '_'])

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_character_check_with_none_base_whitelist_mode.')

    def test_path_content_character_check_with_empty_base_whitelist_mode(self):
        """
        测试路径内容字符白名单校验, 入参base_whitelist_mode传空字符串的情况
        """
        logging.info('Start test_path_content_character_check_with_none_base_whitelist_mode.')

        path = os.path.join(LOCAL_PATH, 'path_content_character_check')
        if not os.path.exists(path):
            os.makedirs(path)

        with self.assertRaises(ValueError):
            PathContentCharacterCheck(option_name='path_content_character_check',
                                      option_value=path,
                                      base_whitelist_mode=EMPTY_STRING,
                                      extra_whitelist=['/', '_'])

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_character_check_with_none_base_whitelist_mode.')

    def test_path_content_character_check_with_invalid_base_whitelist_mode_type(self):
        """
        测试路径内容字符白名单校验, 入参base_whitelist_mode传不合理类型值的情况
        """
        logging.info('Start test_path_content_character_check_with_invalid_base_whitelist_mode_type.')

        path = os.path.join(LOCAL_PATH, 'path_content_character_check')
        if not os.path.exists(path):
            os.makedirs(path)

        with self.assertRaises(TypeError):
            PathContentCharacterCheck(option_name='path_content_character_check',
                                      option_value=path,
                                      base_whitelist_mode=True,
                                      extra_whitelist=['/', '_'])

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_character_check_with_invalid_base_whitelist_mode_type.')

    def test_path_content_character_check_with_invalid_extra_whitelist_type(self):
        """
        测试路径内容字符白名单校验, 入参extra_whitelist_type传不合理类型值的情况
        """
        logging.info('Start test_path_content_character_check_with_invalid_extra_whitelist_type.')

        path = os.path.join(LOCAL_PATH, 'path_content_character_check')
        if not os.path.exists(path):
            os.makedirs(path)

        with self.assertRaises(TypeError):
            PathContentCharacterCheck(option_name='path_content_character_check',
                                      option_value=path,
                                      base_whitelist_mode='ALL',
                                      extra_whitelist=('/', '_'))

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_character_check_with_invalid_extra_whitelist_type.')

    def test_path_content_character_check_with_invalid_base_whitelist_mode(self):
        """
        测试路径内容字符白名单校验, 入参base_whitelist_mode传不合理值的情况
        """
        logging.info('Start test_path_content_character_check_with_invalid_base_whitelist_mode.')

        path = os.path.join(LOCAL_PATH, 'path_content_character_check')
        if not os.path.exists(path):
            os.makedirs(path)

        with self.assertRaises(ValueError):
            PathContentCharacterCheck(option_name='path_content_character_check',
                                      option_value=path,
                                      base_whitelist_mode='EXTRA',
                                      extra_whitelist=['/', '_'])

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_content_character_check_with_invalid_base_whitelist_mode.')

    def test_path_content_character_check(self):
        """
        测试路径内容字符白名单校验
        """
        logging.info('Start test_path_content_character_check.')

        path = os.path.join(LOCAL_PATH, 'path_content_character_check')
        if not os.path.exists(path):
            os.makedirs(path)

        with self.assertRaises(PathContentError):
            PathContentCharacterCheck(option_name='path_content_character_check',
                                      option_value=path,
                                      base_whitelist_mode='ALL',
                                      extra_whitelist=['_'])

        logging.info('Finish test_path_content_character_check.')

    def test_path_granularity_check_with_invalid_path_including_file_type(self):
        """
        测试路径粒度校验, 参数path_including_file传不合理类型值的情况
        """
        logging.info('Start test_path_granularity_check_with_invalid_path_including_file_type.')

        path = os.path.join(LOCAL_PATH, 'path_granularity_check')
        if not os.path.exists(path):
            os.makedirs(path)

        with self.assertRaises(TypeError):
            PathGranularityCheck(option_name='path_granularity_check',
                                 option_value=path,
                                 path_including_file='True')

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_granularity_check_with_invalid_path_including_file_type.')

    def test_path_granularity_check_with_none_path_including_file(self):
        """
        测试路径粒度校验, 参数path_including_file传None值的情况
        """
        logging.info('Start test_path_granularity_check_with_none_path_including_file.')

        path = os.path.join(LOCAL_PATH, 'path_granularity_check')
        if not os.path.exists(path):
            os.makedirs(path)

        PathGranularityCheck(option_name='path_granularity_check',
                             option_value=path,
                             path_including_file=None)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_granularity_check_with_none_path_including_file.')

    def test_path_granularity_check_with_path(self):
        """
        测试路径粒度校验, 待测试参数为路径(文件夹)类型的情况
        """
        logging.info('Start test_path_granularity_check_with_path.')

        path = os.path.join(LOCAL_PATH, 'path_granularity_check')
        if not os.path.exists(path):
            os.makedirs(path)

        with self.assertRaises(PathGranularityError):
            PathGranularityCheck(option_name='path_granularity_check',
                                 option_value=path,
                                 path_including_file=True)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_granularity_check_with_path.')

    def test_path_granularity_check_with_file(self):
        """
        测试路径粒度校验, 待测试参数为文件类型的情况
        """
        logging.info('Start test_path_granularity_check_with_file.')
        with self.assertRaises(PathGranularityError):
            PathGranularityCheck(option_name='path_granularity_check',
                                 option_value=LOCAL_FILE,
                                 path_including_file=False)
        logging.info('Finish test_path_granularity_check_with_file.')

    def test_path_right_escalation_check_with_none_mode(self):
        """
        测试路径权限提升校验, 参数mode传None值的情况
        """
        logging.info('Start test_path_right_escalation_check_with_none_mode.')
        path = os.path.join(LOCAL_PATH, 'path_right_escalation_check')
        if not os.path.exists(path):
            os.makedirs(path)

        os.chmod(path, 0o750)
        entrance_monitor.set_value(ENTRANCE_TYPE, 'SDK')

        with self.assertRaises(ValueError):
            PathRightEscalationCheck(option_name='path_right_escalation_check',
                                     option_value=path,
                                     mode=None,
                                     force_quit=True,
                                     quiet=False)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_right_escalation_check_with_none_mode.')

    def test_path_right_escalation_check_with_invalid_mode_type(self):
        """
        测试路径权限提升校验, 参数mode传不合理类型值的情况
        """
        logging.info('Start test_path_right_escalation_check_with_invalid_mode_type.')
        path = os.path.join(LOCAL_PATH, 'path_right_escalation_check')
        if not os.path.exists(path):
            os.makedirs(path)

        os.chmod(path, 0o750)
        entrance_monitor.set_value(ENTRANCE_TYPE, 'SDK')

        with self.assertRaises(TypeError):
            PathRightEscalationCheck(option_name='path_right_escalation_check',
                                     option_value=path,
                                     mode=True,
                                     force_quit=True,
                                     quiet=False)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_right_escalation_check_with_invalid_mode_type.')

    def test_path_right_escalation_check_with_invalid_mode_length(self):
        """
        测试路径权限提升校验, 参数mode传不合理长度值的情况
        """
        logging.info('Start test_path_right_escalation_check_with_invalid_mode_length.')
        path = os.path.join(LOCAL_PATH, 'path_right_escalation_check')
        if not os.path.exists(path):
            os.makedirs(path)

        os.chmod(path, 0o750)
        entrance_monitor.set_value(ENTRANCE_TYPE, 'SDK')

        with self.assertRaises(ValueError):
            PathRightEscalationCheck(option_name='path_right_escalation_check',
                                     option_value=path,
                                     mode='75',
                                     force_quit=True,
                                     quiet=False)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_right_escalation_check_with_invalid_mode_length.')

    def test_path_right_escalation_check_with_invalid_mode_letter(self):
        """
        测试路径权限提升校验, 参数mode传值包含不合理字符的情况
        """
        logging.info('Start test_path_right_escalation_check_with_invalid_mode_letter.')
        path = os.path.join(LOCAL_PATH, 'path_right_escalation_check')
        if not os.path.exists(path):
            os.makedirs(path)

        os.chmod(path, 0o750)
        entrance_monitor.set_value(ENTRANCE_TYPE, 'SDK')

        with self.assertRaises(TypeError):
            PathRightEscalationCheck(option_name='path_right_escalation_check',
                                     option_value=path,
                                     mode='7A0',
                                     force_quit=True,
                                     quiet=False)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_right_escalation_check_with_invalid_mode_letter.')

    def test_path_right_escalation_check_with_invalid_mode_range(self):
        """
        测试路径权限提升校验, 参数mode传值数字越界的情况
        """
        logging.info('Start test_path_right_escalation_check_with_invalid_mode_range.')
        path = os.path.join(LOCAL_PATH, 'path_right_escalation_check')
        if not os.path.exists(path):
            os.makedirs(path)

        os.chmod(path, 0o750)
        entrance_monitor.set_value(ENTRANCE_TYPE, 'SDK')

        with self.assertRaises(ValueError):
            PathRightEscalationCheck(option_name='path_right_escalation_check',
                                     option_value=path,
                                     mode='787',
                                     force_quit=True,
                                     quiet=False)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_right_escalation_check_with_invalid_mode_range.')

    def test_path_right_escalation_check_with_invalid_mode(self):
        """
        测试路径权限提升校验, 参数mode传不合理值的情况
        """
        logging.info('Start test_path_right_escalation_check_with_invalid_mode.')

        path = os.path.join(LOCAL_PATH, 'path_right_escalation_check')
        if not os.path.exists(path):
            os.makedirs(path)

        os.chmod(path, 0o777)
        entrance_monitor.set_value(ENTRANCE_TYPE, 'SDK')

        with self.assertRaises(PathRightEscalationError):
            PathRightEscalationCheck(option_name='path_right_escalation_check',
                                     option_value=path,
                                     mode='770',
                                     force_quit=True,
                                     quiet=False)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info('Finish test_path_right_escalation_check_with_invalid_mode.')

    def test_file_size_check_with_none_path_including_file(self):
        """
        测试文件大小校验, 参数path_including_file传None值的情况
        """
        logging.info('Start test_file_size_check_with_none_path_including_file.')

        path = os.path.join(LOCAL_PATH, 'file_size_check')
        if not os.path.exists(path):
            os.makedirs(path)

        file_name = 'size_check.txt'
        full_path = os.path.join(path, file_name)

        with os.fdopen(os.open(full_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('size_check')

        with self.assertRaises(ValueError):
            FileSizeCheck(option_name='file_size_check',
                          option_value=full_path,
                          path_including_file=None)

        if os.path.exists(full_path):
            os.remove(full_path)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info("Finish test_file_size_check_with_none_path_including_file.")

    def test_file_size_check_with_invalid_path_including_file_type(self):
        """
        测试文件大小校验, 参数path_including_file传不合理类型值的情况
        """
        logging.info('Start test_file_size_check_with_invalid_path_including_file_type.')

        path = os.path.join(LOCAL_PATH, 'file_size_check')
        if not os.path.exists(path):
            os.makedirs(path)

        file_name = 'size_check.txt'
        full_path = os.path.join(path, file_name)

        with os.fdopen(os.open(full_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('size_check')

        with self.assertRaises(TypeError):
            FileSizeCheck(option_name='file_size_check',
                          option_value=full_path,
                          path_including_file="True")

        if os.path.exists(full_path):
            os.remove(full_path)

        if os.path.exists(path):
            os.rmdir(path)

        logging.info("Finish test_file_size_check_with_invalid_path_including_file_type.")

    def test_file_size_check(self):
        """
        测试文件大小校验
        """
        logging.info('Start test_file_size_check.')

        path = os.path.join(LOCAL_PATH, 'file_size_check')
        if not os.path.exists(path):
            os.makedirs(path)

        file_name = 'size_check.txt'
        full_path = os.path.join(path, file_name)

        with os.fdopen(os.open(full_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('*' * 20 * 1024 * 1024)  # 生成20MB文件

        with self.assertRaises(FileOversizeError):
            FileSizeCheck(option_name='file_size_check',
                          option_value=full_path,
                          path_including_file=True)

        if os.path.exists(full_path):
            os.remove(full_path)

        if os.path.exists(path):
            os.rmdir(path)

    def test_disk_free_space_check_with_none_free_space_limit(self):
        """
        测试路径所在磁盘剩余空间是否充足校验, 参数free_space_limit传None值的情况
        """
        logging.info('Start testing test_disk_free_space_check_with_none_free_space_limit.')

        path = os.path.join(LOCAL_PATH, 'file_size_check')
        if not os.path.exists(path):
            os.makedirs(path)

        file_name = 'size_check.txt'
        full_path = os.path.join(path, file_name)

        with os.fdopen(os.open(full_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('*' * 100 * 1024 * 1024)  # 生成100MB文件

        with self.assertRaises(ValueError):
            DiskFreeSpaceCheck(option_name='disk_free_space_check',
                               option_value=full_path,
                               free_space_limit=None,
                               force_quit=True,
                               quiet=False)

        logging.info('Finish testing test_disk_free_space_check_with_none_free_space_limit.')

    def test_disk_free_space_check_with_invalid_free_space_limit_type(self):
        """
        测试路径所在磁盘剩余空间是否充足校验, 参数free_space_limit传不合理类型值的情况
        """
        logging.info('Start testing test_disk_free_space_check_with_invalid_free_space_limit_type.')

        path = os.path.join(LOCAL_PATH, 'file_size_check')
        if not os.path.exists(path):
            os.makedirs(path)

        file_name = 'size_check.txt'
        full_path = os.path.join(path, file_name)

        with os.fdopen(os.open(full_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('*' * 100 * 1024 * 1024)  # 生成100MB文件

        with self.assertRaises(TypeError):
            DiskFreeSpaceCheck(option_name='disk_free_space_check',
                               option_value=full_path,
                               free_space_limit='1000',
                               force_quit=True,
                               quiet=False)

        logging.info('Finish testing test_disk_free_space_check_with_invalid_free_space_limit_type.')

    def test_disk_free_space_check_with_negative_free_space_limit(self):
        """
        测试路径所在磁盘剩余空间是否充足校验, 参数free_space_limit传负数值的情况
        """
        logging.info('Start test_disk_free_space_check_with_negative_free_space_limit.')

        path = os.path.join(LOCAL_PATH, 'disk_free_space_check')
        if not os.path.exists(path):
            os.makedirs(path)

        file_name = 'disk_free_size_check.txt'
        full_path = os.path.join(path, file_name)

        with os.fdopen(os.open(full_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('*' * 100 * 1024 * 1024)  # 生成100MB文件

        with self.assertRaises(ValueError):
            DiskFreeSpaceCheck(option_name='disk_free_space_check',
                               option_value=full_path,
                               free_space_limit=-1,
                               force_quit=True,
                               quiet=False)

        logging.info('Finish test_disk_free_space_check_with_negative_free_space_limit.')

    def test_disk_free_space_check(self):
        """
        测试路径所在磁盘剩余空间是否充足校验
        """
        logging.info('Start test_disk_free_space_check.')

        path = os.path.join(LOCAL_PATH, 'disk_free_space_check')
        if not os.path.exists(path):
            os.makedirs(path)

        file_name = 'disk_free_size_check.txt'
        full_path = os.path.join(path, file_name)

        with os.fdopen(os.open(full_path, DEFAULT_FLAGS, DEFAULT_MODES), 'w') as file:
            file.write('*' * 100 * 1024 * 1024)  # 生成100MB文件

        with self.assertRaises(LowDiskFreeSizeRiskError):
            DiskFreeSpaceCheck(option_name='disk_free_space_check',
                               option_value=full_path,
                               free_space_limit=1000000 * GB_SIZE,
                               force_quit=True,
                               quiet=False)

        logging.info('Finish test_disk_free_space_check.')

    def test_get_real_path_with_none_input(self):
        """
        测试获取真实路径, 入参路径传None值的情况
        """
        logging.info('Start test_get_real_path_with_none_input.')
        with self.assertRaises(ValueError):
            get_real_path(path=None)
        logging.info('Finish test_get_real_path_with_none_input.')

    def test_get_real_path_with_empty_input(self):
        """
        测试获取真实路径, 入参路径传空字符串的情况
        """
        logging.info('Start test_get_real_path_with_empty_input.')
        with self.assertRaises(ValueError):
            get_real_path(path=EMPTY_STRING)
        logging.info('Finish test_get_real_path_with_empty_input.')


if __name__ == '__main__':
    pytest.main(['-s', os.path.abspath(__file__)])
