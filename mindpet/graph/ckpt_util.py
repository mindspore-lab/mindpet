#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

"""
功能: 保存可训练参数功能模块
"""
import os
import time
from collections import OrderedDict

from mindspore import nn
from mindspore import context, Tensor
from mindspore.train.callback._callback import set_cur_net
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.serialization import save_checkpoint, _get_merged_param_data

from mindpet.utils.exceptions import AbsolutePathError, LinkPathError
from mindpet.utils.io_utils import is_link_path
from mindpet.log.log import logger


class TrainableParamsCheckPoint(ModelCheckpoint):
    """TrainableParamsCheckPoint class"""
    def __init__(self, directory, prefix="DELTA_CKP", config=None):
        """
        Callback初始化
        :param directory: ckpt文件的保存路径
        :param prefix: ckpt文件的前缀名称
        :param config: 保存ckpt的配置
        """
        if directory is None or directory == "":
            raise ValueError('Param [directory] is required.')
        if is_link_path(directory):
            raise LinkPathError(f'Path of param [{directory}] is a link path.')
        if not os.path.isabs(directory):
            raise AbsolutePathError(f'Param [{directory}] is not an absolute path.')

        self._last_time_for_keep = time.time()
        self._last_triggered_step = 0
        self._latest_ckpt_file_name = ""
        self._cur_time_for_keep = time.time()
        super().__init__(prefix, directory, config)

    def set_train_info(self, cb_params, cur_step_num):
        """保持ckpt文件始终小于等于配置的ckpt文件最大数"""
        if self._config.keep_checkpoint_max and \
                0 < self._config.keep_checkpoint_max <= self._manager.ckpoint_num:
            self._manager.remove_oldest_ckpoint_file()
        elif self._config.keep_checkpoint_per_n_minutes and self._config.keep_checkpoint_per_n_minutes > 0:
            self._cur_time_for_keep = time.time()
            self._set_per_minute()
        self._last_time_for_keep = time.time()
        self._last_triggered_step = cur_step_num
        if context.get_context("enable_ge"):
            set_cur_net(cb_params.train_network)
            cb_params.train_network.exec_checkpoint_graph()
        # 更新当前epoch数
        if "epoch_num" in self._append_dict:
            self._append_dict["epoch_num"] = self._append_epoch_num + cb_params.cur_epoch_num
        # 更新当前step数
        if "step_num" in self._append_dict:
            self._append_dict["step_num"] = self._append_step_num + cur_step_num

    def trans_network(self, network):
        """从network中选取可训练的参数，仅保存这部分参数"""
        parameter_layout_dict = network.parameter_layout_dict
        network.init_parameters_data()
        param_dict = OrderedDict()
        for param in network.trainable_params():
            param_dict[param.name] = param
        param_list = []
        for param_name, param in param_dict.items():
            each = {"name": param_name}
            param_data = Tensor(param.data.asnumpy())
            if param_name in network.parameter_layout_dict:
                param_data = _get_merged_param_data(network, parameter_layout_dict, param_name, param_data,
                                                    self._config.integrated_save)
            each["data"] = param_data
            param_list.append(each)
        return param_list

    def _set_per_minute(self):
        """只保留每分钟内保存的最后一份ckpt文件"""
        if (self._cur_time_for_keep - self._last_time_for_keep) \
                < self._config.keep_checkpoint_per_n_minutes * 60:
            self._manager.keep_one_ckpoint_per_minutes(self._config.keep_checkpoint_per_n_minutes,
                                                       self._cur_time_for_keep)

    def _save_ckpt(self, cb_params, force_to_save=False):
        """从上下文中获取参数，并且保存ckpt文件"""

        cur_step_num = cb_params.cur_step_num

        if cur_step_num == self._last_triggered_step:
            return

        # 如果参数支持缓存，则在保存ckpt文件之前刷新缓存
        if self._need_flush_from_cache:
            self._flush_from_cache(cb_params)

        enable_save = self._check_save_ckpt(cb_params, force_to_save)

        if not enable_save:
            return

        logger.info("Start to save checkpoint.")

        ckpt_prefix = self._prefix
        if cb_params.batch_num <= 0:
            raise ValueError("[batch_num] can\'t be 0 or less.")

        step_num_in_epoch = int((cur_step_num - 1) % cb_params.batch_num + 1)
        cur_ckpoint_file = ckpt_prefix + "-" + str(cb_params.cur_epoch_num) + "_" \
                           + str(step_num_in_epoch) + ".ckpt"
        ckpt_directory = self._directory
        cur_file = os.path.join(ckpt_directory, cur_ckpoint_file)

        self._manager.update_ckpoint_filelist(ckpt_directory, ckpt_prefix)

        # 训练时维护训练的信息
        self.set_train_info(cb_params, cur_step_num)

        # 获取当前训练的网络
        network = self._config.saved_network if self._config.saved_network is not None else cb_params.train_network

        if isinstance(network, nn.Cell):
            network = self.trans_network(network)
        # 将参数保存为ckpt
        save_checkpoint(network, cur_file, self._config.integrated_save, self._config.async_save,
                        self._append_dict, self._config.enc_key, self._config.enc_mode)
        self._latest_ckpt_file_name = cur_file

        logger.info("Save checkpoint successfully.")
