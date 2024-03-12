#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

"""
功能: 保存可训练参数功能单元测试模块
"""
import sys
sys.path.append('.')
import os
import shutil
import logging
import time
from unittest import TestCase, mock

import mindspore.ops
import pytest

import mindspore.nn as nn
from mindspore.nn import Accuracy
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype, LossMonitor, Model, CheckpointConfig

from mindpet.graph.ckpt_util import TrainableParamsCheckPoint
from mindpet.utils.exceptions import AbsolutePathError, LinkPathError

logging.getLogger().setLevel(logging.INFO)

cur_dir = os.path.abspath(os.path.dirname(__file__))
BASE_CKPT_NAME = "checkpoint_base-1_3.ckpt"
DELTA_CKPT_NAME = "checkpoint_delta-1_3.ckpt"


class TestNet(nn.Cell):
    """
    Lenet网络结构
    """

    def __init__(self, num_class=10, num_channel=1):
        super(TestNet, self).__init__()
        # 定义所需要的运算
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc_temp = nn.Dense(32 * 32, 10)
        logging.info("Init net successfully.")

    def construct(self, value):
        # 使用定义好的运算构建前向网络
        value = mindspore.ops.reshape(value, (value.shape[0], -1))
        value = self.fc_temp(value)
        return value


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    logging.info("Start to create dataset.")
    # 定义数据集
    mnist_ds = ds.MnistDataset(dataset_dir=data_path, num_samples=30)
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # 定义所需要操作的map映射
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # 使用map映射函数，将数据操作应用到数据集
    mnist_ds = mnist_ds.map(operations=type_cast_op,
                            input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op,
                            input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op,
                            input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op,
                            input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op,
                            input_columns="image", num_parallel_workers=num_parallel_workers)

    # 进行shuffle、batch操作
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    logging.info("End to create dataset.")
    return mnist_ds


def train_net(model, data_path, ckpoint_cb, sink_mode):
    """定义训练的方法"""
    logging.info("Start to train.")
    # 加载训练数据集
    ds_train = create_dataset(os.path.join(data_path, "data"), 10, 1)
    model.train(1, ds_train, callbacks=[ckpoint_cb, LossMonitor(125)], dataset_sink_mode=sink_mode)
    logging.info("End to train.")


def train(ckpt_callback=None, enable=False):
    net = TestNet()
    params = net.trainable_params()
    if enable:
        logging.info("Enable to save trainable params.")
        for param in params:
            if "fc" in param.name:
                param.requires_grad = False

    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
    ckpt_path = os.path.join(cur_dir, "output")
    os.makedirs(ckpt_path, exist_ok=True)
    # 设置模型保存参数
    if ckpt_callback is None:
        config_ck = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=1)
        # 应用模型保存参数
        ckpt_callback = ModelCheckpoint(prefix="checkpoint_base", directory=ckpt_path, config=config_ck)
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    train_net(model, cur_dir, ckpt_callback, False)


class TestSaveCkpt(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ckpt_path = os.path.join(cur_dir, "output")
        cls.ckpt_file_path = os.path.join(cls.ckpt_path, BASE_CKPT_NAME)
        if not (os.path.exists(cls.ckpt_path) and os.path.exists(cls.ckpt_file_path)):
            train()

    @classmethod
    def tearDownClass(cls) -> None:
        ckpt_path = os.path.join(cur_dir, "ckpt")
        if os.path.exists(ckpt_path):
            shutil.rmtree(ckpt_path, ignore_errors=True)
            logging.info("Delete ckpt file successfully.")

    def test_directory_none(self):
        logging.info('Start test_directory_none.')
        with self.assertRaises(ValueError):
            params_check_point = TrainableParamsCheckPoint(prefix='test', directory=None, config=None)
        logging.info('Finish test_directory_none.')

    def test_directory_empty(self):
        logging.info('Start test_directory_empty.')
        with self.assertRaises(ValueError):
            params_check_point = TrainableParamsCheckPoint(prefix='test', directory="", config=None)
        logging.info('Finish test_directory_empty.')

    def test_directory_abs(self):
        logging.info('Start test_directory_abs.')
        with self.assertRaises(AbsolutePathError):
            params_check_point = TrainableParamsCheckPoint(prefix='test', directory="../", config=None)
        logging.info('Finish test_directory_abs.')

    def test_directory_link(self):
        logging.info('Start test_directory_link.')
        src_path = os.path.join(cur_dir, "src")
        os.makedirs(src_path, exist_ok=True)

        dst_path = os.path.join(cur_dir, "dst")
        os.symlink(src_path, dst_path)
        with self.assertRaises(LinkPathError):
            params_check_point = TrainableParamsCheckPoint(prefix='test', directory=dst_path, config=None)
        os.rmdir(src_path)
        os.remove(dst_path)
        logging.info('Finish test_directory_link.')

    def test_config_type(self):
        logging.info('Start test_config_type.')
        config = dict()
        with self.assertRaises(TypeError):
            params_check_point = TrainableParamsCheckPoint(prefix='test', directory=cur_dir, config=config)
        logging.info('Finish test_config_type.')

    def test_config_illegal_args(self):
        logging.info('Start test_config_illegal_args.')
        with self.assertRaises(ValueError):
            config = CheckpointConfig(save_checkpoint_steps=-1,
                                      save_checkpoint_seconds=0,
                                      keep_checkpoint_max=5,
                                      keep_checkpoint_per_n_minutes=0)
        with self.assertRaises(ValueError):
            config = CheckpointConfig(save_checkpoint_steps=1,
                                      save_checkpoint_seconds=-1,
                                      keep_checkpoint_max=5,
                                      keep_checkpoint_per_n_minutes=0)
        with self.assertRaises(ValueError):
            config = CheckpointConfig(save_checkpoint_steps=1,
                                      save_checkpoint_seconds=0,
                                      keep_checkpoint_max=-1,
                                      keep_checkpoint_per_n_minutes=0)
        with self.assertRaises(ValueError):
            config = CheckpointConfig(save_checkpoint_steps=1,
                                      save_checkpoint_seconds=0,
                                      keep_checkpoint_max=5,
                                      keep_checkpoint_per_n_minutes=-1)

        logging.info('Finish test_config_illegal_args.')

    def test_prefix_type(self):
        logging.info('Start test_prefix_type.')
        prefix = 0
        with self.assertRaises(ValueError):
            params_check_point = TrainableParamsCheckPoint(prefix=prefix, directory=cur_dir, config=None)
        logging.info('Finish test_prefix_type.')

    def test_prefix_contains_illegal_char(self):
        logging.info('Start test_prefix_contains_illegal_char.')
        prefix = "prefix/name"
        with self.assertRaises(ValueError):
            params_check_point = TrainableParamsCheckPoint(prefix=prefix, directory=cur_dir, config=None)
        logging.info('Finish test_prefix_contains_illegal_char.')

    def test_save_ckpt(self):
        logging.info('Start test_save_ckpt.')

        ckpt_path = os.path.join(cur_dir, "ckpt")
        config_ck = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=1)
        params_check_point = TrainableParamsCheckPoint(prefix='checkpoint_delta',
                                                       directory=ckpt_path, config=config_ck)
        train(params_check_point, enable=True)
        self.assertTrue(os.path.exists(ckpt_path))

        delta_ckpt_file_path = os.path.join(ckpt_path, DELTA_CKPT_NAME)

        self.assertTrue(os.path.exists(delta_ckpt_file_path))

        origin_fsize = os.path.getsize(self.ckpt_file_path)
        delta_fsize = os.path.getsize(delta_ckpt_file_path)
        self.assertGreater(origin_fsize, delta_fsize)

        delta_param_dict = load_checkpoint(delta_ckpt_file_path)
        if os.path.exists(self.ckpt_file_path):
            origin_param_dict = load_checkpoint(self.ckpt_file_path)

        self.assertGreater(len(origin_param_dict), len(delta_param_dict))

        for k, _ in delta_param_dict.items():
            self.assertNotIn("fc", k)
        logging.info('Finish test_save_ckpt.')

    def test_trans_network(self):
        """
        测试network中选取可训练的参数时，网络的parameter_layout_dict中参数能否合并
        """
        logging.info('Start test_trans_network.')

        ckpt_path = os.path.join(cur_dir, "ckpt")
        config_ck = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=1)
        params_check_point = TrainableParamsCheckPoint(prefix='checkpoint_delta',
                                                       directory=ckpt_path, config=config_ck)
        net = TestNet()
        if os.path.exists(self.ckpt_file_path):
            origin_param_dict = load_checkpoint(self.ckpt_file_path)
        load_param_into_net(net, parameter_dict=origin_param_dict)
        params = net.trainable_params()

        for param in params:
            if param.name == "fc_temp.weight":
                net.parameter_layout_dict[param.name] = [0, 0]
        param_list = params_check_point.trans_network(net)

        self.assertEqual(param_list[0].get("name"), params[0].name)

        logging.info('Finish test_trans_network.')

    def test_ckpt_num(self):
        logging.info('Start test_ckpt_num.')
        num = 3
        ckpt_path = os.path.join(cur_dir, "ckpt_num")
        config_ck = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=num)
        params_check_point = TrainableParamsCheckPoint(prefix='checkpoint_delta',
                                                       directory=ckpt_path, config=config_ck)
        train(params_check_point, enable=True)
        self.assertTrue(os.path.exists(ckpt_path))

        file_list = os.listdir(ckpt_path)

        ckpt_num = 0

        for file in file_list:
            if ".ckpt" in file:
                ckpt_num += 1
        self.assertEqual(ckpt_num, num)

        shutil.rmtree(ckpt_path, ignore_errors=True)

        logging.info('Finish test_ckpt_num.')

    @mock.patch("mindpet.graph.ckpt_util.TrainableParamsCheckPoint._check_save_ckpt")
    def test_check_save_ckpt(self, mock_func):
        logging.info('Start test_check_save_ckpt.')
        ckpt_path = os.path.join(cur_dir, "temp")
        config_ck = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=1)
        params_check_point = TrainableParamsCheckPoint(prefix='checkpoint_delta',
                                                       directory=ckpt_path, config=config_ck)
        mock_func.return_value = False
        train(params_check_point, enable=True)
        self.assertTrue(os.path.exists(ckpt_path))
        file_list = os.listdir(ckpt_path)

        for file in file_list:
            self.assertNotIn(".ckpt", file)

        shutil.rmtree(ckpt_path, ignore_errors=True)
        logging.info('Finish test_check_save_ckpt.')

    def test_batch_num_illegal(self):
        logging.info('Start test_batch_num_illegal.')

        class TestCheckPoint(TrainableParamsCheckPoint):
            def step_end(self, run_context):
                cb_params = run_context.original_args()
                cb_params.batch_num = -1
                super(TestCheckPoint, self).step_end(run_context)

        ckpt_path = os.path.join(cur_dir, "tmp")
        config_ck = CheckpointConfig(save_checkpoint_steps=3, keep_checkpoint_max=1)
        params_check_point = TestCheckPoint(prefix='checkpoint_delta',
                                            directory=ckpt_path, config=config_ck)
        with self.assertRaises(ValueError):
            train(params_check_point, enable=True)

        shutil.rmtree(ckpt_path, ignore_errors=True)

        logging.info('Finish test_batch_num_illegal.')

    def test_epoch_num(self):
        logging.info('Start test_epoch_num.')

        ckpt_path = os.path.join(cur_dir, "epoch")
        config_ck = CheckpointConfig(save_checkpoint_steps=3, keep_checkpoint_max=1,
                                     append_info=["epoch_num", "step_num"])
        params_check_point = TrainableParamsCheckPoint(prefix='checkpoint_delta',
                                                       directory=ckpt_path, config=config_ck)
        train(params_check_point, enable=True)

        shutil.rmtree(ckpt_path, ignore_errors=True)

        logging.info('Finish test_epoch_num.')

    def test_keep_ckpt_per_min(self):
        logging.info('Start test_keep_ckpt_per_min.')

        ckpt_path = os.path.join(cur_dir, "per_min")
        config_ck = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=None, keep_checkpoint_per_n_minutes=2)
        params_check_point = TrainableParamsCheckPoint(prefix='checkpoint_delta',
                                                       directory=ckpt_path, config=config_ck)
        train(params_check_point, enable=True)

        shutil.rmtree(ckpt_path, ignore_errors=True)

        logging.info('Start test_keep_ckpt_per_min.')

    def test_set_per_minute(self):
        logging.info('Start test_set_per_minute.')
        now = time.time() - 2 * 60

        class TestCheckPoint(TrainableParamsCheckPoint):
            def __init__(self, directory, prefix="DELTA_CKP", config=None):
                super(TestCheckPoint, self).__init__(directory, prefix=prefix, config=config)
                self._last_time_for_keep = now

            def test_method(self):
                self._set_per_minute()

        ckpt_path = os.path.join(cur_dir, "per_min")
        config_ck = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=None, keep_checkpoint_per_n_minutes=1)
        params_check_point = TestCheckPoint(prefix='checkpoint_delta',
                                            directory=ckpt_path, config=config_ck)

        params_check_point.test_method()

        logging.info('Finish test_set_per_minute.')

    @mock.patch("mindspore.common.api._get_compile_cache_dep_files")
    @mock.patch("mindspore.context.get_context")
    def test_enable_ge(self, mock_api, mock_func):
        logging.info('Start test_enable_ge.')
        mock_api.return_value = True
        mock_func.return_value = []
        ckpt_path = os.path.join(cur_dir, "ge")
        config_ck = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=1)
        params_check_point = TrainableParamsCheckPoint(prefix='checkpoint_delta',
                                                       directory=ckpt_path, config=config_ck)
        train(params_check_point, enable=True)

        self.assertTrue(os.path.exists(ckpt_path))

        shutil.rmtree(ckpt_path, ignore_errors=True)

        logging.info('Finish test_enable_ge.')


if __name__ == '__main__':
    pytest.main(['-s', os.path.abspath(__file__)])

# Set source attribute for function TestNet.construct to support run so or pyc file in Graph Mode.
setattr(TestNet.construct, 'source', (['    def construct(self, value):\n', '        # 使用定义好的运算构建前向网络\n', '        value = mindspore.ops.reshape(value, (value.shape[0], -1))\n', '        value = self.fc_temp(value)\n', '        return value\n'], 56))
