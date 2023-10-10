#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os
import pwd
import logging
import subprocess

MODE_750 = 0o750
TIMEOUT = 10

logging.getLogger().setLevel(logging.INFO)


def change_path_mode(path, mode):
    if os.path.exists(path):
        os.chmod(path, mode)


def finetune_launcher(user_name):
    root_path = os.path.join('/home/', user_name, 'mxTuningKit/test/developer_test/task').replace('\\', '/')

    dataset_path = os.path.join(root_path, 'dataset').replace('\\', '/')
    output_path = os.path.join(root_path, 'outputs').replace('\\', '/')
    pretrained_model_path = os.path.join(root_path, 'pretrained_models').replace('\\', '/')
    model_config_path = os.path.join(root_path, 'model_config_finetune.yaml').replace('\\', '/')
    boot_file_path = os.path.join(root_path, 'finetune_launcher.py').replace('\\', '/')

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(pretrained_model_path, exist_ok=True)

    change_path_mode(dataset_path, MODE_750)
    change_path_mode(output_path, MODE_750)
    change_path_mode(pretrained_model_path, MODE_750)
    change_path_mode(model_config_path, MODE_750)
    change_path_mode(boot_file_path, MODE_750)

    cmd = ['mindpet',
           'finetune',
           '--quiet',
           '--data_path', dataset_path,
           '--output_path', output_path,
           '--pretrained_model_path', pretrained_model_path,
           '--model_config_path', model_config_path,
           '--boot_file_path', boot_file_path]

    logging.info('finetune task is running.')

    process = subprocess.Popen(cmd, env=os.environ, shell=False,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    std_err, std_msg = process.communicate(timeout=TIMEOUT)
    logging.info(f'STDERR:{std_err}')
    logging.info(f'STDMSG:{std_msg}')
    logging.info(f'get finetune task exit code: {process.poll()}')


def evaluate_launcher(user_name):
    root_path = os.path.join('/home/', user_name, 'mxTuningKit/test/developer_test/task').replace('\\', '/')

    dataset_path = os.path.join(root_path, 'dataset').replace('\\', '/')
    output_path = os.path.join(root_path, 'outputs').replace('\\', '/')
    ckpt_path = os.path.join(root_path, 'ckpt').replace('\\', '/')
    model_config_path = os.path.join(root_path, 'model_config_eval.yaml').replace('\\', '/')
    boot_file_path = os.path.join(root_path, 'eval_launcher.py').replace('\\', '/')

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)

    change_path_mode(dataset_path, MODE_750)
    change_path_mode(output_path, MODE_750)
    change_path_mode(ckpt_path, MODE_750)
    change_path_mode(model_config_path, MODE_750)
    change_path_mode(boot_file_path, MODE_750)

    cmd = ['mindpet',
           'evaluate',
           '--quiet',
           '--data_path', dataset_path,
           '--output_path', output_path,
           '--ckpt_path', ckpt_path,
           '--model_config_path', model_config_path,
           '--boot_file_path', boot_file_path]

    logging.info('evaluate task is running.')

    process = subprocess.Popen(cmd, env=os.environ, shell=False,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    std_err, std_msg = process.communicate(timeout=TIMEOUT)
    logging.info(f'STDERR:{std_err}')
    logging.info(f'STDMSG:{std_msg}')
    logging.info(f'get evaluate task exit code: {process.poll()}')


def infer_launcher(user_name):
    root_path = os.path.join('/home/', user_name, 'mxTuningKit/test/developer_test/task').replace('\\', '/')

    dataset_path = os.path.join(root_path, 'dataset').replace('\\', '/')
    output_path = os.path.join(root_path, 'outputs').replace('\\', '/')
    ckpt_path = os.path.join(root_path, 'ckpt').replace('\\', '/')
    model_config_path = os.path.join(root_path, 'model_config_infer.yaml').replace('\\', '/')
    boot_file_path = os.path.join(root_path, 'infer_launcher.py').replace('\\', '/')

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)

    change_path_mode(dataset_path, MODE_750)
    change_path_mode(output_path, MODE_750)
    change_path_mode(ckpt_path, MODE_750)
    change_path_mode(model_config_path, MODE_750)
    change_path_mode(boot_file_path, MODE_750)

    cmd = ['mindpet',
           'infer',
           '--quiet',
           '--data_path', dataset_path,
           '--output_path', output_path,
           '--ckpt_path', ckpt_path,
           '--model_config_path', model_config_path,
           '--boot_file_path', boot_file_path]

    logging.info('infer task is running.')

    process = subprocess.Popen(cmd, env=os.environ, shell=False,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    std_err, std_msg = process.communicate(timeout=TIMEOUT)
    logging.info(f'STDERR:{std_err}')
    logging.info(f'STDMSG:{std_msg}')
    logging.info(f'get infer task exit code: {process.poll()}')


if __name__ == '__main__':
    task = input('which task (finetune/evaluate/infer): ')
    current_user = input('user name (press enter to use current user): ')

    if not current_user:
        current_user = pwd.getpwuid(os.geteuid()).pw_name

    logging.info(f'user name: {current_user}')

    if task == 'finetune':
        finetune_launcher(current_user)
    elif task == 'evaluate':
        evaluate_launcher(current_user)
    elif task == 'infer':
        infer_launcher(current_user)
    else:
        logging.error('invalid task type.')
