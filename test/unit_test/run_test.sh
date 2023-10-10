#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

set -e

MXTUNINGKIT_TAR_FILE_NAME="Ascend-mindxsdk-mxtuningkit*.tar.gz"
MXTUNINGKIT_WHL_FILE_NAME="Ascend_mindxsdk_mxTuningKit*.whl"

function info() {
  echo -e "\033[1;34m[INFO][mxTuningKit] $1\033[1;37m"
}

function warn() {
  echo >&2 -e "\033[1;31m[WARN][mxTuningKit] $1\033[1;37m"
}

function get_dt_path() {
  info "Start getting DT path."

  CUR_PATH=$(
    cd "$(dirname "$0")" || {
      warn "Failed to get path/to/run_test.sh"
      exit
    }
    pwd
  )

  cd "${CUR_PATH}"

  info "Finish getting DT path."
}

function clear_dt_cache() {
  info "Start cleaning cache files."

  rm -rf result
  mkdir result

  info "Finish cleaning cache files."
}

function unpack_tar() {
  info "Start unpacking tar file."

  for tar_file in $(ls $MXTUNINGKIT_TAR_FILE_NAME); do
    tar_file_name=$(basename $tar_file)
  done

  if [[ -z "$tar_file_name" ]]; then
    warn "File $MXTUNINGKIT_TAR_FILE_NAME does not exist."
    exit 1
  fi

  tar -zxvf $tar_file_name

  for whl_file in $(ls $MXTUNINGKIT_WHL_FILE_NAME); do
    whl_file_name=$(basename $whl_file)
  done

  info "Finish unpacking tar file."
}

function install_dependence() {
  info "Start installing dependence."

  # 卸载镜像预置mindspore 1.7及mindspore-ascend包, 避免冲突
  pip3 uninstall mindspore -y
  pip3 uninstall mindspore-ascend -y

  # 安装mxTuningKit相关依赖
  pip3 install $whl_file_name

  # 安装pytest相关依赖
  pip3 install pytest
  pip3 install pytest-cov

  # 安装mindspore相关依赖
  pip3 install --trusted-host mirrors.tools.huawei.com -i https://mirrors.tools.huawei.com/pypi/simple mindspore

  info "Finish installing dependence."
}

function run_dt_test_cases() {
  info "Start getting testcase final result."

  pytest -v --junit-xml=./final.xml --cov=mindpet --cov-report=html --cov-report=xml --disable-pytest-warnings \
    --cov-branch --cache-clear

  mv .coverage coverage.xml final.xml htmlcov result

  info "Finish getting testcase final result."
}

function main() {
  get_dt_path
  clear_dt_cache
  unpack_tar
  install_dependence
  run_dt_test_cases
}

main
