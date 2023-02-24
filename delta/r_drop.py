#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation, 2022-2023, All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# Copyright © Huawei Technologies Co., Ltd. 2010-2023. All rights reserved.

import numpy as np
from mindspore import nn
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
import mindspore.ops as ops


class RDropLoss(nn.Cell):
    """The loss between input and target with r_drop."""
    def __init__(self):
        super(RDropLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.softmax = nn.Softmax(axis=-1)
        self.reshape = ops.Reshape()
        self.gather = ops.Gather()

    def construct(self, logits, label_ids, alpha=4):
        """
        Inputs:
            - **logits** (Tensor) - Tensor of shape :math:`(C,)` :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` ,
            where `C = number of classes`. Data type must be float16 or float32.

            - **labels** (Tensor) - For class indices, tensor of shape :math:`()`, :math:`(N)` or
            :math:`(N, d_1, d_2, ..., d_K)` , data type must be int32.
            For probabilities, tensor of shape :math:`(C,)` :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` ,
            data type must be float16 or float32.

            - **alpha** (int) - Super parameter, specify the disturbance effect.

        Returns:
            Tensor, the computed cross entropy loss value.
        """
        batch_size = logits.shape[0]
        label_ids = self.reshape(label_ids, (-1,))

        indices_1 = Tensor(np.arange(0, batch_size, 2), mstype.int32)
        indices_2 = Tensor(np.arange(1, batch_size, 2), mstype.int32)

        logits_1 = self.gather(logits, indices_1, 0)
        logits_2 = self.gather(logits, indices_2, 0)

        ce_loss = self.cross_entropy_loss(logits, label_ids)

        kl_loss_1 = ops.kl_div(logits=self.log_softmax(logits_1), labels=self.softmax(logits_2), reduction='sum')
        kl_loss_2 = ops.kl_div(logits=self.log_softmax(logits_2), labels=self.softmax(logits_1), reduction='sum')
        kl_loss = (kl_loss_1 + kl_loss_2).mean() / 4 * alpha
        return_value = ce_loss + kl_loss
        return return_value


def rdrop_repeat(*items):
    res = list()
    for item in items:
        res.append(item.repeat(2, axis=0))
    return tuple(res)
