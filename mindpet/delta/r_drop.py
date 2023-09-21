#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation, 2022-2023, All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# Copyright © Huawei Technologies Co., Ltd. 2010-2023. All rights reserved.
"""R drop Cell"""
import numpy as np
from mindspore import nn
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore import ops


class RDropLoss(nn.Cell):
    """The loss between input and target with r_drop."""
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.softmax = nn.Softmax(axis=-1)
        self.reshape = ops.Reshape()
        self.gather = ops.Gather()
        self.kl_div = ops.KLDivLoss(reduction='sum')

    def construct(self, logits, label_ids, alpha=4):
        """
        Inputs:
            - **logits** (Tensor) - Predictive value.
            Data type must be mindspore.dtype.float16 or mindspore.dtype.float32.

            - **labels** (Tensor) - Target value.
            Data type must be mindspore.dtype.float16 or mindspore.dtype.float32.

            - **alpha** (int) - Super parameter, specify the disturbance effect.

        Returns:
            Tensor, the computed loss value.
        """
        batch_size = logits.shape[0]
        label_ids = self.reshape(label_ids, (-1,))

        indices_1 = Tensor(np.arange(0, batch_size, 2), mstype.int32)
        indices_2 = Tensor(np.arange(1, batch_size, 2), mstype.int32)

        logits_1 = self.gather(logits, indices_1, 0)
        logits_2 = self.gather(logits, indices_2, 0)

        ce_loss = self.cross_entropy_loss(logits, label_ids)

        kl_loss_1 = self.kl_div(self.log_softmax(logits_1), self.softmax(logits_2))
        kl_loss_2 = self.kl_div(self.log_softmax(logits_2), self.softmax(logits_1))
        kl_loss = (kl_loss_1 + kl_loss_2).mean() / 4 * alpha
        return_value = ce_loss + kl_loss
        return return_value


def rdrop_repeat(*items):
    res = []
    for item in items:
        res.append(item.repeat(2, axis=0))
    return tuple(res)
