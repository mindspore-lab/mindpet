#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
import mindspore as ms
import mindspore.nn as nn

from tk.utils.version_utils import is_version_ge

if is_version_ge(ms.__version__, '2.0.0'):
    import mindspore._checkparam as Validator

    INC_LEFT = Validator.INC_LEFT
else:
    from mindspore._checkparam import Validator, Rel

    INC_LEFT = Rel.INC_LEFT


class PromptTuning(nn.Cell):
    """Define a cell with PromptTuning structure.

    Attributes:
        num_virtual_tokens (int): The number of virtual tokens to use.
        token_dim (int): The hidden embedding dimension of the base model.
        num_transformer_submodules (int): The number of transformer submodules in the base model.
    """

    def __init__(self,
                 num_virtual_tokens: int,
                 token_dim: int,
                 num_transformer_submodules: int = 1):
        super().__init__()
        self.num_virtual_tokens = Validator.check_positive_int(num_virtual_tokens, int, "num_virtual_tokens")
        self.token_dim = Validator.check_positive_int(token_dim, int, "token_dim")
        self.num_transformer_submodules = Validator.check_positive_int(num_transformer_submodules, int,
                                                                       "num_transformer_submodules")
        self.total_virtual_tokens = self.num_virtual_tokens * self.num_transformer_submodules
        self.tk_delta_prompttuning_embedding = nn.Embedding(self.total_virtual_tokens, self.token_dim)

    def construct(self, indices):
        prompt_embeddings = self.tk_delta_prompttuning_embedding(indices)
        return prompt_embeddings
