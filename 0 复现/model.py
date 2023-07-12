# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/10 11:08
@Auth ： chenxi
@File ：model.py.py
@IDE ：PyCharm
@Motto：Stay hungry, Stay young.
"""
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, *params):
        """
        1. 传入词向量
        2. 并行机制 textcnn-attention
                   lstm-attention
        3. 全连接层
        4. 隐藏层
        5. softmax
        """
