# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/5 15:51
@Auth ： chenxi
@File ：plotTest.py
@IDE ：PyCharm
@Motto：Stay hungry, Stay young.
"""
import matplotlib.pyplot as plt
from d2l import torch as d2l
import torch
y_hat = torch.tensor([[ 0.7739, -0.2810,  0.4911,  0.8867,  1.1168, -0.5165],
        [ 0.7744, -0.2805,  0.4905,   0.8862,  1.1187, -0.5169],
        [ 0.7734, -0.2806,  0.4905,  0.8868,  1.1170, -0.5176]])
y_hat = y_hat.to('cpu')
y_hat = d2l.argmax(y_hat, axis=1)
y_hat = y_hat.to('mps')
print(y_hat)