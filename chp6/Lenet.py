# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/5 14:45
@Auth ： chenxi
@File ：Lenet.py
@IDE ：PyCharm
@Motto：Stay hungry, Stay young.
"""
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import time


net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),  # 28 - kernel_size=5 + padding*2 +1 = 28
    nn.AvgPool2d(kernel_size=2, stride=2),  # (28 - 2 + 2) / 2 = 14 * 14
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),  # (14 - 5 + 1) = 10 * 10
    nn.AvgPool2d(kernel_size=2, stride=2),  # (10 - 2 + 2) / 2 = 5
    nn.Flatten(),
    nn.Linear(16*5*5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)






def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_lenet`"""
    def init_weights(m):  # 初始化权重
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()  # 开始计算时间
            optimizer.zero_grad()  # 梯度清零
            X, y = X.to(device), y.to(device)
            y_hat = net(X)  # 计算预测值
            # print(f'train_y_hat:{y_hat}')
            l = loss(y_hat, y)  # 计算损失函数
            l.backward()  # 反向传播
            optimizer.step()  # 参数更新
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]  # 一个batch的损失函数
            train_acc = metric[1] / metric[2]  # 一个batch的训练准确率
            print(f'metric[1]:{metric[1]}')
            print(f'train_acc:{train_acc}')
            # break
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        # break
        # x_first_Sigmoid_layer = net[0:2](X)[0:9, 1, :, :]  # batch_size, channels, height, width
        # d2l.show_images(x_first_Sigmoid_layer.reshape(9, 28, 28).cpu().detach(), 1, 9, title=f'epoch-{epoch+1}-softmax1')
        # x_second_Sigmoid_layer = net[0:5](X)[0:9, 1, :, :]
        # d2l.show_images(x_second_Sigmoid_layer.reshape(9, 10, 10).cpu().detach(), 1, 9, title=f'epoch-{epoch+1}-softmax2')
        # # d2l.plt.show()
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

lr, num_epochs = 0.9, 3
train_ch6(net, train_iter, test_iter, num_epochs, lr, 'mps')
