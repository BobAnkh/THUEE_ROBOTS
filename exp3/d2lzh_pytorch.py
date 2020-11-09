# import numpy as np
import random
import time

import torch
import torchvision
import torchvision.transforms as transforms
from IPython import display
from matplotlib import pyplot as plt
from torch import nn


def use_svg_display():
    '''
    用矢量图显示
    '''
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    '''
    设置矢量图的尺寸

    Keyword Arguments:
        figsize {tuple} -- 图形尺寸 (default: {(3.5, 2.5)})
    '''
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def data_iter(batch_size, features, labels):
    '''
    迭代读取样本

    Arguments:
        batch_size {interger} -- 样本批大小
        features {tensor} -- 特征
        labels {tensor} -- 标签

    Yields:
        tensor -- 返回对应的特征和标签
    '''
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(
            indices[i:min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)
        # yield会返回值，并让下一次调用这个函数的时候从此处接着运行
        # index_select()函数第一个参数是查找索引的维度，第二个参数是索引列表


def linreg(X, w, b):
    return torch.mm(X, w) + b  # 用mm函数做矩阵运算


def squared_loss(y_hat, y):
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size()))**2 / 2


def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


def get_fashion_mnist_labels(
        labels):  # Fashion-MNIST中一共包括了10个类别，以下函数可以将数值标签转成相应的文本标签。
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    '''
    在一行中画出多张图像和对应标签的函数
    '''
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def load_data_fashion_mnist(batch_size, num_workers=0):
    '''
    读取数据集

    Arguments:
        batch_size {int} -- the size of a batch

    Keyword Arguments:
        num_workers {int} -- number of workers (default: {0})

    Returns:
        DataLoader --
    '''
    mnist_train = torchvision.datasets.FashionMNIST(
        root='~/Datasets/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(
        root='~/Datasets/FashionMNIST',
        train=False,
        download=True,
        transform=transforms.ToTensor())
    train_iter = torch.utils.data.DataLoader(mnist_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter


def softmax(X):
    '''
    softmax function

    Args:
        X (Tensor): number of its columns is the number of batch_size,
        the number of its rows is the number of output

    Returns:
        Tensor
    '''
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


def cross_entropy(y_hat, y):
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def train_ch3(net,
              train_iter,
              test_iter,
              loss,
              num_epochs,
              batch_size,
              params=None,
              lr=None,
              optimizer=None):
    '''
    Train softmax model

    Args:
        net (Tensor): the primary model of softmax
        train_iter (Dataloader):
        test_iter (Dateloader):
        loss(): cross_entropy function
        num_epochs (int):
        batch_size (int):
        params (list, optional): Defaults to None.
        lr (float, optional): Defaults to None.
        optimizer (NoneType, optional): Defaults to None.
    '''
    for epoch in range(num_epochs):
        start_time = time.time()
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            loss_sum = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            loss_sum.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”
            train_l_sum += loss_sum.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print(
            'epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time_use %.3f'
            % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
               time.time() - start_time))


class FlattenLayer(nn.Module):
    '''
    Change input to flatten
    '''
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)
