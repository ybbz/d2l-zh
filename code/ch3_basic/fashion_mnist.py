#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from mxnet.gluon import data as gdata

mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)

print(len(mnist_train), len(mnist_test))