#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: config.py
# Author: Qian Ge <geqian1001@gmail.com>


import platform


if platform.node() == 'Qians-MacBook-Pro.local':
    mnist_dir = '/Users/gq/workspace/Dataset/MNIST_data'
elif platform.node() == 'arostitan':
    mnist_dir = '/home/qge2/workspace/data/MNIST_data/'
    mnist_save_path = '/home/qge2/workspace/data/out/triplet/mnist/'
else:
    raise ValueError('No data dir setup on this platform!')