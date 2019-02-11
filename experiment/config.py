#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: config.py
# Author: Qian Ge <geqian1001@gmail.com>


import platform


if platform.node() == 'Qians-MacBook-Pro.local':
    mnist_dir = '/Users/gq/workspace/Dataset/MNIST_data'
    mars_dir = '/Users/gq/workspace/Dataset/MARS/'
    market_dir = '/Users/gq/workspace/Dataset/market/Market-1501-v15.09.15/'

    market_save_path = '/Users/gq/workspace/GitHub/triplet-loss/lib/'
elif platform.node() == 'arostitan':
    mnist_dir = '/home/qge2/workspace/data/MNIST_data/'
    mnist_save_path = '/home/qge2/workspace/data/out/triplet/mnist/'
    mars_save_path = '/home/qge2/workspace/data/out/triplet/mars/'
    mars_dir = '/home/qge2/workspace/data/dataset/MARS/'

    market_dir = '/home/qge2/workspace/data/dataset/market/Market-1501-v15.09.15/'
    market_save_path = '/home/qge2/workspace/data/out/triplet/market/'
elif platform.node() == 'aros04':
    market_dir = 'E:/Dataset/market/Market-1501-v15.09.15/'
    market_save_path = 'E:/GITHUB/workspace/triplet/'
else:
    raise ValueError('No data dir setup on this platform!')