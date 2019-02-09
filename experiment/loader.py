#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: loader.py
# Author: Qian Ge <geqian1001@gmail.com>


import platform
import numpy as np
import sys
sys.path.append('../')
from src.dataflow.mnist import MNISTData


def loadMNIST(data_dir='', sample_per_class=12):
    def normalize_im(im):
        return np.clip(im/255.0, 0., 1.)

    train_data = MNISTData(
        'train',
        n_class=10,
        data_dir=data_dir,
        batch_dict_name=['im', 'label'],
        shuffle=True,
        pf=normalize_im)
    train_data.setup(epoch_val=0, sample_n_class=10, sample_per_class=sample_per_class)

    valid_data = MNISTData(
        'test',
        n_class=10,
        data_dir=data_dir,
        batch_dict_name=['im', 'label'],
        shuffle=True,
        pf=normalize_im)
    valid_data.setup(epoch_val=0, sample_n_class=10, sample_per_class=sample_per_class)

    return train_data, valid_data