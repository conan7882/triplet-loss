#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: loader.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import platform
import numpy as np
import sys
import skimage.transform
sys.path.append('../')
from src.dataflow.mnist import MNISTData
from src.dataflow.mars import MARSTriplet, MARSChild
from src.dataflow.testset import prepare_query_gallery


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

def loadMARS(data_dir='', sample_per_class=4, rescale_im=[128, 64]):
    def normalize_im(im):
        im = skimage.transform.resize(
            im, rescale_im,
            mode='constant', preserve_range=True)

        return np.clip(im/255.0, 0., 1.)

    train_dir = os.path.join(data_dir, 'bbox_train')
    train_data = MARSTriplet(
        n_class=None,
        data_dir=train_dir,
        batch_dict_name=['im', 'label'],
        shuffle=True,
        pf=normalize_im)
    train_data.setup(epoch_val=0, sample_n_class=32, sample_per_class=sample_per_class)

    valid_dir = os.path.join(data_dir, 'bbox_test')
    # valid_data = MARSTriplet(
    #     'test',
    #     n_class=None,
    #     data_dir=valid_dir,
    #     batch_dict_name=['im', 'label'],
    #     shuffle=True,
    #     pf=normalize_im)
    # valid_data.setup(epoch_val=0, sample_n_class=32, sample_per_class=sample_per_class)

    query_data, gallery_data = prepare_query_gallery(
        TripletDataFlow=MARSTriplet,
        ChildDataFlow=MARSChild,
        data_dir=valid_dir,
        batch_dict_name=['im', 'label'],
        shuffle=True,
        pf=normalize_im,
        query_ratio=0.3)

    return train_data, query_data, gallery_data
    