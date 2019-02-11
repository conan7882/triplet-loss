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
from src.dataflow.mars import MARSTriplet, MARSChild, MARS
from src.dataflow.testset import prepare_query_gallery
from src.dataflow.market import MarketTriplet, Market


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

def loadMARSTrain(data_dir='', sample_per_class=4, rescale_im=[128, 64], batch_size=256):
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

    # valid_dir = os.path.join(data_dir, 'bbox_test')
    # # valid_data = MARSTriplet(
    # #     'test',
    # #     n_class=None,
    # #     data_dir=valid_dir,
    # #     batch_dict_name=['im', 'label'],
    # #     shuffle=True,
    # #     pf=normalize_im)
    # # valid_data.setup(epoch_val=0, sample_n_class=32, sample_per_class=sample_per_class)

    # query_data, gallery_data = prepare_query_gallery(
    #     TripletDataFlow=MARSTriplet,
    #     ChildDataFlow=MARSChild,
    #     data_dir=valid_dir,
    #     batch_dict_name=['im', 'label', 'filename'],
    #     batch_size=batch_size,
    #     shuffle=True,
    #     pf=normalize_im,
    #     query_ratio=0.3)

    return train_data

def loadMARSInference(data_dir='', rescale_im=[128, 64], batch_size=1):

    def normalize_im(im):
        im = skimage.transform.resize(
            im, rescale_im,
            mode='constant', preserve_range=True)
        return np.clip(im/255.0, 0., 1.)

    test_dir = os.path.join(data_dir, 'bbox_test')
    dataflow = MARS(data_dir=test_dir,
                    batch_dict_name=['im', 'label'],
                    shuffle=False,
                    pf=normalize_im)
    dataflow.setup(epoch_val=0, batch_size=batch_size)

    return dataflow

def loadMarketTrain(data_dir='', sample_per_class=4, rescale_im=[128, 64]):
    def normalize_im(im):
        im = skimage.transform.resize(
            im, rescale_im,
            mode='constant', preserve_range=True)
        return np.clip(im/255.0, 0., 1.)

    train_dir = os.path.join(data_dir, 'bounding_box_train')
    train_data = MarketTriplet(
        n_class=None,
        data_dir=train_dir,
        batch_dict_name=['im', 'label'],
        shuffle=True,
        pf=normalize_im)
    train_data.setup(epoch_val=0, sample_n_class=32, sample_per_class=sample_per_class)
    return train_data

def loadMarketInference(data_dir='', rescale_im=[128, 64], batch_size=1):

    def normalize_im(im):
        im = skimage.transform.resize(
            im, rescale_im,
            mode='constant', preserve_range=True)
        return np.clip(im/255.0, 0., 1.)

    train_dir = os.path.join(data_dir, 'bounding_box_train')
    train_data = Market(
        data_dir=train_dir,
        batch_dict_name=['im', 'label'],
        shuffle=False,
         pf=normalize_im)
    train_data.setup(epoch_val=0, batch_size=batch_size)

    test_dir = os.path.join(data_dir, 'bounding_box_test')
    test_data = Market(
        data_dir=test_dir,
        batch_dict_name=['im', 'label'],
        shuffle=False,
        pf=normalize_im)
    test_data.setup(epoch_val=0, batch_size=batch_size)

    return train_data, test_data



    