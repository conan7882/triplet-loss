#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: market.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import gzip
import struct
import ntpath
import numpy as np 
import src.utils.dataflow as dfutil
from src.dataflow.base import DataflowBaseTriplet, DataflowBaseChild, DataflowBase


def parse_filename(file_path_list):
    pid_list = []
    camera_id_list = []
    for idx, file_path in enumerate(file_path_list):
        head, tail = ntpath.split(file_path)
        pid = tail.split('_')[0]
        camera_id = tail.split('_')[1][:2]

        pid_list.append(pid)
        camera_id_list.append(camera_id)
    return np.array(pid_list), np.array(camera_id_list)



def distractor_IDs(file_path_list):
    IDs = []
    for idx, file_path in enumerate(file_path_list):
        head, tail = ntpath.split(file_path)
        label = tail.split('_')[0]
        if label == '0000' or label == '-1':
            IDs.append(idx)

    return IDs

class Market(DataflowBase):
    def _load_data(self):
        im_list = np.array(sorted(dfutil.get_file_list(self._data_dir, 'jpg')))
        label_list = [0 for _ in range(len(im_list))]

        return im_list, label_list

    def _read_batch_im(self, batch_file_list):
        
        im_list = []
        for file_name in batch_file_list:
            im = dfutil.load_image(file_name, read_channel=3, pf=self._pf)
            im_list.append(im)
        return im_list

class MarketTriplet(DataflowBaseTriplet):
    def __init__(self,
                 n_class=None,
                 data_dir='',
                 batch_dict_name=None,
                 shuffle=True,
                 pf=dfutil.identity):


        n_class = 10 # temp

        super(MarketTriplet, self).__init__(
            n_class=n_class,
            data_dir=data_dir,
            batch_dict_name=batch_dict_name,
            shuffle=shuffle,
            pf=pf)

    def _prepare_data(self):
        all_im_list = dfutil.get_file_list(self._data_dir, '.jpg')
        class_dict = {}
        class_cnt = 0

        for im_path in all_im_list:
            head, tail = ntpath.split(im_path)
            class_label = tail.split('_')[0]
            if class_label != '0000' and class_label != '-1':
                if class_label in class_dict:
                    class_id = class_dict[class_label]
                    self.im_list[class_id].append(im_path)
                else:
                    class_dict[class_label] = class_cnt
                    try:
                        self.im_list.append([im_path])
                    except AttributeError:
                        self.im_list = [[]]
                    class_cnt += 1

        self._n_class = len(self.im_list)
        for idx, class_im_list in enumerate(self.im_list):
            self.im_list[idx] = np.array(self.im_list[idx])

        self._shuffle_files()

    def next_batch(self):
        class_id_list = np.arange(self._n_class)
        self.rng.shuffle(class_id_list)
        pick_classes = class_id_list[:self._sample_n_class]

        batch_im_all = []
        batch_label_all = []
        for class_id in pick_classes:
            start = self._image_id[class_id]
            self._image_id[class_id] += self._sample_per_class
            end = self._image_id[class_id]
            batch_im = self._read_batch_im(self.im_list[class_id][start:end])
            # batch_im = list(self.im_list[class_id][start:end])
            batch_label = [class_id for _ in range(len(batch_im))]
            # batch_label = class_id * np.ones(len(batch_im))

            batch_im_all += batch_im
            batch_label_all += batch_label

            if self._image_id[class_id] > self.size(class_id):
                self._epochs_completed[class_id] += 1
                self._image_id[class_id] = 0
                self._shuffle_files(class_id)

        return [np.array(batch_im_all), np.array(batch_label_all)]

    def _read_batch_im(self, batch_file_list):

        im_list = []
        for file_name in batch_file_list:
            im = dfutil.load_image(file_name, read_channel=3, pf=self._pf)
            im_list.append(im)

        return im_list
