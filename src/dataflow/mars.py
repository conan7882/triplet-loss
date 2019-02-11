#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mars.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import gzip
import struct
import numpy as np 
import src.utils.dataflow as dfutil
from src.dataflow.base import DataflowBaseTriplet, DataflowBaseChild, DataflowBase


class MARS(DataflowBase):
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


class MARSChild(DataflowBaseChild):
    def _read_batch_im(self, batch_file_list):
        im_list = []
        for file_name in batch_file_list:
            im = dfutil.load_image(file_name, read_channel=3, pf=self._pf)
            im_list.append(im)

        return im_list


class MARSTriplet(DataflowBaseTriplet):
    def __init__(self,
                 n_class=None,
                 data_dir='',
                 batch_dict_name=None,
                 shuffle=True,
                 pf=dfutil.identity):


        n_class = 10 # temp

        super(MARSTriplet, self).__init__(
            n_class=n_class,
            data_dir=data_dir,
            batch_dict_name=batch_dict_name,
            shuffle=shuffle,
            pf=pf)

    def _prepare_data(self):
        class_folder = [f_name for f_name in os.listdir(self._data_dir) if os.path.isdir(os.path.join(self._data_dir, f_name))]
        self._n_class = len(class_folder)

        self.im_list = [[] for _ in range(self._n_class)]
        for label_id, folder_name in enumerate(class_folder):
            file_list = dfutil.get_file_list(os.path.join(self._data_dir, folder_name), '.jpg')
            self.im_list[label_id] = np.array(file_list)

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

