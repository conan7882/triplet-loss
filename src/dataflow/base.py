#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import numpy as np 
import src.utils.dataflow as dfutil


class DataflowBase(object):
    def __init__(self,
                 data_dir,
                 batch_dict_name=None,
                 shuffle=True,
                 pf=dfutil.identity):

        self._data_dir = data_dir
        self._shuffle = shuffle
        self._pf = pf

        if not isinstance(batch_dict_name, list):
            batch_dict_name = [batch_dict_name]
        self._batch_dict_name = batch_dict_name

        self.rng = dfutil.get_rng(self)
        self._prepare_data()
        self.setup(epoch_val=0, batch_size=1)

    def _load_data(self):
        raise NotImplementedError()

    def _prepare_data(self):
        im_list, label_list = self._load_data()
        self.im_list = np.array(im_list)
        self.label_list = np.array(label_list)
        self._shuffle_files()

    def next_batch_dict(self):
        batch_data = self.next_batch()
        data_dict = {key: data for key, data
                     in zip(self._batch_dict_name, batch_data)}
        return data_dict

    def _shuffle_files(self):
        if self._shuffle:
            n_sample = self.size()
            sample_idxs = np.arange(n_sample)
            self.rng.shuffle(sample_idxs)
            self.im_list = self.im_list[sample_idxs]
            self.label_list = self.label_list[sample_idxs]

    def setup(self, epoch_val, batch_size, **kwargs):
        self._image_id = 0
        self._epochs_completed = epoch_val
        self._batch_size = batch_size
        assert self._batch_size <= self.size(), \
            "batch_size {} cannot be larger than data size {}".\
            format(self._batch_size, self.size())
        try:
            self._shuffle_files()
        except AttributeError:
            pass

    def size(self):
        return len(self.im_list)

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def reset_epochs_completed(self):
        self._image_id = 0
        self._epochs_completed = 0

    def next_batch(self):

        start = self._image_id
        self._image_id += self._batch_size
        end = self._image_id
        batch_im = self._read_batch_im(self.im_list[start:end])
        batch_label = self.label_list[start:end]
        batch_file_name = self.im_list[start:end]

        if self._image_id > self.size():
            self._epochs_completed += 1
            self._image_id = 0
            self._shuffle_files()

        return [np.array(batch_im), np.array(batch_label), np.array(batch_file_name)]

    def _read_batch_im(self, batch_file_list):
        raise NotImplementedError

    @property
    def epoch_step(self):
        return int(self.size() / self._batch_size)

class DataflowBaseChild(DataflowBase):
    def __init__(self,
                 im_list,
                 label_list,
                 batch_dict_name=None,
                 shuffle=True,
                 pf=dfutil.identity):

        self._shuffle = shuffle
        self._pf = pf

        if not isinstance(batch_dict_name, list):
            batch_dict_name = [batch_dict_name]
        self._batch_dict_name = batch_dict_name

        self.rng = dfutil.get_rng(self)
        self._prepare_data(im_list, label_list)
        self.setup(epoch_val=0, batch_size=1)

    def _prepare_data(self, im_list, label_list):
        assert len(im_list) == len(label_list)
        self.im_list = np.array(im_list)
        self.label_list = np.array(label_list)
        self._shuffle_files()

    # def next_batch_dict(self):
    #     batch_data = self.next_batch()
    #     data_dict = {key: data for key, data
    #                  in zip(self._batch_dict_name, batch_data)}
    #     return data_dict

    

    # def _shuffle_files(self):
    #     if self._shuffle:
    #         n_sample = self.size()
    #         sample_idxs = np.arange(n_sample)
    #         self.rng.shuffle(sample_idxs)
    #         self.im_list = self.im_list[sample_idxs]
    #         self.label_list = self.label_list[sample_idxs]

    # def setup(self, epoch_val, batch_size, **kwargs):
    #     self._image_id = 0
    #     self._epochs_completed = epoch_val
    #     self._batch_size = batch_size
    #     assert self._batch_size <= self.size(), \
    #         "batch_size {} cannot be larger than data size {}".\
    #         format(self._batch_size, self.size())
    #     try:
    #         self._shuffle_files()
    #     except AttributeError:
    #         pass

    # def size(self):
    #     return len(self.im_list)

    # @property
    # def epochs_completed(self):
    #     return self._epochs_completed

    # def reset_epochs_completed(self):
    #     self._image_id = 0
    #     self._epochs_completed = 0

    # def next_batch(self):

    #     start = self._image_id
    #     self._image_id += self._batch_size
    #     end = self._image_id
    #     batch_im = self._read_batch_im(self.im_list[start:end])
    #     batch_label = self.label_list[start:end]
    #     batch_file_name = self.im_list[start:end]

    #     if self._image_id > self.size():
    #         self._epochs_completed += 1
    #         self._image_id = 0
    #         self._shuffle_files()

    #     return [np.array(batch_im), np.array(batch_label), np.array(batch_file_name)]

    # def _read_batch_im(self, batch_file_list):
    #     raise NotImplementedError

    # @property
    # def epoch_step(self):
    #     return int(self.size() / self._batch_size)


class DataflowBaseTriplet(object):
    def __init__(self,
                 n_class=None,
                 data_dir='',
                 batch_dict_name=None,
                 shuffle=True,
                 pf=dfutil.identity):

        assert os.path.isdir(data_dir), 'Invalid path {}'.format(data_dir)
        self._data_dir = data_dir
        self._n_class = n_class

        self._shuffle = shuffle
        self._pf = pf

        if not isinstance(batch_dict_name, list):
            batch_dict_name = [batch_dict_name]
        self._batch_dict_name = batch_dict_name

        self.rng = dfutil.get_rng(self)
        self._prepare_data()
        self.setup(epoch_val=0, sample_n_class=2, sample_per_class=1)
        
    def _load_data(self):
        return im_list, label_list

    def next_batch_dict(self):
        batch_data = self.next_batch()
        data_dict = {key: data for key, data
                     in zip(self._batch_dict_name, batch_data)}
        return data_dict

    def _prepare_data(self):
        im_list, label_list = self._load_data()

        self.im_list = [[] for _ in range(self._n_class)]
        for im, label in zip(im_list, label_list):
            self.im_list[label].append(im)

        for idx, im_list in enumerate(self.im_list):
            self.im_list[idx] = np.array(im_list)

        self._shuffle_files()

    def _shuffle_files(self, class_id=None):
        if self._shuffle:
            if class_id is not None and class_id < self._n_class:
                n_sample = len(self.im_list[class_id])
                sample_idxs = np.arange(n_sample)
                self.rng.shuffle(sample_idxs)
                self.im_list[class_id] = self.im_list[class_id][sample_idxs]
            else:
                for idx, im_list in enumerate(self.im_list):
                    n_sample = len(self.im_list[idx])
                    sample_idxs = np.arange(n_sample)
                    self.rng.shuffle(sample_idxs)
                    self.im_list[idx] = self.im_list[idx][sample_idxs]

    def size(self, class_id):
        return self.im_list[class_id].shape[0]

    def setup(self, epoch_val, sample_n_class, sample_per_class, **kwargs):
        self._image_id = [0 for _ in range(self._n_class)]
        self._epochs_completed = [epoch_val for _ in range(self._n_class)]
        # self._iteration_completed = iteration_val
        self._sample_n_class = sample_n_class
        self._sample_n_class = np.minimum(self._sample_n_class, self._n_class)
        self._sample_per_class = sample_per_class
        # for class_id in range(0, self._n_class):
        #     assert self._sample_per_class <= self.size(class_id), \
        #         "sample_per_class {} cannot be larger than data size {}".\
        #         format(self._sample_per_class, self.size(class_id))
        try:
            self._shuffle_files()
        except AttributeError:
            pass

    @property
    def epochs_completed(self):
        return min(self._epochs_completed)

    def reset_epochs_completed(self):
        self._image_id = [0 for _ in range(self._n_class)]
        self._epochs_completed = [0 for _ in range(self._n_class)]

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
            batch_im = list(self.im_list[class_id][start:end])
            batch_label = [class_id for _ in range(len(batch_im))]
            # batch_label = class_id * np.ones(len(batch_im))

            batch_im_all += batch_im
            batch_label_all += batch_label

            if self._image_id[class_id] > self.size(class_id):
                self._epochs_completed[class_id] += 1
                self._image_id[class_id] = 0
                self._shuffle_files(class_id)

        return [np.array(batch_im_all), np.array(batch_label_all)]

