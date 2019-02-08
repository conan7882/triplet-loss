# usr/bin/env python
# -*- coding utf-8 -*-
# File: base.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import gzip
import numpy as np 
import utils.dataflow as dfutil

_RNG_SEED = None

def get_rng(obj=None):
    """
    This function is copied from `tensorpack
    <https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/utils.py>`__.
    Get a good RNG seeded with time, pid and the object.

    Args:
        obj: some object to use to generate random seed.

    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    if _RNG_SEED is not None:
        seed = _RNG_SEED
    return np.random.RandomState(seed)


def DataflowBase(object):
    def __init__(self,
                 n_class,
                 data_dir='',
                 batch_dict_name=None,
                 shuffle=True,
                 pf=dfutil.identity):

        assert os.path.isdir(data_dir)
        self._data_dir = data_dir
        self._n_class = n_class

        self._shuffle = shuffle
        self._pf = pf

        if not isinstance(batch_dict_name, list):
            batch_dict_name = [batch_dict_name]
        self._batch_dict_name = batch_dict_name

        self._prepare_data()
        self.setup(iteration_val=0, sample_n_class=2, sample_per_class=1)
        
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

        self._suffle_files()

    def _suffle_files(self):
        if self._shuffle:
            for idx, im_list in enumerate(self.im_list):
                n_sample = len(self.im_list[idx])
                sample_idxs = np.arange(n_sample)
                self.rng.shuffle(sample_idxs)
                self.im_list[idx] = self.im_list[idx][sample_idxs]

    def size(self, class_id):
        return self.im_list[class_id].shape[0]

    def setup(self, iteration_val, sample_n_class, sample_per_class, **kwargs):
        self._image_id = [0 for _ in range(self._n_class)]
        self._iteration_completed = iteration_val
        self._sample_n_class = sample_n_class
        self._sample_per_class = sample_per_class
        for class_id in range(0, self._n_class):
            assert self._sample_per_class <= self.size(class_id), \
                "batch_size {} cannot be larger than data size {}".\
                format(self._sample_per_class, self.size(class_id))
        
        self.rng = get_rng(self)
        try:
            self._suffle_files()
        except AttributeError:
            pass

    @property
    def iteration_completed(self):
        return self._iteration_completed

    def next_batch(self):
        pick_classes = np.random.choice(self._n_class, self._sample_n_class)

        start = self._image_id
        self._image_id += self._batch_size
        end = self._image_id
        batch_files = self.im_list[start:end]
        batch_label = self.label_list[start:end]

        if self._image_id + self._batch_size > self.size():
            self._epochs_completed += 1
            self._image_id = 0
            self._suffle_files()
        return [batch_files, batch_label]

