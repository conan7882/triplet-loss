#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dataflow.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
from datetime import datetime
import numpy as np
import imageio
# import scipy.misc

import src.utils.utils as utils

def vec2onehot(vec, n_class):
    vec = np.array(vec)
    one_hot = np.zeros((len(vec), n_class))
    one_hot[np.arange(len(vec)), vec] = 1
    return one_hot
    # a = np.array([1, 0, 3])
    # b = np.zeros((3, 4))
    # b[np.arange(3), a] = 1

def identity(inputs):
    return inputs

def fill_pf_list(pf_list, n_pf, fill_with_fnc=identity):
    """ Fill the pre-process function list.

    Args:
        pf_list (list): input list of pre-process functions
        n_pf (int): required number of pre-process functions 
        fill_with_fnc: function used to fill the list

    Returns:
        list of pre-process function
    """
    if pf_list == None:
        return [identity for i in range(n_pf)]

    new_list = []
    pf_list = utils.make_list(pf_list)
    for pf in pf_list:
        if not pf:
            pf = identity
        new_list.append(pf)
    pf_list = new_list

    if len(pf_list) > n_pf:
        raise ValueError('Invalid number of preprocessing functions')
    pf_list = pf_list + [fill_with_fnc for i in range(n_pf - len(pf_list))]
    return pf_list

def load_image(im_path, read_channel=None, pf=identity):
    """ Load one image from file and apply pre-process function.

    Args:
        im_path (str): directory of image
        read_channel (int): number of image channels. Image will be read
            without channel information if ``read_channel`` is None.
        pf: pre-process fucntion

    Return:
        image after pre-processed with size [heigth, width, channel]

    """

    if read_channel is None:
        im = imageio.imread(im_path)
    elif read_channel == 3:
        im = imageio.imread(im_path, as_gray=False, pilmode="RGB")
    else:
        im = imageio.imread(im_path, as_gray=True)

    if len(im.shape) < 3:
        im = pf(im)
        im = np.reshape(im, [im.shape[0], im.shape[1], 1])
    else:
        im = pf(im)

    return im

def get_file_list(file_dir, file_ext, sub_name=None):
    """ Get file list in a directory with sepcific filename and extension

    Args:
        file_dir (str): directory of files
        file_ext (str): filename extension
        sub_name (str): Part of filename. Can be None.

    Return:
        List of filenames under ``file_dir`` as well as subdirectories

    """
    re_list = []

    if sub_name is None:
        return np.array([os.path.join(root, name)
            for root, dirs, files in os.walk(file_dir) 
            for name in sorted(files) if name.lower().endswith(file_ext)])
    else:
        return np.array([os.path.join(root, name)
            for root, dirs, files in os.walk(file_dir) 
            for name in sorted(files)
            if name.lower().endswith(file_ext) and sub_name.lower() in name.lower()])

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
