#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: viz.py
# Author: Qian Ge <geqian1001@gmail.com>

import imageio
import numpy as np
import tensorflow as tf


def viz_batch_im(batch_im, grid_size, save_path,
                 gap=0, gap_color=0, shuffle=False):
    """ save batch of image as a single image 

    Args:
        batch_im (list): list of images 
        grid_size (list of 2): size (number of samples in row and column) of saving image
        save_path (str): directory for saving sampled images
        gap (int): number of pixels between two images
        gap_color (int): color of gap between images
        shuffle (bool): shuffle batch images for saving or not
    """

    batch_im = np.array(batch_im)
    if len(batch_im.shape) == 4:
        n_channel = batch_im.shape[-1]
    elif len(batch_im.shape) == 3:
        n_channel = 1
        batch_im = np.expand_dims(batch_im, axis=-1)
    assert len(grid_size) == 2

    h = batch_im.shape[1]
    w = batch_im.shape[2]

    merge_im = np.zeros((h * grid_size[0] + (grid_size[0] + 1) * gap,
                         w * grid_size[1] + (grid_size[1] + 1) * gap,
                         n_channel)) + gap_color

    n_viz_im = min(batch_im.shape[0], grid_size[0] * grid_size[1])
    if shuffle == True:
        pick_id = np.random.permutation(batch_im.shape[0])
    else:
        pick_id = range(0, batch_im.shape[0])
    for idx in range(0, n_viz_im):
        i = idx % grid_size[1]
        j = idx // grid_size[1]
        cur_im = batch_im[pick_id[idx], :, :, :]
        merge_im[j * (h + gap) + gap: j * (h + gap) + h + gap,
                 i * (w + gap) + gap: i * (w + gap) + w + gap, :]\
            = (cur_im)
    imageio.imwrite(save_path, np.squeeze(merge_im))

def display(global_step,
            step,
            scaler_sum_list,
            name_list,
            collection,
            summary_val=None,
            summary_writer=None,
            ):
    """ Display averaged intermediate results for a period during training.

    The intermediate result will be displayed as:
    [step: global_step] name_list[0]: scaler_sum_list[0]/step ...
    Those result will be saved as summary as well.

    Args:
        global_step (int): index of current iteration
        step (int): number of steps for this period
        scaler_sum_list (float): list of summation of the intermediate
            results for this period
        name_list (str): list of display name for each intermediate result
        collection (str): list of graph collections keys for summary
        summary_val : additional summary to be saved
        summary_writer (tf.FileWriter): write for summary. No summary will be
            saved if None.
    """
    print('[step: {}]'.format(global_step), end='')
    for val, name in zip(scaler_sum_list, name_list):
        print(' {}: {:.4f}'.format(name, val * 1. / step), end='')
    print('')
    if summary_writer is not None:
        s = tf.Summary()
        for val, name in zip(scaler_sum_list, name_list):
            s.value.add(tag='{}/{}'.format(collection, name),
                        simple_value=val * 1. / step)
        summary_writer.add_summary(s, global_step)
        if summary_val is not None:
            summary_writer.add_summary(summary_val, global_step)
