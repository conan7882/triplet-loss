#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resblock.py
# Author: Qian Ge <geqian1001@gmai.com>


import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope
import src.models.layers as L


# def leaky_relu(x):
#     return L.leaky_relu(x, leak=0.3, name='LeakyRelu')

@add_arg_scope
def res_block_bottleneck(n1, n2, n3, layer_dict, init_w=None, wd=0, bn=True, is_training=True, name='res_block_bottleneck'):
    # Deep residual learning for image recognition
    # bottleneck version (Fig5 right)
    with tf.variable_scope(name):
        inputs = layer_dict['cur_input']
        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.conv],
                       layer_dict=layer_dict, bn=bn, init_w=init_w,
                       is_training=is_training, wd=wd):
            L.conv(filter_size=1, out_dim=n2, nl=L.leaky_relu, padding='SAME', name='conv1')
            L.conv(filter_size=3, out_dim=n2, nl=L.leaky_relu, padding='SAME', name='conv2')
            L.conv(filter_size=1, out_dim=n3, padding='SAME', name='conv3')
            res_out = layer_dict['cur_input']

            if n1 != n3:
                inputs = L.conv(inputs=inputs, filter_size=1, out_dim=n3, padding='SAME', name='shortcut')

    outputs = inputs + res_out
    layer_dict['cur_input'] = L.leaky_relu(outputs)

    return layer_dict['cur_input']

@add_arg_scope
def res_block(layer_dict, n1=512, n2=128, init_w=None, wd=0, bn=True, is_training=True, name='res_block'):
    with tf.variable_scope(name):
        inputs = layer_dict['cur_input']
        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.conv],
                       layer_dict=layer_dict, bn=bn, init_w=init_w,
                       is_training=is_training, wd=wd):
            L.conv(filter_size=3, out_dim=n1, nl=L.leaky_relu, padding='SAME', name='conv1')
            L.conv(filter_size=3, out_dim=n2, padding='SAME', name='conv2')
            res_out = layer_dict['cur_input']

        if n1 != n2:
            inputs = L.conv(inputs=inputs, filter_size=1, out_dim=n2, padding='SAME', name='shortcut')

        outputs = inputs + res_out
        layer_dict['cur_input'] = L.leaky_relu(outputs)

        return layer_dict['cur_input']
