#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: lunet.py
# Author: Qian Ge <geqian1001@gmai.com>

import numpy as np
import tensorflow as tf
import src.models.layers as L
import src.models.resblock as resblock
from src.models.base import BaseModel
from src.triplet.hard_mining import batch_hard_triplet_loss
import src.utils.viz as viz



INIT_W = tf.keras.initializers.he_normal()
WD = 0

class LuNet(BaseModel):
    def __init__(self, im_size, n_channels, embedding_dim, margin):
        im_size = L.get_shape2D(im_size)
        self.im_h, self.im_w = im_size
        self.n_channels = n_channels
        self.embedding_dim = embedding_dim
        self.margin = margin

        self.layers = {}
        pass

    def _create_train_input(self):
        """ input for training """
        self.image = tf.placeholder(
            tf.float32, 
            [None, self.im_h, self.im_w, self.n_channels],
            name='image')
        self.label = tf.placeholder(tf.int64, [None], name='label')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def create_train_model(self):
        """ create graph for training """
        self.set_is_training(True)
        self._create_train_input()
        self.embedding = self._creat_model(self.image)

        self.train_op = self.get_train_op()
        self.loss_op = self.get_loss()

        self.global_step = 0
        self.epoch_id = 0

    def _get_loss(self):
        with tf.name_scope('loss'):
            loss = batch_hard_triplet_loss(self.embedding, self.label, self.margin)
            return loss

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(self.lr)

    def _create_model(self, inputs):
        with tf.variable_scope('LuNet', reuse=tf.AUTO_REUSE):
            self.layers['cur_input'] = inputs

            arg_scope = tf.contrib.framework.arg_scope()
            with arg_scope([L.conv, L.linear, resblock.res_block_bottleneck, resblock.res_block],
                           layer_dict=self.layers, bn=True, init_w=INIT_W,
                           is_training=self.is_training, wd=WD):

                with tf.variable_scope('block_1'):
                    L.conv(filter_size=7, out_dim=128, nl=L.leaky_relu, padding='SAME', name='conv1')
                    resblock.res_block_bottleneck(128, 32, 128, name='resblock1')
                    L.max_pool(filter_size=3, stride=2, padding='SAME', name='max_pool1')

                with tf.variable_scope('block_2'):
                    resblock.res_block_bottleneck(128, 32, 128, name='resblock1')
                    resblock.res_block_bottleneck(128, 32, 128, name='resblock2')
                    resblock.res_block_bottleneck(128, 64, 256, name='resblock3')
                    L.max_pool(filter_size=3, stride=2, padding='SAME', name='max_pool1')

                with tf.variable_scope('block_3'):
                    resblock.res_block_bottleneck(256, 64, 256, name='resblock1')
                    resblock.res_block_bottleneck(256, 64, 256, name='resblock2')
                    L.max_pool(filter_size=3, stride=2, padding='SAME', name='max_pool1')

                with tf.variable_scope('block_4'):
                    resblock.res_block_bottleneck(256, 64, 256, name='resblock1')
                    resblock.res_block_bottleneck(256, 64, 256, name='resblock2')
                    resblock.res_block_bottleneck(256, 128, 512, name='resblock3')
                    L.max_pool(filter_size=3, stride=2, padding='SAME', name='max_pool1')

                with tf.variable_scope('block_5'):
                    resblock.res_block_bottleneck(512, 128, 512, name='resblock1')
                    resblock.res_block_bottleneck(512, 128, 512, name='resblock2')
                    L.max_pool(filter_size=3, stride=2, padding='SAME', name='max_pool1')

                with tf.variable_scope('block_6'):
                    resblock.res_block(n1=512, n2=128, name='res_block1')
                    L.linear(out_dim=512, nl=L.leaky_relu, name='linear1')
                    L.linear(out_dim=128, name='linear2')

            return self.layers['cur_input']

    def train_epoch(self, sess, train_data, lr, summary_writer=None):
        self.epoch_id += 1
        display_name_list = ['loss']
        cur_epoch = train_data.epochs_completed

        cur_summary = None
        step = 0
        while train_data.epochs_completed <= cur_epoch:
            step += 1
            self.global_step += 1

            loss_sum = 0 
            batch_data = train_data.next_batch_dict()
            _, loss = sess.run(
                [self.train_op, self.loss_op],
                feed_dict={self.lr:lr, self.image: batch_data['im'],
                           self.label: batch_data['label']})

            loss_sum += loss

            if step % 100 == 0:
                viz.display(
                    global_step=self.global_step,
                    step=step,
                    scaler_sum_list=[loss_sum],
                    name_list=display_name_list,
                    collection='train',
                    summary_val=cur_summary,
                    summary_writer=summary_writer)

        print('==== epoch: {}, lr:{} ===='.format(cur_epoch, lr))
        viz.display(
            global_step=self.global_step,
            step=step,
            scaler_sum_list=[loss_sum],
            name_list=display_name_list,
            collection='train',
            summary_val=cur_summary,
            summary_writer=summary_writer)



