#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist.py
# Author: Qian Ge <geqian1001@gmai.com>

import numpy as np
import tensorflow as tf
import src.models.layers as L
from src.models.base import BaseModel
from src.triplet.hard_mining import hard_triplet_loss
from src.utils.viz import display


INIT_W = tf.keras.initializers.he_normal()
WD = 0

class MetricNet(BaseModel):
    def __init__(self, im_size, n_channels, embedding_dim, margin):
        """
        Args:
            im_size (int or list with length 2): size of generate image 
            n_channels (int): number of image channels
        """
        im_size = L.get_shape2D(im_size)
        self.im_h, self.im_w = im_size
        self.n_channels = n_channels
        self.embedding_dim = embedding_dim
        self.margin = margin

        self.layers = {}

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

    def _create_inference_input(self):
        """ input for inference """
        self.image = tf.placeholder(
            tf.float32, 
            [None, self.im_h, self.im_w, self.n_channels],
            name='image')
        # self.label = tf.placeholder(tf.int64, [None], name='label')
        # self.lr = tf.placeholder(tf.float32, name='lr')
        # self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def create_inference_model(self):
        """ create graph for inference """
        self.set_is_training(False)
        self._create_inference_input()
        self.embedding = self._creat_model(self.image)

        self.epoch_id = 0
        # self.loss_op = self.get_loss()

    def _get_loss(self):
        with tf.name_scope('loss'):
            loss = hard_triplet_loss(self.embedding, self.label, self.margin)
            return loss

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(self.lr)
        
    def _creat_model(self, inputs):
        self.layers['cur_input'] = inputs

        with tf.variable_scope('embedding_net', reuse=tf.AUTO_REUSE):
            arg_scope = tf.contrib.framework.arg_scope
            with arg_scope([L.conv, L.linear],
                           layer_dict=self.layers, bn=True, init_w=INIT_W,
                           is_training=self.is_training, wd=WD):

                L.conv(filter_size=5, out_dim=32, nl=tf.nn.relu, padding='SAME', name='conv1')
                L.max_pool(layer_dict=self.layers, padding='SAME', name='max_pool1')

                L.conv(filter_size=5, out_dim=64, nl=tf.nn.relu, padding='SAME', name='conv2')
                L.max_pool(layer_dict=self.layers, padding='SAME', name='max_pool2')

                L.linear(out_dim=256, name='linear1', nl=tf.nn.relu,)
                L.linear(out_dim=self.embedding_dim, bn=False, name='linear2')

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
                display(
                    global_step=self.global_step,
                    step=step,
                    scaler_sum_list=[loss_sum],
                    name_list=display_name_list,
                    collection='train',
                    summary_val=cur_summary,
                    summary_writer=summary_writer)

        print('==== epoch: {}, lr:{} ===='.format(cur_epoch, lr))
        display(
            global_step=self.global_step,
            step=step,
            scaler_sum_list=[loss_sum],
            name_list=display_name_list,
            collection='train',
            summary_val=cur_summary,
            summary_writer=summary_writer)

    def inference_epoch(self, sess, dataflow, save_path=None):
        dataflow.reset_epochs_completed()
        embedding_all = np.empty((0, self.embedding_dim))
        self.epoch_id += 1
        # loss_sum = 0
        step = 0
        while dataflow.epochs_completed <= 1:
            step += 1
            batch_data = dataflow.next_batch_dict()
            embedding = sess.run(
                self.embedding,
                feed_dict={self.image: batch_data['im']})
            # print(batch_data['label'])
            embedding_all = np.concatenate((embedding_all, embedding), axis=0)
            # loss_sum += loss

        # print(loss_sum / step)
        if save_path is not None and self.embedding_dim == 2:
            import os
            import matplotlib as mpl
            mpl.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure()
            plt.scatter(embedding_all[:,0], embedding_all[:,1], s=2)
            save_dir = os.path.join(save_path, 'embedding_{}'.format(self.epoch_id))
            plt.savefig(save_dir, bbox_inches="tight")
            plt.close()



