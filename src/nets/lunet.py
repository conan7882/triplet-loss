#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: lunet.py
# Author: Qian Ge <geqian1001@gmai.com>

import numpy as np
import tensorflow as tf
import progressbar
import src.models.layers as L
import src.models.resblock as resblock
from src.models.base import BaseModel
from src.triplet.hard_mining import batch_hard_triplet_loss
import src.utils.viz as viz


INIT_W = tf.keras.initializers.he_normal()
WD = 0

class LuNet(BaseModel):
    def __init__(self, im_size, n_channels, embedding_dim=128, margin=0.5):
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
        self.embedding = self._create_model(self.image)

        self.train_op = self.get_train_op()
        self.loss_op = self.get_loss()
        self.train_summary_op = self._get_summary('train')

        self.global_step = 0
        self.epoch_id = 0

    def _create_inference_input(self):
        """ input for inference """
        self.image = tf.placeholder(
            tf.float32, 
            [None, self.im_h, self.im_w, self.n_channels],
            name='image')
        # self.label = tf.placeholder(tf.int64, [None], name='label')

    def create_inference_model(self):
        """ create graph for inference """
        self.set_is_training(False)
        self._create_inference_input()
        self.embedding = self._create_model(self.image)

        self.global_step = 0

    def _get_loss(self):
        with tf.name_scope('loss'):
            loss, nonzero_ratio = batch_hard_triplet_loss(self.embedding, self.label, self.margin)
            self.nonzero_loss_ratio = nonzero_ratio
            return loss

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(self.lr)

    def _get_summary(self, collection):
        with tf.name_scope(collection):
            tf.summary.scalar('non_zero_losses', self.nonzero_loss_ratio, collections=[collection])
            tf.summary.histogram('embedding_norm', tf.norm(self.embedding, axis=-1), collections=[collection])
            tf.summary.histogram('embedding_element', self.embedding, collections=[collection])

            return tf.summary.merge_all(key=collection)

    def _create_model(self, inputs):
        with tf.variable_scope('LuNet', reuse=tf.AUTO_REUSE):
            self.layers['cur_input'] = inputs

            arg_scope = tf.contrib.framework.arg_scope
            with arg_scope([L.conv, L.linear, resblock.res_block_bottleneck, resblock.res_block],
                           layer_dict=self.layers, bn=True, init_w=INIT_W,
                           is_training=self.is_training, wd=WD):

                with tf.variable_scope('block_1'):
                    L.conv(filter_size=7, out_dim=128, nl=L.leaky_relu, padding='SAME', name='conv1')
                    resblock.res_block_bottleneck(128, 32, 128, name='resblock1')
                    L.max_pool(layer_dict=self.layers, filter_size=3, stride=2, padding='SAME', name='max_pool1')

                with tf.variable_scope('block_2'):
                    resblock.res_block_bottleneck(128, 32, 128, name='resblock1')
                    resblock.res_block_bottleneck(128, 32, 128, name='resblock2')
                    resblock.res_block_bottleneck(128, 64, 256, name='resblock3')
                    L.max_pool(layer_dict=self.layers, filter_size=3, stride=2, padding='SAME', name='max_pool1')

                with tf.variable_scope('block_3'):
                    resblock.res_block_bottleneck(256, 64, 256, name='resblock1')
                    resblock.res_block_bottleneck(256, 64, 256, name='resblock2')
                    L.max_pool(layer_dict=self.layers, filter_size=3, stride=2, padding='SAME', name='max_pool1')

                with tf.variable_scope('block_4'):
                    resblock.res_block_bottleneck(256, 64, 256, name='resblock1')
                    resblock.res_block_bottleneck(256, 64, 256, name='resblock2')
                    resblock.res_block_bottleneck(256, 128, 512, name='resblock3')
                    L.max_pool(layer_dict=self.layers, filter_size=3, stride=2, padding='SAME', name='max_pool1')

                with tf.variable_scope('block_5'):
                    resblock.res_block_bottleneck(512, 128, 512, name='resblock1')
                    resblock.res_block_bottleneck(512, 128, 512, name='resblock2')
                    L.max_pool(layer_dict=self.layers, filter_size=3, stride=2, padding='SAME', name='max_pool1')

                with tf.variable_scope('block_6'):
                    resblock.res_block(n1=512, n2=128, name='res_block1')
                    L.linear(out_dim=512, nl=L.leaky_relu, name='linear1')
                    L.linear(out_dim=self.embedding_dim, name='linear2')

            return self.layers['cur_input']

    def train_steps(self, sess, train_data, init_lr=1e-3, t0=15000, t1=25000, max_step=100, summary_writer=None):
        self.epoch_id += 1
        display_name_list = ['loss']
        # cur_epoch = train_data.epochs_completed

        cur_summary = None
        step = 0

        loss_sum = 0 
        while step < max_step and self.global_step <= t1:
            if self.global_step <= t0:
                lr = init_lr
            else:
                lr = init_lr * (0.001 ** ((self.global_step - t0) / (t1 - t0)))
            step += 1
            self.global_step += 1

            batch_data = train_data.next_batch_dict()
            _, loss, cur_summary = sess.run(
                [self.train_op, self.loss_op, self.train_summary_op],
                feed_dict={self.lr:lr, self.image: batch_data['im'],
                           self.label: batch_data['label']})
            loss_sum += loss
            summary_writer.add_summary(cur_summary, self.global_step)

            # if step % 100 == 0:
            #     viz.display(
            #         global_step=self.global_step,
            #         step=step,
            #         scaler_sum_list=[loss_sum],
            #         name_list=display_name_list,
            #         collection='train',
            #         summary_val=cur_summary,
            #         summary_writer=summary_writer)

        print('==== lr:{} ===='.format(lr))
        viz.display(
            global_step=self.global_step,
            step=step,
            scaler_sum_list=[loss_sum],
            name_list=display_name_list,
            collection='train',
            summary_val=cur_summary,
            summary_writer=summary_writer)

    def inference_epoch(self, sess, dataflow, save_path=None):
        dataflow.reset_epochs_completed()

        # embedding_all = np.empty((0, self.embedding_dim))
        # label_all = np.empty((0))
        # filename_all = np.empty((0))

        embedding_all = []
        # label_all = []
        # filename_all = []

        step = 0
        print(dataflow.epoch_step)
        bar = progressbar.ProgressBar(maxval=dataflow.epoch_step + 2, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        while dataflow.epochs_completed < 1:
            step += 1
            bar.update(step)
            batch_data = dataflow.next_batch_dict()
            embedding = sess.run(
                self.embedding,
                feed_dict={self.image: batch_data['im']})

            # embedding_all = np.concatenate((embedding_all, embedding), axis=0)
            # label_all = np.concatenate((label_all, batch_data['label']), axis=0)
            # filename_all = np.concatenate((batch_data['filename'], batch_data['filename']), axis=0)

            # embedding_all += list(embedding)
            # label_all += list(batch_data['label'])
            # filename_all += list(batch_data['filename'])
            embedding_all.append(embedding)
            # label_all.append(batch_data['label'])
            # filename_all.append(batch_data['filename'])


        bar.finish()
        embedding_all = np.vstack(embedding_all)
        # label_all = np.hstack(label_all)
        # filename_all = np.hstack(filename_all)

        return np.array(embedding_all)

        

        # print(loss_sum / step)
        # if save_path is not None and self.embedding_dim == 2:
        #     import os
        #     save_dir = os.path.join(save_path, 'embedding_{}'.format(self.epoch_id))
        #     viz.viz_embedding(embedding=embedding_all, labels=label_all, save_dir=save_dir)



