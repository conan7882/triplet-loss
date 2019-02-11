#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: main.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import argparse
import tensorflow as tf
import numpy as np
import scipy.io

import sys
sys.path.append('../')
import loader
from src.nets.mnist import MetricNet
from src.nets.lunet import LuNet
import config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--folder', type=str, default='test')

    parser.add_argument('--embed', type=int, default=2)
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--load', type=int, default=249)
    parser.add_argument('--lr', type=float, default=1e-4)
    return parser.parse_args()

def train_mnist():
    FLAGS = get_args()

    embedding_dim = FLAGS.embed
    margin = FLAGS.margin

    train_data, valid_data = loader.loadMNIST(config.mnist_dir, sample_per_class=12)

    train_net = MetricNet(im_size=28, n_channels=1, embedding_dim=embedding_dim, margin=margin)
    train_net.create_train_model()

    infer_net = MetricNet(im_size=28, n_channels=1, embedding_dim=embedding_dim, margin=margin)
    infer_net.create_inference_model()

    writer = tf.summary.FileWriter(config.mnist_save_path)
    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        for epoch_id in range(50):
            train_net.train_epoch(sess, train_data, lr=FLAGS.lr, summary_writer=None)
            infer_net.inference_epoch(sess, valid_data, save_path=config.mnist_save_path)

def train_market():
    FLAGS = get_args()
    save_path = os.path.join(config.market_save_path, FLAGS.folder)

    embedding_dim = FLAGS.embed # 128
    im_size = [128, 64]
    margin = FLAGS.margin # 0.5

    train_data = loader.loadMarketTrain(config.market_dir, sample_per_class=4, rescale_im=im_size)

    train_net = LuNet(im_size=im_size, n_channels=3, embedding_dim=embedding_dim, margin=margin)
    train_net.create_train_model()

    # infer_net = LuNet(im_size=im_size, n_channels=3, embedding_dim=embedding_dim, margin=0.5)
    # infer_net.create_inference_model()

    writer = tf.summary.FileWriter(save_path)
    saver = tf.train.Saver()
    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        for epoch_id in range(250):
            train_net.train_steps(sess, train_data, init_lr=FLAGS.lr, t0=15000, t1=25000, max_step=100, summary_writer=writer)
            # infer_net.inference_epoch(sess, valid_data, save_path=config.mars_save_path)
            if epoch_id % 50 == 0:
                saver.save(sess, '{}/market_step_{}'.format(save_path, epoch_id*100))
        saver.save(sess, '{}/market_step_{}'.format(save_path, epoch_id*100))

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.dataset == 'mnist':
        train_mnist()
    elif FLAGS.dataset == 'market':
        train_market()

