#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: main.py
# Author: Qian Ge <geqian1001@gmail.com>

import argparse
import tensorflow as tf

import sys
sys.path.append('../')
import loader
from src.nets.mnist import MetricNet
import config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    return parser.parse_args()

def train():
    FLAGS = get_args()

    embedding_dim = 2

    train_data, valid_data = loader.loadMNIST(config.mnist_dir, sample_per_class=12)

    train_net = MetricNet(im_size=28, n_channels=1, embedding_dim=embedding_dim, margin=0.5)
    train_net.create_train_model()

    infer_net = MetricNet(im_size=28, n_channels=1, embedding_dim=embedding_dim, margin=0.5)
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

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.train:
        train()

