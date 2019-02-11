#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: viz_ranking.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import argparse
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')
import loader
import config
from src.nets.lunet import LuNet
import src.inference.retrieve as retrieve
from src.dataflow.market import distractor_IDs, parse_filename
from src.eval.mAP import mean_ap

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--folder', type=str, default='test')
    parser.add_argument('--load_embed', action='store_true')

    parser.add_argument('--n_query', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--embed', type=int, default=128)
    parser.add_argument('--load_id', type=int, default=249)
    return parser.parse_args()

def inference_market():
    
    FLAGS = get_args()
    save_path = os.path.join(config.market_save_path, FLAGS.folder)

    embedding_dim = FLAGS.embed
    im_size = [128, 64]

    train_data, test_data = loader.loadMarketInference(config.market_dir, rescale_im=im_size, batch_size=256)

    infer_net = LuNet(im_size=im_size, n_channels=3, embedding_dim=embedding_dim)
    infer_net.create_inference_model()

    saver = tf.train.Saver()
    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '{}/{}_step_{}'.format(save_path, FLAGS.dataset, FLAGS.load_id*100))
        embedding = infer_net.inference_epoch(sess, test_data)
        # save_dict = {'embedding': embedding, 'filename': train_data.im_list}
        # np.save(os.path.join(save_path, '{}_train.npy'.format(FLAGS.dataset)), save_dict)

    return embedding, test_data.im_list

def ranking_markert():
    FLAGS = get_args()
    if FLAGS.load_embed:
        # test_dict = np.load('E:/GITHUB/workspace/triplet/market_test.npy', encoding='latin1').item()
        test_dict = np.load('../lib/market_test.npy', encoding='latin1').item()
        embedding = test_dict['embedding']
        file_path = test_dict['filename']
    else:
        embedding, file_path = inference_market()

    # pids, camera_ids = parse_filename(file_path)
    distractor_idx = distractor_IDs(file_path)
    embedding = np.delete(embedding, distractor_idx, axis=0)
    file_path = np.delete(file_path, distractor_idx, axis=0)

    data_dir = os.path.join(config.market_dir, 'bounding_box_test')
    save_path = os.path.join(config.market_save_path, FLAGS.folder)

    query_file_name, ranking_file_mat = retrieve.viz_ranking_single_testset(
        embedding, file_path, n_query=FLAGS.n_query, top_k=FLAGS.top_k,
        data_dir=data_dir, save_path=save_path, is_viz=True)

if __name__ == '__main__':
    ranking_markert()
