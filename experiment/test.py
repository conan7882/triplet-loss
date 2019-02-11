#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test.py
# Author: Qian Ge <geqian1001@gmail.com>

import loader
import config
import os
import imageio
import sys
import numpy as np
import scipy.io

sys.path.append('../')
# import src.inference.distance as distance
import src.inference.retrieve as retrieve

import src.utils.viz as viz
# import src.utils.dataflow as dfutil

def test_dataflow():
    import matplotlib.pyplot as plt
    dataflow = loader.loadMarketTrain(config.market_dir)
    im_list = dataflow.im_list
    # head, tail = ntpath.split(im_list[0])
    # print(head)
    # print(tail)
    # print(dataflow.im_list)
    # np.save('imlist.npy', dataflow.im_list)
    for i in range(dataflow._n_class):
        print(dataflow.size(i))
    batch_data = dataflow.next_batch_dict()
    # print(batch_data)
    # print(batch_data['im'].shape)
    plt.figure()
    plt.imshow(batch_data['im'][0])
    plt.show()


def test_distance():
    embedding_1 = [[1,2,3],[1,3,4]]
    embedding_2 = [[3,4,5], [1,2,2], [1,3,4]]
    gallary_list = ['im1','im2','im3']

    dist = distance.pair_distance(embedding_1, embedding_2)
    ranking = retrieve.ranking(dist, gallary_list, top_k=2)
    print(ranking)

def test_map():
    from src.dataflow.market import distractor_IDs, parse_filename
    from src.eval.mAP import mean_ap
    import src.inference.inference_tools as infertool
    from src.inference.re_ranking import re_ranking

    # test_dict = np.load('../lib/market_test.npy', encoding='latin1').item()
    test_dict = np.load('E:/GITHUB/workspace/triplet/market_test.npy', encoding='latin1').item()
    embedding = test_dict['embedding']
    file_path = test_dict['filename']

    pids, camera_ids = parse_filename(file_path)

    distractor_idx = distractor_IDs(file_path)
    embedding = np.delete(embedding, distractor_idx, axis=0)
    file_path = np.delete(file_path, distractor_idx, axis=0)
    pids = np.delete(pids, distractor_idx, axis=0)
    camera_ids = np.delete(camera_ids, distractor_idx, axis=0)

    n_test = len(embedding)
    n_query = 10
    pert_idx = np.random.permutation(n_test)
    query_id = pert_idx[:n_query]
    gallery_id = pert_idx[n_query:]

    query_embedding = embedding[query_id]
    query_file = file_path[query_id]
    query_pids = pids[query_id]
    query_camera_ids = camera_ids[query_id]

    gallery_embedding = embedding[gallery_id]
    gallery_file = file_path[gallery_id]
    gallery_pids = pids[gallery_id]
    gallery_camera_ids = camera_ids[gallery_id]

    # q_g_dist = infertool.pair_distance(query_embedding, gallery_embedding)
    # q_q_dist = infertool.pair_distance(query_embedding, query_embedding)
    # g_g_dist = infertool.pair_distance(gallery_embedding, gallery_embedding)
    # pair_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3)

    pair_dist = infertool.pair_distance(query_embedding, gallery_embedding)

    aps = mean_ap(
        distmat=pair_dist,
        query_ids=query_pids,
        gallery_ids=gallery_pids,
        query_cams=query_camera_ids,
        gallery_cams=gallery_camera_ids,
        average=True)
    print(aps)


def test_ranking():
    from src.dataflow.market import distractor_IDs
    test_dict = np.load('../lib/market_test.npy', encoding='latin1').item()
    embedding = test_dict['embedding']
    file_path = test_dict['filename']

    distractor_idx = distractor_IDs(file_path)
    embedding = np.delete(embedding, distractor_idx, axis=0)
    file_path = np.delete(file_path, distractor_idx, axis=0)

    data_dir = os.path.join(config.market_dir, 'bounding_box_test')
    save_path = config.market_save_path

    query_file_name, ranking_file_mat = retrieve.viz_ranking_single_testset(
        embedding, file_path, n_query=10, top_k=10,
        data_dir=data_dir, save_path=save_path, is_viz=True)


if __name__ == '__main__':
    test_ranking()
