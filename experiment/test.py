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

def test_ranking():
    from src.dataflow.market import distractor_IDs
    test_dict = np.load('E:/GITHUB/workspace/triplet/market_test.npy', encoding='latin1').item()
    embedding = test_dict['embedding']
    file_path = test_dict['filename']

    distractor_idx = distractor_IDs(file_path)
    embedding = np.delete(embedding, distractor_idx, axis=0)
    file_path = np.delete(file_path, distractor_idx, axis=0)

    data_dir = os.path.join(config.market_dir, 'bounding_box_test')
    save_path = config.market_save_path

    query_file_name, ranking_file_mat = retrieve.viz_ranking_single_testset(
        embedding, file_path, n_query=20, top_k=5,
        data_dir=data_dir, save_path=save_path, is_viz=False)

    # n_test = len(embedding)
    # n_query = 20
    # top_k = 5
    
    # pert_idx = np.random.permutation(n_test)
    # query_id = pert_idx[:n_query]
    # gallery_id = pert_idx[n_query:]

    # query_embedding = embedding[query_id]
    # query_file = file_name[query_id]

    # gallery_embedding = embedding[gallery_id]
    # gallery_file = file_name[gallery_id]

    # data_dir = os.path.join(config.market_dir, 'bounding_box_test')
    # save_path = config.market_save_path

    # query_file_name, ranking_file_mat = retrieve.ranking(
    #     query_embedding, gallery_embedding, query_file, gallery_file, top_k=5,
    #     data_dir=data_dir, save_path=save_path, is_viz=True)
    print(query_file_name, ranking_file_mat)


    # dist = distance.pair_distance(query_embedding, gallery_embedding)
    # ranking = retrieve.ranking(dist, gallery_file, top_k=top_k)

    # save_path = config.market_save_path

    # for idx, q_im in enumerate(query_file):
    #     im_list = []
    #     head, q_file_name = ntpath.split(q_im)
    #     im = imageio.imread(
    #         os.path.join(config.market_dir, 'bounding_box_test', q_file_name),
    #         as_gray=False, pilmode="RGB")
    #     im_list.append(im)

    #     for g_im in ranking[idx]:
    #         head, g_file_name = ntpath.split(g_im)
    #         im = imageio.imread(
    #             os.path.join(config.market_dir, 'bounding_box_test', g_file_name),
    #             as_gray=False, pilmode="RGB")
    #         im_list.append(im)

    #     viz.viz_batch_im(
    #         im_list, grid_size=[1, 1 + top_k], save_path=os.path.join(config.market_save_path, 'query_{}.png'.format(idx)),
    #         gap=2, gap_color=0, shuffle=False)

    # np.save('imlist.npy', ranking)

def test():
    im_list = np.load('../lib/imlist.npy')
    test_dict = np.load('/Users/gq/workspace/GitHub/MARS-evaluation/data/test_embed.npy', encoding='latin1').item()
    embedding = test_dict['embedding']

    id_list = []
    id_dict = {}
    cnt = 0
    tracklet_id = -1
    embed_sum = 0
    avg_embed = []

    t = 0
    for idx, im_dir in enumerate(im_list):
        head, tail = ntpath.split(im_dir)
        id_name = ntpath.basename(head)
        if id_name in id_dict:
            embed_sum += embedding[idx]
            cnt += 1
        else:
            if cnt > 0:
                avg_embed.append(1.0 * embed_sum / cnt)
            cnt = 0
            embed_sum = 0
            tracklet_id += 1
            id_dict[id_name] = tracklet_id

        t += 1
        if t > 2000:
            break

    print(tracklet_id)



if __name__ == '__main__':
    test_ranking()
