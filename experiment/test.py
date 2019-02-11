#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test.py
# Author: Qian Ge <geqian1001@gmail.com>

import loader
import config
import sys
import numpy as np
import scipy.io
import ntpath
sys.path.append('../')
import src.inference.distance as distance
import src.inference.retrieve as retrieve


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
    test_dict = np.load('/Users/gq/workspace/GitHub/MARS-evaluation/data/train_embed.npy', encoding='latin1').item()
    embedding = test_dict['embedding']
    scipy.io.savemat('/Users/gq/workspace/GitHub/MARS-evaluation/data/train_embed.mat',
        {'test_feat': embedding})
    # label_q = query_dict['label_q']
    # filename_q = query_dict['filename_q']

    # gallery_dict = np.load('../lib/gallery.npy', encoding='latin1').item()
    # embedding_g = gallery_dict['embedding_g']
    # label_g = gallery_dict['label_g']
    # filename_g = gallery_dict['filename_g']

    print(embedding.shape)
    # print(label_g[0])
    # print(filename_g[0])

    # dist = distance.pair_distance(embedding_q, embedding_g)
    # ranking = retrieve.ranking(dist, filename_g, top_k=5)

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
    test_dataflow()
