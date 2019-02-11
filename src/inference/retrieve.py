#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: retrieve.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import imageio
import ntpath
import numpy as np
import src.utils.viz as viz
import src.inference.inference_tools as infertool


def viz_ranking_single_testset(embedding, file_path, n_query=20, top_k=5,
                               data_dir=None, save_path=None, is_viz=False):
    n_test = len(embedding)
    
    pert_idx = np.random.permutation(n_test)
    query_id = pert_idx[:n_query]
    gallery_id = pert_idx[n_query:]

    query_embedding = embedding[query_id]
    query_file = file_path[query_id]

    gallery_embedding = embedding[gallery_id]
    gallery_file = file_path[gallery_id]

    query_file_name, ranking_file_mat = viz_ranking(
        query_embedding, gallery_embedding, query_file, gallery_file, top_k=top_k,
        data_dir=data_dir, save_path=save_path, is_viz=is_viz)

    return query_file_name, ranking_file_mat

def viz_ranking(query_embedding, gallery_embedding, query_file_name, gallery_file_name, top_k=5,
                data_dir=None, save_path=None, is_viz=False):
    pair_dist = infertool.pair_distance(query_embedding, gallery_embedding)
    ranking_file_mat = infertool.ranking_distance(pair_dist, gallery_file_name, top_k=top_k)

    frame_width = 2
    frame_color_correct = [0, 255, 0]
    frame_color_wrong = [255, 0, 0]

    if is_viz:
        assert data_dir and save_path
        for idx, q_im in enumerate(query_file_name):
            im_list = []
            head, q_file_name = ntpath.split(q_im)
            q_class_id = q_file_name.split('_')[0]
            im = imageio.imread(
                os.path.join(data_dir, q_file_name),
                as_gray=False,
                pilmode="RGB")
            im = viz.add_frame_im(im, frame_width, frame_color=0)
            im_list.append(im)
            
            for g_im in ranking_file_mat[idx]:
                head, g_file_name = ntpath.split(g_im)
                g_class_id = g_file_name.split('_')[0]
                im = imageio.imread(
                    os.path.join(data_dir, g_file_name),
                    as_gray=False,
                    pilmode="RGB")
                if g_class_id == q_class_id:
                    frame_color = frame_color_correct
                else:
                    frame_color = frame_color_wrong

                im = viz.add_frame_im(im, frame_width, frame_color=frame_color)
                im_list.append(im)

            viz.viz_batch_im(
                im_list,
                grid_size=[1, 1 + top_k],
                save_path=os.path.join(save_path, 'query_{}.png'.format(idx)),
                gap=0, gap_color=0, shuffle=False)

    return query_file_name, ranking_file_mat
