#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: retrieve.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np


def ranking(distance_mat, gallary_list, top_k=5):
    # distance_mat [query_len, gallary_len]
    distance_mat = np.array(distance_mat)
    gallary_list = np.array(gallary_list)
    assert len(gallary_list) == distance_mat.shape[1]
    # get top k closest distance for each query input without sorting
    top_k_ind_mat = np.argpartition(distance_mat, top_k, axis=1)[:, :top_k]
    query_id_list = [[i for _ in range(top_k)] for i in range(distance_mat.shape[0])]
    query_idx = np.reshape(np.array(query_id_list), (-1))
    top_k_idx = np.reshape(top_k_ind_mat, (-1))
    top_k_distance = np.reshape(distance_mat[(query_idx, top_k_idx)], (distance_mat.shape[0], top_k))
    # sort distance for each query input and get the corresponding gallary id
    sort_top_k = np.reshape(np.argsort(top_k_distance, axis=1), (-1))
    sort_top_k = top_k_ind_mat[(query_idx, sort_top_k)]
    top_k_gallary = np.reshape(gallary_list[sort_top_k], (distance_mat.shape[0], top_k))
    
    return top_k_gallary