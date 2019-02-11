#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: inference_tools.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np


def pair_distance(embedding_1, embedding_2):
    """ Compute pairwise distance between two embedding list

        Args:
            embedding_1, embedding_2: two embeddings with shape [len_1, embedding_dim], [len_2, embedding_dim]

        Returns:
            Matix of pairwise Euclidean distance between embedding_1 and embedding_2 with shape [len_1, len_2]
    """
    embedding_1 = np.array(embedding_1)
    embedding_2 = np.array(embedding_2)
    assert embedding_1.shape[1] == embedding_2.shape[1]
    

    dot_12 = np.matmul(embedding_1, np.transpose(embedding_2)) # [len_1, len_2]
    dot_11 = np.matmul(embedding_1, np.transpose(embedding_1)) # [len_1, len_1]
    dot_22 = np.matmul(embedding_2, np.transpose(embedding_2)) # [len_2, len_2]

    square_1 = np.expand_dims(dot_11.diagonal(), axis=1)
    square_2 = np.expand_dims(dot_22.diagonal(), axis=0)

    distance = np.sqrt(np.maximum((square_1 + square_2 - 2 * dot_12), 0.))

    # # Use too many memeries
    # # make use of broadcasting
    # embedding_1 = np.expand_dims(embedding_1, axis=1) # [len_1, 1. embedding_dim]
    # embedding_2 = np.expand_dims(embedding_2, axis=0) # [1, len_2, embedding_dim]
    # diff_embedding = embedding_1 - embedding_2 # [len_1, len_2, embedding_dim]
    # distance2 = np.linalg.norm(diff_embedding, axis=-1)

    return distance

def ranking_distance(distance_mat, gallary_list, top_k=5):
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