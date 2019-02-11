#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: distance.py
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
    # make use of broadcasting
    embedding_1 = np.expand_dims(embedding_1, axis=1) # [len_1, 1. embedding_dim]
    embedding_2 = np.expand_dims(embedding_2, axis=0) # [1, len_2, embedding_dim]

    diff_embedding = embedding_1 - embedding_2 # [len_1, len_2, embedding_dim]
    distance = np.linalg.norm(diff_embedding, axis=-1)
    return distance