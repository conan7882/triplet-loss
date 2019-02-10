#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: hard_mining.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

SMALL_VAL = 1e-6

def pair_wise_distance(embedding):
    """ Compute pair wise distance of embedding
        
        Args:
            embedding: input embedding with shape [bsize, embedding_dim]

    """

    with tf.name_scope('pair_wise_distance'):
        dot_embed = tf.linalg.matmul(embedding, embedding, transpose_b=True) # [bsize, bsize]
        # a^2
        squared_embed = tf.linalg.tensor_diag_part(dot_embed) # [bsize, ]
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2* <a, b>
        # make use of broadcasting
        squared_a = tf.expand_dims(squared_embed, axis=0) # [1, bsize]
        squared_b = tf.expand_dims(squared_embed, axis=1) # [bsize, 1]
        squared_distance = squared_a + squared_b - 2. * dot_embed # [bsize, bsize]

        squared_distance = tf.math.maximum(squared_distance, 0.)
        squared_distance = tf.cast(squared_distance, tf.float32)

        distance = squared_distance

        mask = tf.to_float(tf.math.equal(distance, 0.0))
        distance = distance + mask * SMALL_VAL
        distance = tf.sqrt(distance)
        distance = distance * (1.0 - mask)

        return distance

def _get_mask(label, mask_type):
    # label [bsize, ]

    with tf.name_scope('get_mask'):
        if mask_type == 'positive':
            # make use of broadcasting
            mask = tf.math.equal(tf.expand_dims(label, axis=0), tf.expand_dims(label, axis=1)) # [bsize, bsize]
            # mask out diagonal distance (distance of the same sample)
            equal_mask = tf.logical_not(tf.cast(tf.eye(tf.shape(label)[0]), tf.bool))
            mask = tf.logical_and(mask, equal_mask)
        elif mask_type == 'negative':
            mask = tf.math.equal(tf.expand_dims(label, axis=0), tf.expand_dims(label, axis=1)) # [bsize, bsize]
            mask = tf.logical_not(mask)
        else:
            raise ValueError('Wrong mask_type: {}'.format(mask_type))
        return tf.cast(mask, tf.float32)

def batch_hard_triplet_loss(embedding, label, margin):
    with tf.name_scope('hard_mining'):
        pair_wise_d = pair_wise_distance(embedding) # [bsize, bsize]

        # hard positive
        positive_mask = _get_mask(label, mask_type='positive')
        positive_distance = tf.multiply(pair_wise_d, positive_mask)
        hard_positive = tf.math.reduce_max(positive_distance, axis=1, keepdims=True) # [bsize, 1]

        # hard negative
        negative_mask = 1. - _get_mask(label, mask_type='negative')
        negative_mask = tf.reduce_max(pair_wise_d) * negative_mask
        
        negative_distance = pair_wise_d + negative_mask
        hard_negative = tf.math.reduce_min(negative_distance, axis=1, keepdims=True) # [bsize, 1]

        triplet_loss = tf.maximum(0., hard_positive - hard_negative + margin) # [bsize, 1]
        nonzero_loss = tf.math.count_nonzero(tf.math.not_equal(triplet_loss, 0.), dtype=tf.float32)
        nonzero_ratio = nonzero_loss / tf.cast(tf.shape(embedding)[0], tf.float32)

        return tf.reduce_mean(triplet_loss), nonzero_ratio










