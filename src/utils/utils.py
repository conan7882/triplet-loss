#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: utils.py
# Author: Qian Ge <geqian1001@gmail.com>

def make_list(inputs):
    if not isinstance(inputs, list):
        return [inputs]
    else:
        return inputs

def get_shape4D(in_val):
    """
    Return a 4D shape
    Args:
        in_val (int or list with length 2)
    Returns:
        list with length 4
    """
    # if isinstance(in_val, int):
    return [1] + get_shape2D(in_val) + [1]

def get_shape2D(in_val):
    """
    Return a 2D shape 
    Args:
        in_val (int or list with length 2) 
    Returns:
        list with length 2
    """
    # in_val = int(in_val)
    if isinstance(in_val, int):
        return [in_val, in_val]
    if isinstance(in_val, list):
        assert len(in_val) == 2
        return in_val
    raise RuntimeError('Illegal shape: {}'.format(in_val))