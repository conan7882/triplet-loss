#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: testset.py
# Author: Qian Ge <geqian1001@gmail.com>


import numpy as np
from src.dataflow.base import DataflowBaseChild, DataflowBaseTriplet
import src.utils.dataflow as dfutil


def prepare_query_gallery(TripletDataFlow, ChildDataFlow, data_dir, batch_dict_name,
                          batch_size=256, shuffle=True, pf=dfutil.identity, query_ratio=0.3):
    assert 0 <= query_ratio <= 1, 'query_ratio must be within [0, 1]!'
    # assert isinstance(TripletDataFlow, DataflowBaseTriplet)
    triplet_dataflow = TripletDataFlow(
        data_dir=data_dir,
        batch_dict_name=batch_dict_name[:2],
        shuffle=True,
        pf=pf)

    query_im_list = []
    query_label_list = []
    gallery_im_list = []
    gallery_label_list = []
    for class_id, im_list in enumerate(triplet_dataflow.im_list):
        n_sample = triplet_dataflow.size(class_id)
        query_sample = int(n_sample * query_ratio)

        query_im_list += list(im_list[:query_sample])
        query_label_list += [class_id for _ in range(query_sample)]
        gallery_im_list += list(im_list[query_sample:])
        gallery_label_list += [class_id for _ in range(n_sample - query_sample)]

    # print(query_im_list)
    query_data = ChildDataFlow(
        im_list=query_im_list,
        label_list=query_label_list,
        batch_dict_name=batch_dict_name,
        shuffle=shuffle,
        pf=pf)
    query_data.setup(epoch_val=0, batch_size=batch_size)

    gallery_data = ChildDataFlow(
        im_list=gallery_im_list,
        label_list=gallery_label_list,
        batch_dict_name=batch_dict_name,
        shuffle=shuffle,
        pf=pf)
    gallery_data.setup(epoch_val=0, batch_size=batch_size)

    return query_data, gallery_data


