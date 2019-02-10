#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test.py
# Author: Qian Ge <geqian1001@gmail.com>

import loader
import config


def test_dataflow():
    import matplotlib.pyplot as plt
    dataflow, valid_data = loader.loadMARS(config.mars_dir)
    batch_data = dataflow.next_batch_dict()
    print(batch_data)
    print(batch_data['im'].shape)
    plt.figure()
    plt.imshow(batch_data['im'][0])
    plt.show()


if __name__ == '__main__':
    test_dataflow()
