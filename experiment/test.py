#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test.py
# Author: Qian Ge <geqian1001@gmail.com>

import loader


def test_dataflow():
    import matplotlib.pyplot as plt
    dataflow, valid_data = loader.loadMNIST()
    batch_data = dataflow.next_batch_dict()
    print(batch_data)
    print(batch_data['im'].shape)
    plt.figure()
    plt.imshow(batch_data['im'][0,:,:,0])
    plt.show()


if __name__ == '__main__':
    test_dataflow()
