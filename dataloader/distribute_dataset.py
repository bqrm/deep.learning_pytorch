# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2020/2/26 22:27

from __future__ import absolute_import

import numpy as np
import os
import random

from utils.common_method import delete_recursively
from utils.common_method import get_filename_iteratively


def distribute_dataset(image_path, segmentation_path, distribute_ratio=[0.6, 0.2, 0.2]):
    if not os.path.exists(image_path):
        raise ValueError("image_path \"%s\" not exists" % image_path)
    if os.path.exists(segmentation_path):
        delete_recursively(segmentation_path)
    os.makedirs(segmentation_path)
    if not os.path.exists(segmentation_path):
        raise ValueError("failed to create segmentation_path \"%s\"" % segmentation_path)
    if not (1 < len(distribute_ratio) < 4):
        raise ValueError("distribute_ratio should be a list of 2 or 3 items")
    ratio_cum_sum = np.cumsum(distribute_ratio)
    if ratio_cum_sum[-1] != 1:
        raise ValueError("summation of distribute_ratio should be 1")

    list_filename = get_filename_iteratively(image_path)
    num_sample = len(list_filename)

    file_index = [idx for idx in range(num_sample)]
    random.shuffle(file_index)

    num_cum_sum = (num_sample * ratio_cum_sum).astype(int)
    train_index = file_index[:num_cum_sum[0]]
    test_index = file_index[num_cum_sum[0]:num_cum_sum[1]]
    if len(distribute_ratio) > 2:
        val_index = file_index[num_cum_sum[1]:]
    else:
        val_index = []

    train_split_name = os.path.join(segmentation_path, "train.txt")
    test_split_name = os.path.join(segmentation_path, "test.txt")
    val_split_name = os.path.join(segmentation_path, "val.txt")

    for image_index, image_name in enumerate(list_filename):
        if 0 == image_index % 1000:
            print("\t%5d" % image_index)
        base_name = os.path.basename(image_name).split('.')[0]

        if image_index in train_index:
            with open(train_split_name, "a") as f:
                f.write(str(base_name) + "\n")
        if image_index in test_index:
            with open(test_split_name, "a") as f:
                f.write(str(base_name) + "\n")
        if image_index in val_index:
            with open(val_split_name, "a") as f:
                f.write(str(base_name) + "\n")

    print("done")


def main():
    image_path = "D:/Projects/DataSets/mini_image_net/images"
    segmentation_path = "D:/Projects/DataSets/mini_image_net/segmentation"

    distribute_dataset(image_path, segmentation_path)


if "__main__" == __name__:
    main()
































