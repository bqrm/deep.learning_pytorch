# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2020/11/16 14:47

from __future__ import absolute_import

import numpy as np
import os
import random

from utils.common_method import get_filename_iteratively


def split_coco_dataset(dataset_path, split_ratio=list([0.7, 0.2, 0.1])):
    if len(split_ratio) != 3:
        raise ValueError("length of split_ratio should be 3")

    ratio_cumsum = np.cumsum(split_ratio)
    if abs(ratio_cumsum[-1] - 1) >= 1e-6:
        raise ValueError("summation of split_ratio should be 1")

    image_path = os.path.join(dataset_path, "images")
    label_path = os.path.join(dataset_path, "labels")

    list_image = []
    list_label = []
    get_filename_iteratively(image_path, list_image)
    get_filename_iteratively(label_path, list_label)

    if len(list_image) != len(list_label):
        raise ValueError("num of image and num of label does not match")

    num_file = len(list_image)
    rand_idx = np.random.permutation(range(num_file))
    list_train = rand_idx[:int(num_file * ratio_cumsum[0])]
    list_test = rand_idx[int(num_file * ratio_cumsum[0]):int(num_file * ratio_cumsum[1])]

    train_handler = open(os.path.join(dataset_path, "train.txt"), "wb")
    test_handler = open(os.path.join(dataset_path, "test.txt"), "wb")
    valid_handler = open(os.path.join(dataset_path, "valid.txt"), "wb")

    for file_idx, (image_filename, label_filename) in enumerate(zip(list_image, list_label)):
        print("%d" % file_idx)

        # if file_idx > 10:
        #     break

        basename, _ = os.path.splitext(os.path.basename(image_filename))
        if basename not in label_filename:
            continue

        str_info = (image_filename + "\n").encode()
        if file_idx in list_train:
            train_handler.write(str_info)
        elif file_idx in list_test:
            test_handler.write(str_info)
        else:
            valid_handler.write(str_info)

    train_handler.close()
    test_handler.close()
    valid_handler.close()


def main():
    dataset_path = r"D:/Projects/DataSets/dish_recog_20201116/"

    split_coco_dataset(dataset_path)

    print("done")


if "__main__" == __name__:
    main()




























