# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2019/12/26

import numpy as np


pascal_labels = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
                          [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0],
                          [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                          [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]])

cityscapes_labels = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
                              [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
                              [0, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                              [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])


def decode_seg_map(label_img, dataset):
    if dataset in ["pascal", "coco"]:
        num_class = 21
        label_colors = pascal_labels
    elif dataset == "cityscapes":
        num_class = 19
        label_colors = cityscapes_labels
    else:
        raise NotImplementedError

    r = label_img.copy()
    g = label_img.copy()
    b = label_img.copy()
































