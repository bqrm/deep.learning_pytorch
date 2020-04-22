# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2019/12/27

import numpy as np


class Evaluator(object):
    """
    a 2d histogram
    """
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

        np.seterr(invalid='ignore')

    def _generate_matrix(self, gt_image, predicted):
        mask = (gt_image > 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + predicted[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        return count.reshape(self.num_class, self.num_class)

    def _intersection_over_union(self):
        return np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) +
                                                 self.confusion_matrix.sum(axis=1) -
                                                 np.diag(self.confusion_matrix))

    def add_batch(self, gt_image, predict_image):
        assert gt_image.shape == predict_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, predict_image)

    def frequency_weighted_intersection_over_union(self):
        freq = self.confusion_matrix.sum(axis=1) / np.sum(self.confusion_matrix)
        iou = self._intersection_over_union()

        return (freq[freq > 0] * iou[freq > 0]).sum()

    def mean_intersection_over_union(self):
        return np.nanmean(self._intersection_over_union())

    def pixel_accuracy(self):
        return np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

    def pixel_accuracy_class(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return np.nanmean(acc)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, ) * 2)


































