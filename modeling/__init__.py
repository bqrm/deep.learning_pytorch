# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2019/12/13

import torch


def conv_relu(in_channel, out_channel, kernel, stride=1, padding=0, bn=True):
    if bn:
        layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
            torch.nn.BatchNorm2d(out_channel, eps=1e-3),
            torch.nn.ReLU(True),
        )
    else:
        layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
            torch.nn.ReLU(True),
        )

    return layer


class BaseModel(object):
    """
    implements a yolo
    """
    def __init__(self, **kwargs):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = None
    
    def _build_model(self, **kwargs):
        raise NotImplementedError()

    def dump_parameters(self, epoch_idx, **kwargs):
        """
        dump parameters to local file
        """
        raise NotImplementedError()

    def evaluate(self, dataset_path, batch_size, **kwargs):
        """
        evaluate in 
        """
        raise NotImplementedError()

    def predict(self):
        """
        predict with trained model
        """
        raise NotImplementedError()

    def train(self, dataset_path, num_epoch, batch_size, **kwargs):
        raise NotImplementedError()