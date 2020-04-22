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

