# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2020/4/28 10:29

from __future__ import absolute_import

import os
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

    def forward(self, x):
        pass


# class BottleNeckA(nn.Module):
#     """
#     bottleneck topology a of ResNeXt
#     """
#     expansion = 4
#
#     def __init__(self, in_channel, out_channel, cardinality=32, base_width=4, stride=1):
#         super(BottleNeckA, self).__init__()
#
#         self.cardinality = cardinality
#         self.conv_1 = nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)
#
#     def forward(self, x):
#         pass
#
#
# class BottleNeckB(nn.Module):
#     """
#     bottleneck topology b of ResNeXt
#     """
#     expansion = 2
#
#     def __init__(self, in_channel, out_channel, cardinality=32, base_width=4, stride=1):
#         super(BottleNeckB, self).__init__()
#
#         self.cardinality = cardinality
#         self.conv_1 = nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)
#
#     def forward(self, x):
#         pass


class BottleNeckC(nn.Module):
    """
    bottleneck topology c of ResNeXt
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, cardinality=32, shortcut=None, base_width=64):
        super(BottleNeckC, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        mid_channel = int(out_channel * (base_width / 64.)) * cardinality
        self.conv_1 = nn.Conv2d(in_channel, mid_channel, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(mid_channel)
        self.conv_2 = nn.Conv2d(mid_channel, mid_channel, 3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn_2 = nn.BatchNorm2d(mid_channel)
        self.conv_3 = nn.Conv2d(mid_channel, self.expansion * out_channel)
        self.bn_3 = nn.BatchNorm2d(self.expansion * out_channel)
        self.shortcut = shortcut

    def forward(self, x):
        output = self.conv_1(x)
        output = self.bn_1(output)
        output = self.relu(output)

        output = self.conv_2(output)
        output = self.bn_2(output)
        output = self.relu(output)

        output = self.conv_3(output)
        output = self.bn_3(output)

        if self.shortcut is None:
            output += x
        else:
            output += self.shortcut(x)

        output = self.relu(output)

        return output


class ResNeXtBase(nn.Module):
    def __init__(self, block, num_blocks, num_class=1000):
        super(ResNeXtBase, self).__init__()

        self.in_channel = 64
        self.block = block
        self.num_blocks = num_blocks
        self.num_class = num_class

        self.relu = nn.ReLU(True)
        self.bn_1 = nn.BatchNorm2d(64)

    def _make_layer(self, block, out_channel, num_block, stride):
        shortcut = None
        if stride != 1 or self.in_channel != block.expansion * out_channel:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_channel, block.expansion * out_channel, 1, stride=stride, bias=False),
                nn.BatchNorm2d(block.expansion * out_channel),
            )

        layers = list()
        layers.append(block(self.in_channel, out_channel, stride, shortcut))
        self.in_channel = block.expansion * out_channel

        for _ in range(1, num_block):
            layers.append(block(self.in_channel, out_channel))

        return nn.Sequential(*layers)


class ResNeXtImageNet(ResNeXtBase):
    def __init__(self, block, num_blocks, num_class=1000):
        super(ResNeXtImageNet, self).__init__(block, num_blocks, num_class)

        self.conv_1 = nn.Conv2d


def test(model, criterion, data_loader, use_cuda=False):
    pass


def train(model, epoch, criterion, optimizer, data_loader, use_cuda=False):
    pass


def main():
    pass


if "__main__" == __name__:
    import argparse
    
    main()




























