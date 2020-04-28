# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2020/2/21 11:43

from __future__ import absolute_import


import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_plane, out_plane, stride=1):
        super(BasicBlock, self).__init__()

        self.conv_1 = nn.Conv2d(in_plane, out_plane, 3, stride=stride, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_plane)
        self.conv_2 = nn.Conv2d(out_plane, out_plane, 3, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(out_plane)

        if stride != 1 or in_plane != self.expansion*out_plane:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_plane, self.expansion*out_plane, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_plane),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        output = nn.functional.relu(self.bn_1(self.conv_1(x)))
        output = self.bn_2(self.conv_2(output))
        if self.shortcut is None:
            output += x
        else:
            output += self.shortcut(x)
        output = nn.functional.relu(output)
        return output


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_plane, out_plane, stride=1):
        super(BottleNeck, self).__init__()

        self.conv_1 = nn.Conv2d(in_plane, out_plane, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_plane)
        self.conv_2 = nn.Conv2d(out_plane, out_plane, 3, stride=stride, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(out_plane)
        self.conv_3 = nn.Conv2d(out_plane, self.expansion*out_plane, 1, bias=False)
        self.bn_3 = nn.BatchNorm2d(out_plane)

        if stride != 1 or in_plane != self.expansion*out_plane:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_plane, self.expansion*out_plane, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_plane),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        output = nn.functional.relu(self.bn_1(self.conv_1(x)))
        output = nn.functional.relu(self.bn_2(self.conv_2(output)))
        output = self.bn_3(self.conv_3(output))
        if self.shortcut is None:
            output += x
        else:
            output += self.shortcut(x)
        output = nn.functional.relu(output)
        return output


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_class=10):
        super(ResNet, self).__init__()

        self.in_plane = 64

        self.conv_1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.layer_1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer_2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer_3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer_4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_class)

    def _make_layer(self, block, out_plane, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []

        for tmp_stride in strides:
            layers.append(block(self.in_plane, out_plane, tmp_stride))
            self.in_plane = out_plane * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = nn.functional.relu(self.bn_1(self.conv_1(x)))
        output = self.layer_1(output)
        output = self.layer_2(output)
        output = self.layer_3(output)
        output = self.layer_4(output)
        output = nn.functional.avg_pool2d(output, 4, padding=2)
        output = output.view(output.shape[0], -1)
        output = self.linear(output)
        return output


def res_net_18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def res_net_24():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def res_net_50():
    return ResNet(BottleNeck, [3, 4, 6, 3])


def res_net_101():
    return ResNet(BottleNeck, [3, 6, 23, 3])


def res_net_152():
    return ResNet(BottleNeck, [3, 8, 36, 3])
























