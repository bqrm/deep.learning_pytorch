# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2020/2/21 11:43

from __future__ import absolute_import


import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    basic block
    """
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, shortcut=None, base_width=64):
        super(BasicBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        mid_channel = int(out_channel * (base_width / 64.))
        self.conv_1 = nn.Conv2d(in_channel, mid_channel, 3, stride=stride, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(mid_channel)
        self.conv_2 = nn.Conv2d(mid_channel, mid_channel, 3, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(out_channel)
        self.shortcut = shortcut

    def forward(self, x):
        output = self.conv_1(x)
        output = self.bn_1(output)
        output = self.relu(output)

        output = self.conv_2(output)
        output = self.bn_2(output)

        if self.shortcut is None:
            output += x
        else:
            output += self.shortcut(x)

        output = self.relu(output)

        return output


class BottleNeck(nn.Module):
    """
    bottleneck
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, shortcut=None, base_width=64):
        super(BottleNeck, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        mid_channel = int(out_channel * (base_width / 64.))
        self.conv_1 = nn.Conv2d(in_channel, mid_channel, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(mid_channel)
        self.conv_2 = nn.Conv2d(mid_channel, mid_channel, 3, stride=stride, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(mid_channel)
        self.conv_3 = nn.Conv2d(mid_channel, self.expansion * out_channel, 1, bias=False)
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


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_class=1000):
        super(ResNet, self).__init__()

        self.in_channel = 64

        self.conv_1 = nn.Conv2d(3, self.in_channel, 3, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.layer_1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer_2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer_3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer_4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_class)

    def _make_layer(self, block, out_channel, num_block, stride):
        shortcut = None
        if stride != 1 or self.in_channel != block.expansion * out_channel:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_channel, block.expansion * out_channel, 1, stride=stride, bias=False),
                nn.BatchNorm2d(block.expansion * out_channel),
            )

        layers = list()
        layers.append(block(self.in_channel, out_channel, stride, shortcut))
        self.in_channel = out_channel * block.expansion

        for _ in range(1, num_block):
            layers.append(block(self.in_channel, out_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


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


def test(model, criterion, data_loader, use_cuda=False):
    pass


def train(model, epoch, criterion, optimizer, data_loader, use_cuda=False):
    pass


def main():
    import argparse
    model = res_net_50()


if "__main__" == __name__:
    main()























