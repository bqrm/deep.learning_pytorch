# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2020/2/19 14:16

from __future__ import absolute_import

import torch
import torch.nn as nn

from modeling import conv_relu


class BottleNeckPool(torch.nn.Module):
    def __init__(self, in_channel, out1_1, out1_3, out2_1, out2_3):
        super(BottleNeckPool, self).__init__()

        self.branch_1 = nn.Sequential(
            conv_relu(in_channel, out1_1, 1),
            conv_relu(out1_1, out1_3, 3, stride=2, padding=1),
        )
        self.branch_2 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1),
            conv_relu(out2_3, out2_3, 3, stride=2, padding=1),
        )
        self.branch_3 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
        )

    def forward(self, x):
        f1 = self.branch_1(x)
        f2 = self.branch_2(x)
        f3 = self.branch_3(x)

        output = torch.cat((f1, f2, f3), dim=1)
        return output


class BottleNeckNorm(torch.nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_3, out4_1):
        super(BottleNeckNorm, self).__init__()

        self.branch_1 = conv_relu(in_channel, out1_1, 1)

        self.branch_2 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1),
        )

        self.branch_3 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_3, 3, padding=1),
            conv_relu(out3_3, out3_3, 3, padding=1),
        )

        self.branch_4 = nn.Sequential(
            nn.AvgPool2d(3, padding=1),
            conv_relu(in_channel, out4_1, 1),
        )

    def forward(self, x):
        f1 = self.branch_1(x)
        f2 = self.branch_2(x)
        f3 = self.branch_3(x)
        f4 = self.branch_4(x)

        output = torch.cat((f1, f2, f3, f4), dim=1)
        return output


class Inception(nn.Module):
    def __init__(self, in_channel, num_class, verbose=False):
        super(Inception, self).__init__()

        self.verbose = verbose

        self.block1 = nn.Sequential(
            conv_relu(in_channel, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, 2, padding=1),
        )

        self.block2 = nn.Sequential(
            conv_relu(64, 64, 1),
            conv_relu(64, 192, 3, padding=1),
            nn.MaxPool2d(3, 2, padding=1),
        )

        self.block3 = nn.Sequential(
            BottleNeckNorm(192, 64, 64, 64, 64, 96, 32),
            BottleNeckNorm(256, 64, 64, 96, 64, 96, 64),
            BottleNeckPool(320, 128, 160, 64, 96),
        )

        self.block4 = nn.Sequential(
            BottleNeckNorm(576, 224, 64, 96, 96, 128, 128),
            BottleNeckNorm(576, 192, 96, 128, 96, 128, 128),
            BottleNeckNorm(576, 160, 128, 160, 128, 160, 96),
            BottleNeckNorm(576, 96, 128, 192, 160, 192, 96),
            BottleNeckPool(576, 128, 192, 192, 256),
        )

        self.block5 = nn.Sequential(
            BottleNeckNorm(1024, 352, 192, 320, 160, 224, 128),
            BottleNeckNorm(1024, 352, 192, 320, 192, 224, 128),
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(7, padding=3),
            conv_relu(1024, 1000, 1),
        )

    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print("block 1 output: {}".format(x.shape))

        x = self.block2(x)
        if self.verbose:
            print("block 2 output: {}".format(x.shape))

        x = self.block3(x)
        if self.verbose:
            print("block 3 output: {}".format(x.shape))

        x = self.block4(x)
        if self.verbose:
            print("block 4 output: {}".format(x.shape))

        x = self.block5(x)
        if self.verbose:
            print("block 5 output: {}".format(x.shape))

        output = self.classifier(x)
        return output






























