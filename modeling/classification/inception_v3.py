# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2020/2/19 14:16

from __future__ import absolute_import

import torch
import torch.nn as nn

from modeling import conv_relu


# def conv_relu_factorization(in_channel, out_channel, kernel, padding):
#     layer = nn.Sequential(
#         nn.Conv2d(in_channel, out_channel, kernel, 1, padding),
#         nn.BatchNorm2d(out_channel, eps=1e-3),
#         nn.ReLU(True),
#         nn.Conv2d(out_channel, out_channel, kernel, 1, padding),
#         nn.BatchNorm2d(out_channel),
#         nn.ReLU(True),
#     )
#
#     return layer


class ModuleA(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_3, out4_1):
        super(ModuleA, self).__init__()

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
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 3, 1),
        )

    def forward(self, x):
        f1 = self.branch_1(x)
        f2 = self.branch_2(x)
        f3 = self.branch_3(x)
        f4 = self.branch_4(x)

        output = torch.cat([f1, f2, f3, f4], dim=1)
        return output


class ModuleB(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3):
        super(ModuleB, self).__init__()

        self.branch_1 = conv_relu(in_channel, out1_1, 3, stride=2, padding=1)

        self.branch_2 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_1, 3, padding=1),
            conv_relu(out2_1, out2_3, 3, stride=2, padding=1),
        )

        self.branch_3 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
        )

    def forward(self, x):
        f1 = self.branch_1(x)
        f2 = self.branch_2(x)
        f3 = self.branch_3(x)

        output = torch.cat([f1, f2, f3], dim=1)
        return output


class ModuleC(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_3, out4_1):
        super(ModuleC, self).__init__()

        self.branch_1 = conv_relu(in_channel, out1_1, 1)

        self.branch_2 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_1, kernel=[1, 7], padding=[0, 3]),
            conv_relu(out2_1, out2_3, kernel=[7, 1], padding=[3, 0]),
        )

        self.branch_3 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_1, kernel=[1, 7], padding=[0, 3]),
            conv_relu(out3_1, out3_1, kernel=[7, 1], padding=[3, 0]),
            conv_relu(out3_1, out3_1, kernel=[1, 7], padding=[0, 3]),
            conv_relu(out3_1, out3_3, kernel=[7, 1], padding=[3, 0]),
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(3, padding=1),
            conv_relu(in_channel, out4_1, 1),
        )

    def forward(self, x):
        f1 = self.branch_1(x)
        f2 = self.branch_2(x)
        f3 = self.branch_3(x)
        f4 = self.branch_4(x)

        output = torch.cat([f1, f2, f3, f4], dim=1)
        return output


class ModuleD(nn.Module):
    def __init__(self, in_channel, out1_1, out1_3, out2_1, out2_3):
        super(ModuleD, self).__init__()

        self.branch_1 = nn.Sequential(
            conv_relu(in_channel, out1_1, 1),
            conv_relu(out1_1, out1_3, 3, stride=2, padding=1)
        )

        self.branch_2 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_1, kernel=[1, 7], padding=[0, 3]),
            conv_relu(out2_1, out2_1, kernel=[7, 1], padding=[3, 0]),
            conv_relu(out2_1, out2_3, 3, stride=2, padding=1)
        )

        self.branch_3 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
        )

    def forward(self, x):
        f1 = self.branch_1(x)
        f2 = self.branch_2(x)
        f3 = self.branch_3(x)

        output = torch.cat([f1, f2, f3], dim=1)
        return output


class ModuleE(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_3, out4_1):
        super(ModuleE, self).__init__()

        self.branch_1 = conv_relu(in_channel, out1_1, 1)

        self.branch_2_1 = conv_relu(in_channel, out2_1, 1)
        self.branch_2_2_1 = conv_relu(out2_1, out2_3, kernel=[1, 3], padding=[0, 1])
        self.branch_2_2_2 = conv_relu(out2_1, out2_3, kernel=[3, 1], padding=[1, 0])

        self.branch_3_1 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_3, 3, padding=1),
        )
        self.branch_3_2_1 = conv_relu(out3_3, out3_3, kernel=[1, 3], padding=[0, 1])
        self.branch_3_2_2 = conv_relu(out3_3, out3_3, kernel=[3, 1], padding=[1, 0])

        self.branch_4 = nn.Sequential(
            nn.AvgPool2d(3, padding=1),
            conv_relu(in_channel, out4_1, 1),
        )

    def forward(self, x):
        f1 = self.branch_1(x)

        f2_1 = self.branch_2_1(x)
        f2 = torch.cat([self.branch_2_2_1(f2_1), self.branch_2_2_2(f2_1)], dim=1)

        f3_1 = self.branch_3_1(x)
        f3 = torch.cat([self.branch_3_2_1(f3_1), self.branch_3_2_2(f3_1)], dim=1)

        f4 = self.branch_4(x)

        output = torch.cat([f1, f2, f3, f4], dim=1)
        return output


class ModuleAux(nn.Module):
    def __init__(self, in_channel, num_class):
        super(ModuleAux, self).__init__()

        self.branch_1 = nn.Sequential(
            nn.AvgPool2d(5, 3, padding=2),
            conv_relu(in_channel, 128, 1),
            conv_relu(128, 768, 5, padding=2),
        )

        self.drop_out = nn.Dropout(p=0.4)

        self.classifier = nn.Linear(768, num_class)

    def forward(self, x):
        x = self.branch_1(x)
        x = self.drop_out(x)

        x = x.view(x.shape[0], -1)
        output = self.classifier(x)

        return output


class Inception(torch.nn.Module):
    def __init__(self, in_channel, num_class):
        super(Inception, self).__init__()

































