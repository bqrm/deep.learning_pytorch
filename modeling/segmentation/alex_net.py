# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2019/12/30

import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_class, is_training=False):
        super(AlexNet, self).__init__()
        self.num_class = num_class
        self.is_training = is_training

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, padding=3, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.num_class)
        self.relu = nn.ReLU(True)
        self.drop_out = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool5(self.relu(self.conv5(x)))
        x = x.view(-1, 9216)
        x = self.relu(self.fc1(x))
        if self.is_training:
            x = self.drop_out(x)
        x = self.relu(self.fc2(x))
        if self.is_training:
            x = self.drop_out(x)
        return self.fc3(x)


def train():
    pass


def test():
    pass


def main():
    pass


if "__main__" == __name__:
    main()






























