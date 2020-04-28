# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2019/12/25

# https://www.cnblogs.com/Mrzhang3389/p/10127071.html
# https://blog.csdn.net/DL_CreepingBird/article/details/78574059
# https://blog.csdn.net/taigw/article/details/51401448
# https://zhuanlan.zhihu.com/p/32506912

from __future__ import absolute_import, print_function

import math
import torch.nn as nn

from dataloader import make_data_loader


dict_model_config = {
    'A': ((1, 3, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512)),
    'B': ((2, 3, 64), (2, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512)),
    'D': ((2, 3, 64), (2, 64, 128), (3, 128, 256), (3, 256, 512), (3, 512, 512)),
    'E': ((2, 3, 64), (2, 64, 128), (4, 128, 256), (4, 256, 512), (4, 512, 512)),
}


def vgg_conv_block(num_conv, in_channel, out_channel, with_polling=True, batch_norm=False):
    """
    define vgg block
    :param num_conv:
    :param in_channel:
    :param out_channel:
    :param with_polling:
    :param batch_norm:
    :return:
    """
    net = [nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1), nn.ReLU(True)]

    for conv_idx in range(1, num_conv):
        net.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        if batch_norm:
            net.append(nn.BatchNorm2d(out_channel))
        net.append(nn.ReLU(True))

    if with_polling:
        net.append(nn.MaxPool2d(2, 2, padding=1))

    return nn.Sequential(*net)


def vgg_classifier_block(num_class=1000, is_training=False):
    net = [nn.Linear(512*7*7, 4096), nn.ReLU(True)]
    if is_training:
        net.append(nn.Dropout())

    net.extend([nn.Linear(4096, 4096), nn.ReLU(True)])
    if is_training:
        net.append(nn.Dropout())

    net.append(nn.Linear(4096, num_class))

    return nn.Sequential(*net)


class VisualGeometryGroup(nn.Module):
    def __init__(self, layer_config, num_class=1000, with_bn=False, is_training=False, with_fc=True, init_weights=True, verbose=False):
        super(VisualGeometryGroup, self).__init__()

        self.layer_config = layer_config
        self.num_class = num_class
        self.with_bn = with_bn
        self.is_training = is_training
        self.with_fc = with_fc
        self.init_weights = init_weights
        self.verbose = verbose

        if self.with_fc:
            self.ranges = None
        else:
            self.ranges = []
            layer_beg = 0
            for v in self.layer_config:
                layer_end = v[0] * 2 + 1
                # layer_end = layer_beg + v[0] * 2 + 1

                if len(v) == 4 and not v[3]:
                    layer_end -= 1

                self.ranges.append((layer_beg, layer_end))
                # layer_beg = layer_end

            if self.verbose:
                print(self.ranges)

        self.model = self._make_layers(self.layer_config,
                                       num_class=self.num_class,
                                       batch_norm=self.with_bn,
                                       is_training=self.is_training,
                                       with_fc=self.with_fc)
        if verbose:
            print(self.model)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        if self.with_fc:
            return self.model(x)
        else:
            output = {}
            for block_idx, (layer_beg, layer_end) in enumerate(self.ranges):
                for layer_idx in range(layer_beg, layer_end):
                    x = self.model[block_idx][layer_idx](x)
                output["x%d" % (block_idx + 1)] = x
        return output

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                num_feature = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
                layer.weight.data.normal_(0, math.sqrt(2. / num_feature))

                if layer.bias is not None:
                    layer.bias.data.zero_()
                elif isinstance(layer, nn.BatchNorm2d):
                    layer.weight.data.fill_(1)
                    layer.bias.data.zero_()
                elif isinstance(layer, nn.Linear):
                    layer.weight.data.normal_(0, 0.01)
                    layer.bias.data.zero_()

    @staticmethod
    def _make_layers(layer_config, num_class=1000, batch_norm=False, is_training=False, with_fc=True):
        net = []

        for v in layer_config:
            net.append(vgg_conv_block(*v, batch_norm=batch_norm))

        if with_fc:
            net.append(vgg_classifier_block(num_class, is_training))

        return nn.Sequential(*net)


def vgg_backbone(**kwargs):
    layer_config = dict_model_config['D']
    return VisualGeometryGroup(layer_config, **kwargs)


def main():
    kwargs = {
        # "with_bn": True,
        "with_fc": False,
        "verbose": True
    }
    model = VisualGeometryGroup(dict_model_config["D"], **kwargs)


if "__main__" == __name__:
    main()



