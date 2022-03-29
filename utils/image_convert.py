# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2020/12/21 14:05

from __future__ import absolute_import

import cv2
import torch
import torchvision


def save_image_in_tensor(input_tensor: torch.Tensor, filename):
    """
    save tensor to file
    :param input_tensor: tensor to save
    :param filename:
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))

    torchvision.utils.save_image(input_tensor, filename)


def convert_image_tensor_to_cv2(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)



def save_image_tensor_to_pillow(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为pillow
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()

    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为numpy
    input_tensor = input_tensor.permute((1, 2, 0)).mul(255).add_(0.5).clamp_(0, 255).type(torch.uint8).numpy()
    # 转成pillow
    im = Image.fromarray(input_tensor)
    im.save(filename)


def convert_img_to_str(image_array, encoding="utf-8"):
    import base64
    import io
    im = Image.fromarray(image_array.astype("uint8"))
    raw_byte = io.BytesIO()
    im.save(raw_byte, "PNG")
    raw_byte.seek(0)
    return base64.b64encode(raw_byte.read()).decode(encoding)


class ImageConverter(object):
    def __init__(self):
        pass

    @staticmethod
    def tensor_to_numpy(input_tensor: torch.Tensor):
        assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
        # make a copy to cpu
        input_tensor = input_tensor.clone().detach().to(torch.device('cpu'))
        # squeeze off first dimension as batch
        input_tensor = input_tensor.squeeze()

        # [0, 1] to [0,255] of numpy array
        nd_array = input_tensor.permute((1, 2, 0)).mul(255).add_(0.5).clamp_(0, 255).type(torch.uint8).numpy()

        return nd_array



















