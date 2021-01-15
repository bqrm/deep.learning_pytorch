# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2019/12/17

import base64
import cv2
import numpy

from PIL import Image


def base64_to_image(base64_code):
    """
    convert base6 to rgb
    :param base64_code:
    :return:
    """
    img_data = base64.b64decode(base64_code)
    img_array = numpy.fromstring(img_data, numpy.uint8)
    img_rgb = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return img_rgb


def read_cv(image_name, convert_gray=False, reorder_rgb=True):
    """
    read image using OpenCV
    :param image_name:
    :param convert_gray:
    :param reorder_rgb:
    :return:
    """
    img_data = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
    img_shape = numpy.shape(img_data)

    if 2 == len(img_shape):
        return img_data
    elif 3 == len(img_shape) and 1 == img_shape[2]:
        return img_data[:, :, 0]
    else:
        if convert_gray:
            return cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        else:
            if reorder_rgb:
                return img_data[:, :, (2, 1, 0)]
            else:
                return img_data


def read_pil(image_name, convert_gray=False):
    """
    read image using PIL
    :param image_name:
    :param convert_gray:
    :return:
    """

    if convert_gray:
        # L = 0.299R + 0.587G + 0.114B
        return numpy.array(Image.open(image_name).convert("L"), "uint8")
    else:
        return numpy.array(Image.open(image_name))






























