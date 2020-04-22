# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2019/12/25

import numpy as np
import random
import torch

from PIL import Image, ImageOps, ImageFilter


class FixedResize(object):
    def __call__(self, image_data):
        raise NotImplementedError()


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image_data):
        img_w, img_h = image_data.size
        if img_w > img_h:
            oh = self.crop_size
            ow = int(1.0 * img_w * oh / img_h)
        else:
            ow = self.crop_size
            oh = int(1.0 * img_h * ow / img_w)

        image_data = image_data.resize((ow, oh), Image.BILINEAR)

        # left top coordinate
        img_w, img_h = image_data.size
        x1 = int(round((img_w - self.crop_size) / 2.))
        y1 = int(round((img_h - self.crop_size) / 2.))

        image_data = image_data.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return image_data


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

        if len(self.mean) != len(self.std):
            raise ValueError("mean and std should have same length")

    def __call__(self, image_data):
        image_data = np.array(image_data).astype(np.float32)

        assert np.shape(image_data)[2] == len(self.mean)
        image_data = (image_data / 255.0 - self.mean) / self.std

        return image_data


class RandomGaussianBlur(object):
    def __call__(self, image_data):
        if random.random() < 0.5:
            image_data = image_data.filter(ImageFilter.GaussianBlur(radius=random.randint(1, 3)))

        return image_data


class RandomHorizontalFlip(object):
    def __call__(self, image_data):
        if random.random() < 0.5:
            image_data = image_data.transpose(Image.FLIP_LEFT_RIGHT)

        return image_data


class RandomRotate(object):
    def __init__(self, degree_range):
        self.degree_range_pos = degree_range
        self.degree_range_neg = -1 * degree_range

    def __call__(self, image_data):
        rotate_degree = random.uniform(self.degree_range_neg, self.degree_range_pos)

        image_data = image_data.rotate(rotate_degree, Image.BILINEAR)

        return image_data


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, image_data):
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        img_w, img_h = image_data.size

        if img_h > img_w:
            ow = short_size
            oh = int(1.0 * img_h * ow / img_w)
        else:
            oh = short_size
            ow = int(1.0 * img_w * oh / img_h)

        image_data = image_data.resize((ow, oh), Image.BILINEAR)

        # pad crop
        if short_size < self.crop_size:
            pad_h = self.crop_size - oh if oh < self.crop_size else 0
            pad_w = self.crop_size - ow if ow < self.crop_size else 0
            image_data = ImageOps.expand(image_data, border=(0, 0, pad_w, pad_h), fill=0)

        # random crop crop_size
        img_w, img_h = image_data.size
        x1 = random.randint(0, img_w - self.crop_size)
        y1 = random.randint(0, img_h - self.crop_size)
        image_data = image_data.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return image_data


class ToTensor(object):
    def __call__(self, image_data):
        # swap color axis
        # numpy: H x W x C
        # torch: C x H x W
        image_data = np.array(image_data).astype(np.float32).transpose((2, 0, 1))

        image_data = torch.from_numpy(image_data).float()

        return image_data



























