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
    def __call__(self, sample):
        raise NotImplementedError()


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        _img = sample["image"]
        _mask = sample["label"]

        img_w, img_h = _img.size
        if img_w > img_h:
            oh = self.crop_size
            ow = int(1.0 * img_w * oh / img_h)
        else:
            ow = self.crop_size
            oh = int(1.0 * img_h * ow / img_w)

        _img = _img.resize((ow, oh), Image.BILINEAR)
        _mask = _mask.resize((ow, oh), Image.NEAREST)

        # left top coordinate
        img_w, img_h = _img.size
        x1 = int(round((img_w - self.crop_size) / 2.))
        y1 = int(round((img_h - self.crop_size) / 2.))

        _img = _img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        _mask = _mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': _img, 'label': _mask}


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

        if len(self.mean) != len(self.std):
            raise ValueError("mean and std should have same length")

    def __call__(self, sample):
        _img = np.array(sample["image"]).astype(np.float32)
        _mask = np.array(sample["label"]).astype(np.float32)

        assert np.shape(_img)[2] == len(self.mean)
        _img = (_img / 255.0 - self.mean) / self.std

        return {"image": _img, "label": _mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        _img = sample["image"]
        _mask = sample["label"]

        if random.random() < 0.5:
            _img = _img.filter(ImageFilter.GaussianBlur(radius=random.randint(1, 3)))

        return {"image": _img, "label": _mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        _img = sample["image"]
        _mask = sample["label"]

        if random.random() < 0.5:
            _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
            _mask = _mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {"image": _img, "label": _mask}


class RandomRotate(object):
    def __init__(self, degree_range):
        self.degree_range_pos = degree_range
        self.degree_range_neg = -1 * degree_range

    def __call__(self, sample):
        _img = sample["image"]
        _mask = sample["label"]

        rotate_degree = random.uniform(self.degree_range_neg, self.degree_range_pos)

        _img = _img.rotate(rotate_degree, Image.BILINEAR)
        _mask = _mask.rotate(rotate_degree, Image.NEAREST)

        return {"image": _img, "label": _mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        _img = sample["image"]
        _mask = sample["label"]

        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        img_w, img_h = _img.size

        if img_h > img_w:
            ow = short_size
            oh = int(1.0 * img_h * ow / img_w)
        else:
            oh = short_size
            ow = int(1.0 * img_w * oh / img_h)

        _img = _img.resize((ow, oh), Image.BILINEAR)
        _mask = _mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < self.crop_size:
            pad_h = self.crop_size - oh if oh < self.crop_size else 0
            pad_w = self.crop_size - ow if ow < self.crop_size else 0
            _img = ImageOps.expand(_img, border=(0, 0, pad_w, pad_h), fill=0)
            _mask = ImageOps.expand(_mask, border=(0, 0, pad_w, pad_h), fill=self.fill)

        # random crop crop_size
        img_w, img_h = _img.size
        x1 = random.randint(0, img_w - self.crop_size)
        y1 = random.randint(0, img_h - self.crop_size)
        _img = _img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        _mask = _mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {"image": _img, "label": _mask}


class ToTensor(object):
    def __call__(self, sample):
        # swap color axis
        # numpy: H x W x C
        # torch: C x H x W
        _img = np.array(sample["image"]).astype(np.float32).transpose((2, 0, 1))
        _mask = np.array(sample["label"]).astype(np.float32)

        _img = torch.from_numpy(_img).float()
        _mask = torch.from_numpy(_mask).float()

        return {"image": _img, "label": _mask}



























