# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2020/2/26 20:18

from __future__ import absolute_import

import os
import shutil
import torch.utils.data as data

import dataloader.custom_transforms_with_class_label as tr

from PIL import Image

from torchvision import transforms

from utils.common_method import get_filename_iteratively


class MiniImageNet(data.Dataset):
    def __init__(self, root_dir, crop_size, split="train"):
        if not os.path.exists(root_dir):
            raise ValueError("%s not exists" % os.path.realpath(root_dir))

        self.crop_size = crop_size

        self._image_dir = os.path.join(root_dir, "images")
        self._split_dir = os.path.join(root_dir, "segmentation")

        if isinstance(split, str):
            self._categories = [split]
        else:
            self._categories = split

        self.image_names = []
        self.image_labels = []
        self.image_ids = []
        label_dict = {}
        label_index = 0

        for set_split in self._categories:
            split_name_file = os.path.join(self._split_dir, set_split + ".txt")
            if not os.path.exists(split_name_file):
                raise ValueError("%s does not exists" % split_name_file)
            with open(split_name_file, "r") as f:
                list_sample_name = f.read().splitlines()

            for sample_base_name in list_sample_name:
                image_name = os.path.join(self._image_dir, sample_base_name + ".jpg")
                assert os.path.isfile(image_name)
                class_name = sample_base_name[:9]
                if class_name not in label_dict:
                    label_dict[class_name] = label_index
                    label_index += 1
                self.image_names.append(image_name)
                self.image_labels.append(label_dict[class_name])
                self.image_ids.append(sample_base_name)

        assert len(self.image_names) == len(self.image_labels) == len(self.image_ids)
        self.num_class = label_index

    def __getitem__(self, index):
        _image = Image.open(self.image_names[index]).convert("RGB")

        for split in self._categories:
            if "train" == split:
                _image = self._transform_train(_image)
            elif split in ["valid", "test"]:
                _image = self._transform_valid(_image)

            sample = {"image": _image, "label": self.image_labels[index], "id": self.image_ids[index]}

            return sample

    def __len__(self):
        return len(self.image_names)

    def __str__(self):
        return "mini_image_net"

    def _transform_train(self, image):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=512, crop_size=self.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor(),
        ])
        return composed_transforms(image)

    def _transform_valid(self, image):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor(),
        ])
        return composed_transforms(image)


def main():
    root_path = "D:/Projects/DataSets/mini_image_net"

    image_net_loader = MiniImageNet(root_path, 224)


if "__main__" == __name__:
    main()


























