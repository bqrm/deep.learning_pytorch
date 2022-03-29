# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2019/12/25

import numpy as np
import os
import torch.utils.data as data

import dataloader.custom_transforms_with_image_label as tr

from PIL import Image

from torchvision import transforms


class PascalVocSeg(data.Dataset):
    num_class = 21

    def __init__(self, root_dir, crop_size, split="train"):
        if not os.path.exists(root_dir):
            raise ValueError("%s not exists" % os.path.realpath(root_dir))

        self.crop_size = crop_size

        self._image_dir = os.path.join(root_dir, "JPEGImages")
        self._label_dir = os.path.join(root_dir, "SegmentationClass")
        self._split_dir = os.path.join(root_dir, "ImageSets", "Segmentation")

        if isinstance(split, str):
            self._categories = [split]
        else:
            self._categories = split.sort()

        self.image_names = []
        self.label_names = []
        self.sample_ids = []

        for set_split in self._categories:
            with open(os.path.join(self._split_dir, set_split + ".txt"), "r") as f:
                list_sample_name = f.read().splitlines()

            for sample_name in list_sample_name:
                image_name = os.path.join(self._image_dir, sample_name + ".jpg")
                label_name = os.path.join(self._label_dir, sample_name + ".png")
                assert os.path.isfile(image_name)
                assert os.path.isfile(label_name)
                self.image_names.append(image_name)
                self.label_names.append(label_name)
                self.sample_ids.append(sample_name)

        assert len(self.image_names) == len(self.label_names) == len(self.sample_ids)

    def __getitem__(self, index):
        _image = Image.open(self.image_names[index]).convert("RGB")
        _label = Image.open(self.label_names[index])
        _sample_id = self.sample_ids[index]

        tmp = {"image": _image, "label": _label, "id": _sample_id}

        for split in self._categories:
            sample = None
            if "train" == split:
                sample = self._transform_train(tmp)
            elif "val" == split:
                sample = self._transform_valid(tmp)

            if sample is None:
                return None
            else:
                sample["id"] = _sample_id
                return sample

        raise NotImplementedError

    def __len__(self):
        return len(self.image_names)

    def __str__(self):
        return "pascal_voc_seg"

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
    root_path = "D:/Projects/DataSets/VOCdevkit/VOC2012"

    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    import matplotlib
    matplotlib.use("qt5agg")

    batch_size = 5

    voc_train = PascalVocSeg(root_path, split="train")

    kwargs = {'num_workers': 0, 'pin_memory': True}
    data_loader = DataLoader(voc_train, batch_size=batch_size, shuffle=True, **kwargs)

    for batch_idx, sample in enumerate(data_loader):
        batch_image = sample["image"].numpy()
        batch_label = sample["label"].numpy()
        batch_id = sample["id"]
        for sample_idx in range(batch_size):
            img = np.transpose(batch_image[sample_idx], (1, 2, 0))
            gt = np.array(batch_label[sample_idx]).astype(np.uint8)

            img = ((img * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255.0).astype(np.uint8)

            plt.figure()
            plt.suptitle(batch_id[sample_idx])
            plt.subplot(121)
            plt.imshow(img)
            plt.subplot(122)
            plt.imshow(gt)
            plt.show()


if "__main__" == __name__:
    main()


























